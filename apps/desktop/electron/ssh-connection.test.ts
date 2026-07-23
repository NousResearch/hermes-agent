import assert from 'node:assert/strict'
import { EventEmitter } from 'node:events'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { test } from 'vitest'

import {
  baseSshOptions,
  buildControlArgs,
  buildExecArgs,
  buildInteractiveSshArgs,
  buildMasterArgs,
  classifySshError,
  controlSocketPath,
  createSshProbeConnection,
  forwardSpec,
  hostArgs,
  redactSecrets,
  runSsh,
  SSH_ERROR,
  SshConnection,
  sshErrorMessage,
  stopTunnelChild,
  target,
  validateSshTarget
} from './ssh-connection'

test('redactSecrets scrubs the spawn-time session token env var', () => {
  const line = 'setsid env HERMES_DASHBOARD_SESSION_TOKEN=abc123deadbeef HERMES_DESKTOP=1 hermes dashboard'
  const out = redactSecrets(line)
  assert.ok(!out.includes('abc123deadbeef'))
  assert.match(out, /HERMES_DASHBOARD_SESSION_TOKEN=<redacted>/)
  // non-secret env vars are preserved
  assert.match(out, /HERMES_DESKTOP=1/)
})

test('redactSecrets scrubs ?token= and ?ticket= URL params', () => {
  assert.match(redactSecrets('ws://127.0.0.1:5000/api/ws?token=supersecret'), /\?token=<redacted>/)
  assert.match(redactSecrets('ws://127.0.0.1:5000/api/ws?ticket=onetimeticket'), /\?ticket=<redacted>/)
  assert.match(redactSecrets('GET /x?a=1&token=zzz HTTP'), /&token=<redacted>/)
  assert.ok(!redactSecrets('?token=supersecret').includes('supersecret'))
})

test('redactSecrets scrubs Authorization and X-Hermes-Session-Token headers', () => {
  assert.match(redactSecrets('Authorization: Bearer tok_9999'), /Authorization: Bearer <redacted>/)
  assert.ok(!redactSecrets('Authorization: Bearer tok_9999').includes('tok_9999'))
  assert.match(redactSecrets('X-Hermes-Session-Token: hdr_888'), /X-Hermes-Session-Token: ?<redacted>/)
  assert.ok(!redactSecrets('X-Hermes-Session-Token: hdr_888').includes('hdr_888'))
})

test('redactSecrets handles null/undefined and non-secret text untouched', () => {
  assert.equal(redactSecrets(null), '')
  assert.equal(redactSecrets(undefined), '')
  assert.equal(redactSecrets('uname -s -m'), 'uname -s -m')
})

test('controlSocketPath is stable, short, and host-distinct', () => {
  const a = controlSocketPath('me', 'box1', 22, '/tmp/d')
  const a2 = controlSocketPath('me', 'box1', 22, '/tmp/d')
  const b = controlSocketPath('me', 'box2', 22, '/tmp/d')
  assert.equal(a, a2, 'same triple → same socket (ControlMaster reuse)')
  assert.notEqual(a, b, 'different host → different socket')
  // 16 hex chars + .sock keeps the basename short for sun_path 104-byte limit
  assert.match(a, /\/[0-9a-f]{16}\.sock$/)
})

test('controlSocketPath default base stays under sun_path even with the temp-listener suffix', () => {
  // OpenSSH binds a temporary listener at `<ControlPath>.<16 random chars>` (a
  // 17-byte suffix) while opening the master. The macOS regression was the
  // default base under os.tmpdir() (/var/folders/.../T/) pushing it over 104.
  const p = controlSocketPath('hermes', 'remote-build-server', 22) // no baseDir → default
  const worstCase = `${p}.0123456789abcdef` // mimic the .<16-char> temp suffix
  assert.ok(
    worstCase.length <= 104,
    `default control socket + temp suffix must fit sun_path (got ${worstCase.length}: ${worstCase})`
  )
  // And it must NOT live under the deeply-nested macOS per-user temp dir.
  assert.ok(!p.includes('/var/folders/'), 'default base must not be os.tmpdir() on macOS')
})

test('baseSshOptions carries the house ControlMaster/BatchMode/accept-new policy', () => {
  const opts = baseSshOptions('/tmp/x.sock', 15000)
  const joined = opts.join(' ')
  assert.match(joined, /ControlPath=\/tmp\/x\.sock/)
  assert.match(joined, /ControlMaster=auto/)
  assert.match(joined, /ControlPersist=\d+/)
  assert.match(joined, /BatchMode=yes/)
  assert.match(joined, /StrictHostKeyChecking=accept-new/)
  assert.match(joined, /ExitOnForwardFailure=yes/)
  assert.match(joined, /ConnectTimeout=15/)
  assert.ok(!joined.includes('StrictHostKeyChecking=no'), 'never disables host-key checking')
})

test('hostArgs adds -p only for non-default port and -i only with a key', () => {
  assert.deepEqual(hostArgs({ port: 22 }), [])
  assert.deepEqual(hostArgs({ port: 2222 }), ['-p', '2222'])
  assert.deepEqual(hostArgs({ port: 22, keyPath: '/k' }), ['-i', '/k'])
  assert.deepEqual(hostArgs({ port: 2200, keyPath: '/k' }), ['-p', '2200', '-i', '/k'])
})

test('target builds user@host or bare host', () => {
  assert.equal(target('me', 'box'), 'me@box')
  assert.equal(target('', 'box'), 'box')
})

test('buildExecArgs ends with host then the remote command', () => {
  const conn = { user: 'me', host: 'box', port: 22, keyPath: '', controlPath: '/tmp/x.sock' }
  const args = buildExecArgs(conn, 'command -v hermes', 15000)
  assert.equal(args[args.length - 1], 'command -v hermes')
  assert.equal(args[args.length - 2], 'me@box')
  assert.ok(args.includes('BatchMode=yes'))
})

test('buildControlArgs places -O <op> first and never appends a remote command', () => {
  const conn = { user: 'me', host: 'box', port: 2222, keyPath: '/k', controlPath: '/tmp/x.sock' }
  const args = buildControlArgs(conn, 'forward', ['-L', forwardSpec(5000, 6000)], 15000)
  assert.equal(args[0], '-O')
  assert.equal(args[1], 'forward')
  assert.ok(args.includes('-L'))
  assert.ok(args.includes('127.0.0.1:5000:127.0.0.1:6000'))
  assert.equal(args[args.length - 1], 'me@box')
})

test('buildMasterArgs requests a backgrounded master (-M -N -f)', () => {
  const conn = { user: 'me', host: 'box', port: 22, keyPath: '', controlPath: '/tmp/x.sock' }
  const args = buildMasterArgs(conn, 15000)
  assert.ok(args.includes('-M'))
  assert.ok(args.includes('-N'))
  assert.ok(args.includes('-f'))
})

test('forwardSpec binds the local end to 127.0.0.1 only', () => {
  assert.equal(forwardSpec(5000, 6000), '127.0.0.1:5000:127.0.0.1:6000')
  assert.ok(forwardSpec(5000, 6000).startsWith('127.0.0.1:'))
  assert.ok(!forwardSpec(5000, 6000).startsWith('0.0.0.0'))
})

test('buildInteractiveSshArgs requests a PTY, reuses the control master, execs a login shell', () => {
  const conn = { user: 'me', host: 'box', port: 22, keyPath: '', controlPath: '/tmp/x.sock' }
  const args = buildInteractiveSshArgs(conn, '', 15000)
  assert.equal(args[0], '-tt', 'forces a PTY so the remote sees a real terminal')
  assert.ok(args.join(' ').includes('ControlPath=/tmp/x.sock'), 'reuses the existing master (no new auth)')
  assert.equal(args[args.length - 2], 'me@box')
  assert.equal(args[args.length - 1], 'exec "$SHELL" -l')
})

test('buildInteractiveSshArgs cds into the remote cwd (best-effort) before the shell', () => {
  const conn = { user: 'me', host: 'box', port: 22, keyPath: '', controlPath: '/tmp/x.sock' }
  const args = buildInteractiveSshArgs(conn, '/home/me/project', 15000)
  const remoteCmd = args[args.length - 1]
  assert.match(remoteCmd, /^cd '\/home\/me\/project' 2>\/dev\/null; exec "\$SHELL" -l$/)
})

test('buildInteractiveSshArgs single-quotes a cwd with quotes safely', () => {
  const conn = { user: 'me', host: 'box', port: 22, keyPath: '', controlPath: '/tmp/x.sock' }
  const args = buildInteractiveSshArgs(conn, "/tmp/a'b", 15000)
  // the embedded quote must be escaped, not break out of the quoting
  assert.ok(args[args.length - 1].startsWith("cd '/tmp/a'"))
  assert.ok(args[args.length - 1].includes('exec "$SHELL" -l'))
})

test('classifySshError detects a changed host key (fail-closed)', () => {
  assert.equal(
    classifySshError('@@@@ WARNING: REMOTE HOST IDENTIFICATION HAS CHANGED! @@@@'),
    SSH_ERROR.HOST_KEY_CHANGED
  )
  assert.equal(classifySshError('Host key verification failed.'), SSH_ERROR.HOST_KEY_CHANGED)
  assert.equal(classifySshError('Offending ECDSA key in /home/u/.ssh/known_hosts:5'), SSH_ERROR.HOST_KEY_CHANGED)
})

test('classifySshError detects auth failure', () => {
  assert.equal(classifySshError('Permission denied (publickey).'), SSH_ERROR.AUTH_FAILED)
  assert.equal(classifySshError('Too many authentication failures'), SSH_ERROR.AUTH_FAILED)
})

test('classifySshError detects unreachable', () => {
  assert.equal(classifySshError('ssh: Could not resolve hostname nope'), SSH_ERROR.UNREACHABLE)
  assert.equal(classifySshError('connect to host x port 22: Connection refused'), SSH_ERROR.UNREACHABLE)
})

test('sshErrorMessage gives actionable guidance for auth and host-key-change', () => {
  const conn = { user: 'me', host: 'box', port: 22 }
  assert.match(sshErrorMessage(SSH_ERROR.AUTH_FAILED, conn, 'Permission denied'), /ssh-agent|ssh-add|IdentityFile/)
  assert.match(sshErrorMessage(SSH_ERROR.HOST_KEY_CHANGED, conn, 'CHANGED'), /ssh-keygen -R box/)
})

// A fake child process that emits a scripted result on next tick.
function fakeChild({ code = 0, stdout = '', stderr = '', errorEvent = null, hang = false }: any = {}) {
  const child: any = new EventEmitter()
  child.stdout = new EventEmitter()
  child.stderr = new EventEmitter()

  child.kill = () => {
    child._killed = true
  }

  if (hang) {
    return child // never emits close → drives the timeout path
  }

  process.nextTick(() => {
    if (errorEvent) {
      child.emit('error', errorEvent)

      return
    }

    if (stdout) {
      child.stdout.emit('data', Buffer.from(stdout))
    }

    if (stderr) {
      child.stderr.emit('data', Buffer.from(stderr))
    }

    child.emit('close', code)
  })

  return child
}

// Build a spawnFn that returns scripted children per ssh invocation, recording
// the args it was called with.
function scriptedSpawn(scripts) {
  const calls: any[] = []
  let i = 0

  const fn: any = (_cmd, args) => {
    calls.push(args)
    const script = typeof scripts === 'function' ? scripts(args, i) : scripts[Math.min(i, scripts.length - 1)]
    i += 1

    return fakeChild(script || {})
  }

  fn.calls = calls

  return fn
}

test('open() establishes the master when not already alive', async () => {
  // `-O check` fails first (not alive) → master opens (code 0). Track which
  // ssh ops ran rather than re-probing with the same always-failing check.
  const ops: string[] = []

  const spawnFn = scriptedSpawn(args => {
    ops.push(args.includes('check') ? 'check' : args.includes('-M') ? 'master' : 'other')

    if (args.includes('check')) {
      return { code: 255, stderr: 'no control path' }
    }

    return { code: 0 }
  })

  const conn = new SshConnection({ host: 'box', user: 'me' }, { spawnFn, controlDir: '/tmp/d' })
  await conn.open()
  assert.deepEqual(ops, ['check', 'master'], 'probes liveness first, then opens the master')
})

test('open() is a no-op when the master is already alive and execs verify', async () => {
  const ops: string[] = []

  const spawnFn = scriptedSpawn(args => {
    ops.push(args.includes('check') ? 'check' : args.includes('exit 0') ? 'verify' : 'master')

    return { code: 0 } // check succeeds → alive; verify exec succeeds → trusted
  })

  const conn = new SshConnection({ host: 'box', user: 'me' }, { spawnFn, controlDir: '/tmp/d' })
  await conn.open()
  assert.deepEqual(ops, ['check', 'verify'], 'alive master is exec-verified, then trusted without reopening')
})

test('open() evicts a wedged master (check passes, exec hangs) and dials fresh', async () => {
  // The macOS mode-switch wedge: ControlPersist master answers -O check but
  // every exec through it hangs. open() must verify, evict (-O exit), and
  // establish a fresh master instead of trusting the corpse.
  const ops: string[] = []

  const spawnFn = scriptedSpawn(args => {
    if (args.includes('check')) {
      ops.push('check')

      return { code: 0 }
    }

    if (args.includes('exit 0')) {
      ops.push('verify')

      return { hang: true }
    }

    if (args.includes('-O')) {
      ops.push('evict')

      return { code: 0 }
    }

    ops.push('master')

    return { code: 0 }
  })

  const conn = new SshConnection({ host: 'box', user: 'me' }, { spawnFn, controlDir: '/tmp/d', connectTimeoutMs: 50 })

  await conn.open()
  assert.deepEqual(
    ops,
    ['check', 'verify', 'evict', 'master'],
    'wedged master: verified, evicted, then a fresh master is dialed'
  )
})

test('close() removes the control socket when -O exit fails', async () => {
  const dir = path.join(os.tmpdir(), `hermes-ssh-close-${process.pid}-${Date.now()}`)
  fs.mkdirSync(dir, { recursive: true, mode: 0o700 })

  const spawnFn = scriptedSpawn(args => {
    if (args.includes('check')) {
      return { code: 255 }
    } // not alive → open dials master

    if (args.includes('-M')) {
      return { code: 0 }
    }

    return { code: 255, stderr: 'mux: master gone' } // -O exit fails
  })

  const conn = new SshConnection({ host: 'box', user: 'me' }, { spawnFn, controlDir: dir })
  await conn.open()
  fs.writeFileSync(conn.controlPath, '') // simulate the lingering socket file
  await conn.close()
  assert.ok(!fs.existsSync(conn.controlPath), 'failed -O exit drops the socket so the next open dials fresh')
  fs.rmSync(dir, { recursive: true, force: true })
})

test('open() creates the control-socket directory if it does not exist', async () => {
  const dir = path.join(os.tmpdir(), `hermes-ssh-test-${process.pid}-${Date.now()}`)
  assert.ok(!fs.existsSync(dir), 'precondition: control dir absent')
  const spawnFn = scriptedSpawn(args => (args.includes('check') ? { code: 255 } : { code: 0 }))
  const conn = new SshConnection({ host: 'box', user: 'me' }, { spawnFn, controlDir: dir })

  try {
    await conn.open()
    assert.ok(fs.existsSync(dir), 'open() created the control-socket directory before spawning ssh')
  } finally {
    try {
      fs.rmSync(dir, { recursive: true, force: true })
    } catch {
      /* ignore */
    }
  }
})

test('open() surfaces a classified auth error', async () => {
  const spawnFn = scriptedSpawn(args => {
    if (args.includes('check')) {
      return { code: 255 }
    }

    return { code: 255, stderr: 'Permission denied (publickey).' }
  })

  const conn = new SshConnection({ host: 'box', user: 'me' }, { spawnFn, controlDir: '/tmp/d' })
  await assert.rejects(
    () => conn.open(),
    (err: any) => {
      assert.equal(err.kind, SSH_ERROR.AUTH_FAILED)
      assert.match(err.message, /ssh-agent|ssh-add/)

      return true
    }
  )
})

test('exec() returns stdout on success and rejects (classified) on failure', async () => {
  const okSpawn = scriptedSpawn([{ code: 0, stdout: 'Linux\n' }])
  const conn = new SshConnection({ host: 'box', user: 'me' }, { spawnFn: okSpawn, controlDir: '/tmp/d' })
  assert.equal((await conn.exec('uname -s')).trim(), 'Linux')

  const failSpawn = scriptedSpawn([{ code: 1, stderr: 'ssh: Could not resolve hostname box' }])
  const conn2 = new SshConnection({ host: 'box', user: 'me' }, { spawnFn: failSpawn, controlDir: '/tmp/d' })
  await assert.rejects(
    () => conn2.exec('uname -s'),
    (err: any) => {
      assert.equal(err.kind, SSH_ERROR.UNREACHABLE)

      return true
    }
  )
})

test('exec() treats a hung ssh as a timeout (half-open connection)', async () => {
  const spawnFn = scriptedSpawn([{ hang: true }])
  const conn = new SshConnection({ host: 'box', user: 'me' }, { spawnFn, controlDir: '/tmp/d' })
  await assert.rejects(
    () => conn.exec('uname -s', { timeoutMs: 30 }),
    (err: any) => {
      assert.equal(err.kind, SSH_ERROR.TIMEOUT)

      return true
    }
  )
})

test('forward() issues -O forward with a loopback-bound -L spec', async () => {
  const spawnFn = scriptedSpawn([{ code: 0 }])
  const conn = new SshConnection({ host: 'box', user: 'me' }, { spawnFn, controlDir: '/tmp/d' })
  await conn.forward(5000, 6000)
  const args = spawnFn.calls[0]
  assert.equal(args[0], '-O')
  assert.equal(args[1], 'forward')
  assert.ok(args.includes('127.0.0.1:5000:127.0.0.1:6000'))
})

test('lifecycle logging passes through redaction', async () => {
  const logs: string[] = []
  const spawnFn = scriptedSpawn(args => (args.includes('check') ? { code: 255 } : { code: 0 }))

  const conn = new SshConnection(
    { host: 'box', user: 'me' },
    { spawnFn, controlDir: '/tmp/d', rememberLog: l => logs.push(l) }
  )

  await conn.open()

  // none of the emitted log lines may carry a raw token-shaped secret
  for (const line of logs) {
    assert.ok(!/token=[^<]/.test(line))
  }

  assert.ok(logs.some(l => l.includes('[ssh]')))
})

test('no-mux: ssh args carry no ControlMaster/ControlPath options', async () => {
  const spawnFn = scriptedSpawn({ code: 0 })
  const conn = new SshConnection({ host: 'box', user: 'me' }, { spawnFn, mux: false })
  await conn.open()

  for (const args of spawnFn.calls) {
    assert.ok(!args.some(a => /ControlMaster|ControlPath|ControlPersist/.test(a)), `mux option leaked: ${args}`)
  }
})

test('no-mux: open() verifies auth with a one-shot exec, no -M master', async () => {
  const spawnFn = scriptedSpawn({ code: 0 })
  const conn = new SshConnection({ host: 'box', user: 'me' }, { spawnFn, mux: false })
  await conn.open()
  assert.ok(!spawnFn.calls.some(args => args.includes('-M')), 'no master should be spawned')
  assert.ok(
    spawnFn.calls.some(args => args[args.length - 1] === 'exit 0'),
    'liveness/openness via one-shot exec'
  )
})

test('SSH probe never creates or closes a ControlMaster', async () => {
  const spawnFn = scriptedSpawn({ code: 0 })
  const conn = createSshProbeConnection({ host: 'box', user: 'me' }, { spawnFn })
  await conn.open()
  await conn.close()
  const args = spawnFn.calls.flat()
  assert.ok(!args.includes('-M'))
  assert.ok(!args.includes('-O'))
  assert.ok(!args.some(value => /Control(?:Master|Path|Persist)/.test(value)))
})

test('no-mux: open() classifies auth failure', async () => {
  const spawnFn = scriptedSpawn([{ code: 255, stderr: 'me@box: Permission denied (publickey).' }])
  const conn = new SshConnection({ host: 'box', user: 'me' }, { spawnFn, mux: false })
  await assert.rejects(conn.open(), (err: any) => err.kind === 'auth-failed')
})

test('no-mux: forward spawns a persistent -N -L child; cancel + close kill it', async () => {
  // Real listener stands in for the tunnel's local end so waitForLocalPort sees it.
  const net = await import('node:net')
  const srv = net.createServer()
  await new Promise<void>(r => srv.listen(0, '127.0.0.1', () => r()))
  const localPort = (srv.address() as any).port
  const tunnels: any[] = []

  const spawnFn: any = (_cmd, args) => {
    const child: any = new EventEmitter()
    child.stderr = new EventEmitter()
    child.exitCode = null

    child.kill = () => {
      child._killed = true
      child.exitCode = 0
      process.nextTick(() => child.emit('exit', 0))

      return true
    }

    if (args.includes('-N')) {
      tunnels.push({ args, child })
      process.nextTick(() =>
        child.stderr.emit('data', Buffer.from(`Local forwarding listening on 127.0.0.1 port ${localPort}.`))
      )
    } else {
      process.nextTick(() => child.emit('close', 0))
    }

    if (!args.includes('-N')) {
      child.stdout = new EventEmitter()
      process.nextTick(() => child.emit('close', 0))
    }

    return child
  }

  const conn = new SshConnection({ host: 'box', user: 'me' }, { spawnFn, mux: false })
  await conn.forward(localPort, 9119)
  assert.equal(tunnels.length, 1, 'one persistent tunnel child')
  assert.ok(tunnels[0].args.includes('-L'), 'tunnel child carries -L spec')
  assert.ok(!tunnels[0].args.some(a => /ControlPath/.test(a)))

  await conn.cancelForward(localPort, 9119)
  assert.ok(tunnels[0].child._killed, 'cancelForward kills the tunnel child')

  conn._opened = true
  await conn.close() // no-mux close never runs ssh -O exit; must not throw
  srv.close()
})

test('no-mux: forward fails fast when the tunnel child dies (bad spec/auth)', async () => {
  const spawnFn: any = (_cmd, args) => {
    const child: any = new EventEmitter()
    child.stderr = new EventEmitter()
    child.exitCode = null

    child.kill = () => {}

    if (args.includes('-N')) {
      process.nextTick(() => {
        child.stderr.emit('data', Buffer.from('Permission denied (publickey).'))
        child.exitCode = 255
      })
    }

    return child
  }

  const conn = new SshConnection({ host: 'box', user: 'me' }, { spawnFn, mux: false, forwardTimeoutMs: 2000 })
  await assert.rejects(conn.forward(1, 9119), (err: any) => err.kind === 'auth-failed')
})

test('no-mux: an unrelated listener cannot mask a delayed bind failure', async () => {
  const net = await import('node:net')
  const srv = net.createServer()
  await new Promise<void>(resolve => srv.listen(0, '127.0.0.1', resolve))
  const localPort = (srv.address() as any).port

  const spawnFn: any = (_cmd, args) => {
    const child: any = new EventEmitter()
    child.stderr = new EventEmitter()
    child.exitCode = null

    child.kill = () => {}

    if (args.includes('-N')) {
      setTimeout(() => {
        child.stderr.emit('data', Buffer.from(`bind [127.0.0.1]:${localPort}: Address already in use`))
        child.exitCode = 255
        child.emit('exit', 255)
      }, 20)
    }

    return child
  }

  const conn = new SshConnection({ host: 'box' }, { spawnFn, mux: false, forwardTimeoutMs: 1000 })
  await assert.rejects(conn.forward(localPort, 9119), /address already in use/i)
  srv.close()
})

test('no-mux: tunnel death after readiness makes the connection unhealthy', async () => {
  const net = await import('node:net')
  const srv = net.createServer()
  await new Promise<void>(resolve => srv.listen(0, '127.0.0.1', resolve))
  const localPort = (srv.address() as any).port
  let tunnel

  const spawnFn: any = (_cmd, args) => {
    const child: any = new EventEmitter()
    child.stdout = new EventEmitter()
    child.stderr = new EventEmitter()
    child.exitCode = null

    child.kill = () => {}

    if (args.includes('-N')) {
      tunnel = child
      process.nextTick(() =>
        child.stderr.emit('data', Buffer.from(`Local forwarding listening on 127.0.0.1 port ${localPort}.`))
      )
    } else {
      process.nextTick(() => child.emit('close', 0))
    }

    return child
  }

  const conn = new SshConnection({ host: 'box' }, { spawnFn, mux: false })
  await conn.open()
  await conn.forward(localPort, 9119)
  tunnel.emit('exit', 255)
  assert.equal(await conn.isAlive(), false)
  srv.close()
})

test('validateSshTarget rejects a host starting with a dash (option injection)', () => {
  assert.throws(() => validateSshTarget('-oProxyCommand=evil', '', 22), /unsafe/i)
  assert.throws(() => validateSshTarget('--version', '', 22), /unsafe/i)
})

test('validateSshTarget rejects control characters in host', () => {
  assert.throws(() => validateSshTarget('host\x00evil', '', 22), /unsafe/i)
  assert.throws(() => validateSshTarget('host\nnewline', '', 22), /unsafe/i)
  assert.throws(() => validateSshTarget('host\ttab', '', 22), /unsafe/i)
})

test('validateSshTarget rejects control characters in user', () => {
  assert.throws(() => validateSshTarget('box', 'me\x00root', 22), /unsafe/i)
  assert.throws(() => validateSshTarget('box', '-oForward=yes', 22), /unsafe/i)
})

test('validateSshTarget rejects ports outside 1-65535', () => {
  assert.throws(() => validateSshTarget('box', '', 0), /port/i)
  assert.throws(() => validateSshTarget('box', '', 65536), /port/i)
  assert.throws(() => validateSshTarget('box', '', -1), /port/i)
  assert.throws(() => validateSshTarget('box', '', NaN), /port/i)
})

test('validateSshTarget accepts valid targets', () => {
  assert.doesNotThrow(() => validateSshTarget('my-host.example.com', 'alice', 22))
  assert.doesNotThrow(() => validateSshTarget('192.168.1.1', '', 2222))
  assert.doesNotThrow(() => validateSshTarget('::1', 'root', 22))
})

test('SshConnection constructor rejects hostile host/user/port', () => {
  assert.throws(() => new SshConnection({ host: '-oProxyCommand=evil' }), /unsafe/i)
  assert.throws(() => new SshConnection({ host: 'box', user: '-oForward' }), /unsafe/i)
  assert.throws(() => new SshConnection({ host: 'box', port: 99999 }), /port/i)
})

test('buildExecArgs inserts -- before the destination', () => {
  const conn = { user: 'me', host: 'box', port: 22, keyPath: '', controlPath: '/tmp/x.sock' }
  const args = buildExecArgs(conn, 'uname -s', 15000)
  const ddIdx = args.indexOf('--')
  assert.ok(ddIdx >= 0, 'must contain --')
  assert.equal(args[ddIdx + 1], 'me@box', '-- immediately precedes the destination')
  assert.equal(args[ddIdx + 2], 'uname -s', 'remote command follows destination')
})

test('buildMasterArgs inserts -- before the destination', () => {
  const conn = { user: 'me', host: 'box', port: 22, keyPath: '', controlPath: '/tmp/x.sock' }
  const args = buildMasterArgs(conn, 15000)
  const ddIdx = args.indexOf('--')
  assert.ok(ddIdx >= 0, 'must contain --')
  assert.equal(args[ddIdx + 1], 'me@box')
})

test('buildControlArgs inserts -- before the destination', () => {
  const conn = { user: 'me', host: 'box', port: 22, keyPath: '', controlPath: '/tmp/x.sock' }
  const args = buildControlArgs(conn, 'check', [], 15000)
  const ddIdx = args.indexOf('--')
  assert.ok(ddIdx >= 0, 'must contain --')
  assert.equal(args[ddIdx + 1], 'me@box')
})

test('buildInteractiveSshArgs inserts -- before the destination', () => {
  const conn = { user: 'me', host: 'box', port: 22, keyPath: '', controlPath: '/tmp/x.sock' }
  const args = buildInteractiveSshArgs(conn, '', 15000)
  const ddIdx = args.indexOf('--')
  assert.ok(ddIdx >= 0, 'must contain --')
  assert.equal(args[ddIdx + 1], 'me@box')
})

test('hostArgs rejects a keyPath with control characters', () => {
  assert.throws(() => hostArgs({ keyPath: '/tmp/key\x00inject' }), /unsafe/i)
})

test('hostArgs rejects a keyPath starting with a dash', () => {
  assert.throws(() => hostArgs({ keyPath: '-oProxyCommand=evil' }), /unsafe/i)
})

test('hostArgs accepts valid key paths', () => {
  assert.deepEqual(hostArgs({ keyPath: '/home/user/.ssh/id_ed25519' }), ['-i', '/home/user/.ssh/id_ed25519'])
  assert.deepEqual(hostArgs({ keyPath: '~/.ssh/id_rsa' }), ['-i', '~/.ssh/id_rsa'])
})

test('runSsh delivers stdinData to the child and does not log it', async () => {
  let stdinWritten = ''

  const spawnFn: any = (_cmd, _args, opts) => {
    const child: any = new EventEmitter()
    child.stdout = new EventEmitter()
    child.stderr = new EventEmitter()

    child.kill = () => {}
    child.stdin = {
      end(data) {
        stdinWritten = String(data)
      }
    }
    assert.equal(opts.stdio[0], 'pipe', 'stdin must be pipe when stdinData is provided')
    process.nextTick(() => child.emit('close', 0))

    return child
  }

  await runSsh(['host', 'cat'], { timeoutMs: 5000, spawnFn, stdinData: 'secret-token-value' })
  assert.equal(stdinWritten, 'secret-token-value', 'stdinData must be written to child.stdin')
})

test('open() rejects a control-dir that is a symlink', async () => {
  const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'ssh-test-'))
  const real = path.join(tmp, 'real')
  const link = path.join(tmp, 'link')
  fs.mkdirSync(real, { mode: 0o700 })
  fs.symlinkSync(real, link)
  const spawnFn = scriptedSpawn(args => (args.includes('check') ? { code: 255 } : { code: 0 }))
  const conn = new SshConnection({ host: 'box', user: 'me' }, { spawnFn, controlDir: link })
  await assert.rejects(conn.open(), /symlink|unsafe/i)
  fs.rmSync(tmp, { recursive: true, force: true })
})

test('open() enforces 0700 on an existing control dir with lax permissions', async () => {
  if (process.platform === 'win32') {
    return
  }

  const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'ssh-test-'))
  const dir = path.join(tmp, 'ctrl')
  fs.mkdirSync(dir, { mode: 0o755 })
  const spawnFn = scriptedSpawn(args => (args.includes('check') ? { code: 255 } : { code: 0 }))
  const conn = new SshConnection({ host: 'box', user: 'me' }, { spawnFn, controlDir: dir })
  await conn.open()
  const stat = fs.statSync(dir)
  assert.equal(stat.mode & 0o777, 0o700, 'control dir must be tightened to 0700')
  fs.rmSync(tmp, { recursive: true, force: true })
})

test('control socket identity separates installation scope and key identity', () => {
  const base = controlSocketPath('me', 'box', 22, '/tmp/d', {
    ownershipId: 'installation-a',
    scope: 'primary',
    keyPath: '/keys/id'
  })

  assert.equal(
    base,
    controlSocketPath('me', 'box', 22, '/tmp/d', {
      ownershipId: 'installation-a',
      scope: 'primary',
      keyPath: '/keys/./id'
    })
  )
  assert.notEqual(
    base,
    controlSocketPath('me', 'box', 22, '/tmp/d', {
      ownershipId: 'installation-a',
      scope: 'worker',
      keyPath: '/keys/id'
    })
  )
  assert.notEqual(
    base,
    controlSocketPath('me', 'box', 22, '/tmp/d', {
      ownershipId: 'installation-b',
      scope: 'primary',
      keyPath: '/keys/id'
    })
  )
  assert.notEqual(
    base,
    controlSocketPath('me', 'box', 22, '/tmp/d', {
      ownershipId: 'installation-a',
      scope: 'primary',
      keyPath: '/keys/other'
    })
  )
  assert.notEqual(
    base,
    controlSocketPath('me', 'box', 22, '/tmp/d', {
      ownershipId: 'installation-a',
      scope: 'primary',
      keyPath: '/keys/id',
      effectiveConfigFingerprint: 'changed-config'
    })
  )
})

test('closing one scope addresses only that scope control master', async () => {
  const firstSpawn = scriptedSpawn({ code: 0 })
  const secondSpawn = scriptedSpawn({ code: 0 })

  const first = new SshConnection(
    { host: 'box', user: 'me' },
    {
      spawnFn: firstSpawn,
      controlDir: '/tmp/d',
      ownershipId: 'installation',
      scope: 'first'
    }
  )

  const second = new SshConnection(
    { host: 'box', user: 'me' },
    {
      spawnFn: secondSpawn,
      controlDir: '/tmp/d',
      ownershipId: 'installation',
      scope: 'second'
    }
  )

  first._opened = true
  second._opened = true
  await first.close()
  assert.notEqual(first.controlPath, second.controlPath)
  assert.ok(firstSpawn.calls[0].includes(`ControlPath=${first.controlPath}`))
  assert.ok(!firstSpawn.calls[0].includes(`ControlPath=${second.controlPath}`))
  assert.equal(second._opened, true)
})

test('failed ControlMaster close disowns the master instead of retrying it', async () => {
  // Old contract kept _opened=true for a retry — which left wedged ControlPersist
  // masters trusted and reattachable (the macOS mode-switch livelock). New
  // contract: a master that refuses -O exit is disowned — socket dropped,
  // connection marked closed — so the next open dials fresh.
  const spawnFn = scriptedSpawn([{ code: 255, stderr: 'master refused exit' }])
  const conn = new SshConnection({ host: 'box', user: 'me' }, { spawnFn, controlDir: '/tmp/d' })
  conn._opened = true
  await conn.close()
  assert.equal(conn._opened, false)
  assert.equal(spawnFn.calls.length, 1)
})

test('stopTunnelChild waits for process exit', async () => {
  const child: any = new EventEmitter()
  child.exitCode = null

  child.kill = () => {
    process.nextTick(() => {
      child.exitCode = 0
      child.emit('exit', 0)
    })

    return true
  }

  let stopped = false

  const stopping = stopTunnelChild(child).then(() => {
    stopped = true
  })

  assert.equal(stopped, false)
  await stopping
  assert.equal(stopped, true)
})
