import assert from 'node:assert/strict'

import { test } from 'vitest'

import {
  buildSpawnCommand,
  cleanupStale,
  connect,
  expandRemotePath,
  fingerprintToken,
  isForwardBindCollision,
  locateHermes,
  LOCKFILE_SCHEMA_VERSION,
  lockfilePath,
  openForward,
  ownershipDirectory,
  pidIsOurDashboard,
  probeRemotePlatform,
  PROTOCOL_VERSION,
  readLockfile,
  READY_RE,
  remotePidAlive,
  remoteSupportsSshOwnership,
  scrapeReadyPort,
  spawnLogPath,
  spawnRemoteDashboard,
  validateRemotePath,
  writeLockfile
} from './remote-lifecycle'

const OWNERSHIP_ID = '0123456789abcdef0123456789abcdef'
const SPAWN_NONCE = '0123456789abcdef'

function ownedLock(over: any = {}) {
  return {
    schemaVersion: LOCKFILE_SCHEMA_VERSION,
    protocolVersion: PROTOCOL_VERSION,
    ownershipId: OWNERSHIP_ID,
    spawnNonce: SPAWN_NONCE,
    pid: 333,
    port: 40000,
    profile: '',
    hermesPath: '~/.local/bin/hermes',
    hermesHome: '~/.hermes',
    logPath: spawnLogPath(OWNERSHIP_ID, SPAWN_NONCE),
    tokenFingerprint: fingerprintToken('stored-token'),
    startedAt: '2026-07-14T00:00:00.000Z',
    ...over
  }
}

// A fake SshConnection whose exec() is matched against an ordered list of
// [regex|fn, response|fn] rules. First match wins; unmatched commands return ''.
function fakeSsh(rules: any[] = []) {
  const calls: string[] = []

  return {
    calls,
    async exec(cmd) {
      calls.push(cmd)

      for (const [matcher, resp] of rules) {
        const hit = typeof matcher === 'function' ? matcher(cmd) : matcher.test(cmd)

        if (hit) {
          const out = typeof resp === 'function' ? resp(cmd) : resp

          if (out instanceof Error) {
            throw out
          }

          return out
        }
      }

      return ''
    }
  }
}

test('locateHermes prefers the explicit profile path when executable', async () => {
  const ssh = fakeSsh([[/\[ -x .*\/opt\/hermes/, 'OK']])
  assert.equal(await locateHermes(ssh, '/opt/hermes'), '/opt/hermes')
})

test('locateHermes throws (no silent fallback) when an EXPLICIT path is not executable', async () => {
  // command -v WOULD find a different install, but an explicit path must not
  // silently fall back to it — that is the "connected to the wrong hermes" bug.
  const ssh = fakeSsh([
    [/command -v hermes/, '/home/u/.local/bin/hermes\n'],
    [/\[ -x .*\.local\/bin\/hermes/, 'OK']
  ])

  await assert.rejects(
    () => locateHermes(ssh, '/bad/path/hermes'),
    (err: any) => {
      assert.equal(err.kind, 'hermes-not-found')
      assert.match(err.message, /\/bad\/path\/hermes/)

      return true
    }
  )
})

test('locateHermes falls back to the login-shell command -v probe', async () => {
  const ssh = fakeSsh([
    [/command -v hermes/, '/home/u/.local/bin/hermes\n'],
    [/\[ -x .*\.local\/bin\/hermes/, 'OK']
  ])

  assert.equal(await locateHermes(ssh, ''), '/home/u/.local/bin/hermes')
})

test('locateHermes canonicalizes an installer wrapper to its executable target', async () => {
  const ssh = fakeSsh([
    [/command -v hermes/, '/home/u/.local/bin/hermes\n'],
    [/\[ -x .*\.local\/bin\/hermes/, 'OK'],
    [/python3 -c/, '/home/u/.hermes/hermes-agent/venv/bin/hermes\n']
  ])

  assert.equal(await locateHermes(ssh, ''), '/home/u/.hermes/hermes-agent/venv/bin/hermes')
})

test('locateHermes falls back to ~/.local/bin/hermes when the login-shell probe misses', async () => {
  // ~/.local/bin is the non-root installer's command location (scripts/install.sh).
  const ssh = fakeSsh([
    [/command -v hermes/, ''],
    [/\[ -x .*\.local\/bin\/hermes/, 'OK']
  ])

  assert.equal(await locateHermes(ssh, ''), '~/.local/bin/hermes')
})

test('locateHermes tries the conventional venv path last', async () => {
  const ssh = fakeSsh([[/\[ -x .*venv\/bin\/hermes/, 'OK']])
  assert.equal(await locateHermes(ssh, ''), '~/.hermes/hermes-agent/venv/bin/hermes')
})

test('locateHermes throws a hermes-not-found error with an install hint', async () => {
  const ssh = fakeSsh([]) // nothing is executable
  await assert.rejects(
    () => locateHermes(ssh, ''),
    (err: any) => {
      assert.equal(err.kind, 'hermes-not-found')
      assert.match(err.message, /install/i)

      return true
    }
  )
})

test('locateHermes uses a login shell for the command -v probe', async () => {
  const ssh = fakeSsh([
    [/command -v hermes/, '/x/hermes'],
    [/\[ -x/, 'OK']
  ])

  await locateHermes(ssh, '')
  assert.ok(
    ssh.calls.some(c => /bash -lc/.test(c)),
    'must probe in a login shell (PATH pitfall)'
  )
})

test('probeRemotePlatform accepts Linux and macOS', async () => {
  assert.deepEqual(await probeRemotePlatform(fakeSsh([[/uname/, 'Linux\nx86_64']])), {
    os: 'Linux',
    arch: 'x86_64'
  })
  assert.deepEqual(await probeRemotePlatform(fakeSsh([[/uname/, 'Darwin\narm64']])), {
    os: 'Darwin',
    arch: 'arm64'
  })
})

test('probeRemotePlatform rejects unsupported remote platforms', async () => {
  await assert.rejects(
    () => probeRemotePlatform(fakeSsh([[/uname/, 'MINGW64_NT\nx86_64']])),
    (err: any) => {
      assert.equal(err.kind, 'unsupported-platform')

      return true
    }
  )
})

test('ownership paths are isolated by ownership ID and spawn nonce', () => {
  assert.equal(ownershipDirectory(OWNERSHIP_ID), `~/.hermes/desktop-ssh/${OWNERSHIP_ID}`)
  assert.equal(lockfilePath(OWNERSHIP_ID), `~/.hermes/desktop-ssh/${OWNERSHIP_ID}/backend.lock.json`)
  assert.equal(spawnLogPath(OWNERSHIP_ID, SPAWN_NONCE), `~/.hermes/desktop-ssh/${OWNERSHIP_ID}/${SPAWN_NONCE}.log`)
})

test('readLockfile returns null for missing, empty, malformed, or wrong-schema', async () => {
  assert.equal(await readLockfile(fakeSsh([[/cat/, '']]), OWNERSHIP_ID), null)
  assert.equal(await readLockfile(fakeSsh([[/cat/, 'not json']]), OWNERSHIP_ID), null)
  assert.equal(await readLockfile(fakeSsh([[/cat/, JSON.stringify({ schemaVersion: 999 })]]), OWNERSHIP_ID), null)
  const good = ownedLock({ pid: 1, port: 2 })
  assert.deepEqual(await readLockfile(fakeSsh([[/cat/, JSON.stringify(good)]]), OWNERSHIP_ID), good)
})

test('writeLockfile mkdir -ps and stamps the schema version', async () => {
  const ssh = fakeSsh([])
  await writeLockfile(ssh, OWNERSHIP_ID, ownedLock({ pid: 7, port: 9 }))
  const cmd = ssh.calls.join('\n')
  assert.match(cmd, /mkdir -p/)
  assert.match(cmd, new RegExp(`"schemaVersion":${LOCKFILE_SCHEMA_VERSION}`))
})

test('remotePidAlive maps kill -0 ALIVE/DEAD', async () => {
  assert.equal(await remotePidAlive(fakeSsh([[/kill -0/, 'ALIVE']]), 123), true)
  assert.equal(await remotePidAlive(fakeSsh([[/kill -0/, 'DEAD']]), 123), false)
  assert.equal(await remotePidAlive(fakeSsh([]), null), false)
})

test('metadata and process proof transport failures remain indeterminate', async () => {
  const failure = new Error('connection reset')
  await assert.rejects(
    () => readLockfile(fakeSsh([[/cat/, failure]]), OWNERSHIP_ID),
    (error: any) => error.kind === 'transient-transport-error'
  )
  await assert.rejects(
    () => remotePidAlive(fakeSsh([[/kill -0/, failure]]), 123),
    (error: any) => error.kind === 'transient-transport-error'
  )
  await assert.rejects(
    () => pidIsOurDashboard(fakeSsh([[/print\("OWNED"/, failure]]), 5, SPAWN_NONCE, '/x/hermes'),
    (error: any) => error.kind === 'transient-transport-error'
  )
})

test('pidIsOurDashboard requires the exact serve ownership nonce', async () => {
  const ours = `/x/hermes serve --isolated --ssh-owner-nonce ${SPAWN_NONCE}`
  assert.equal(await pidIsOurDashboard(fakeSsh([[/print\("OWNED"/, 'OWNED\n']]), 5, SPAWN_NONCE, '/x/hermes'), true)
  assert.equal(
    await pidIsOurDashboard(
      fakeSsh([[/print\("OWNED"/, command => (command.includes('fedcba9876543210') ? 'FOREIGN\n' : 'OWNED\n')]]),
      5,
      'fedcba9876543210',
      '/x/hermes'
    ),
    false
  )
  assert.equal(await pidIsOurDashboard(fakeSsh([[/print\("OWNED"/, 'FOREIGN\n']]), 5, SPAWN_NONCE, '/x/hermes'), false)
})

test('cleanupStale kills ONLY a provably-ours pid, always drops the lockfile', async () => {
  const notOurs = fakeSsh([[/print\("OWNED"/, 'FOREIGN\n']])
  await cleanupStale(notOurs, OWNERSHIP_ID, {
    pid: 5,
    spawnNonce: SPAWN_NONCE,
    hermesPath: '/x/hermes',
    logPath: spawnLogPath(OWNERSHIP_ID, SPAWN_NONCE)
  })
  assert.ok(!notOurs.calls.some(c => /kill 5\b/.test(c)), 'must not kill a pid that is not our dashboard')
  assert.ok(notOurs.calls.some(c => /rm -f/.test(c)))

  const ours = fakeSsh([[/print\("OWNED"/, 'OWNED\n']])
  await cleanupStale(ours, OWNERSHIP_ID, {
    pid: 9,
    spawnNonce: SPAWN_NONCE,
    hermesPath: '/x/hermes',
    logPath: spawnLogPath(OWNERSHIP_ID, SPAWN_NONCE)
  })
  assert.ok(ours.calls.some(c => /kill 9\b/.test(c)))
  assert.ok(ours.calls.some(c => /rm -f/.test(c)))
})

test('buildSpawnCommand is headless serve, detached, token not in argv', () => {
  const cmd = buildSpawnCommand('/x/hermes', 'work', { logPath: spawnLogPath(OWNERSHIP_ID, SPAWN_NONCE) })
  assert.match(cmd, /serve --isolated/)
  assert.match(cmd, /--host 127\.0\.0\.1 --port 0/)
  assert.doesNotMatch(cmd, /--skip-build|--no-open/)
  assert.doesNotMatch(cmd, /\bdashboard\b/)
  assert.match(cmd, /--profile/)
  assert.match(cmd, /work/)
  assert.match(cmd, /setsid/)
  assert.match(cmd, /<\/dev\/null/)
  assert.match(cmd, /echo \$!/)
  assert.ok(!cmd.includes('tok_secret_value'), 'token must not appear in spawn command')
  assert.ok(!cmd.includes('HERMES_DASHBOARD_SESSION_TOKEN'), 'token env var must not appear')
})

test('buildSpawnCommand always uses serve (legacy dashboard path removed)', () => {
  const cmd = buildSpawnCommand('/x/hermes', 'work', { logPath: spawnLogPath(OWNERSHIP_ID, SPAWN_NONCE) })
  assert.match(cmd, /serve --isolated/)
  assert.match(cmd, /--host 127\.0\.0\.1 --port 0/)
  assert.doesNotMatch(cmd, /dashboard/)
  assert.doesNotMatch(cmd, /--skip-build/)
  assert.match(cmd, /setsid/)
})

test('spawnRemoteDashboard returns exact ownership artifacts', async () => {
  const ssh = fakeSsh([
    [/grep -q ssh-session-token-file/, 'YES\n'],
    [/python3 -c/, ''],
    [/printf '%s\\n'/, ''],
    [/setsid|nohup/, '4242\n']
  ])

  const { pid, spawnNonce, logPath } = await spawnRemoteDashboard(ssh, {
    hermesPath: '/x/hermes',
    profile: '',
    token: 'tk',
    ownershipId: OWNERSHIP_ID
  })

  assert.equal(pid, 4242)
  assert.match(spawnNonce, /^[0-9a-f]{16}$/)
  assert.equal(logPath, spawnLogPath(OWNERSHIP_ID, spawnNonce))
})

test('spawnRemoteDashboard always spawns serve (legacy dashboard path removed)', async () => {
  const ssh = fakeSsh([
    [/grep -q ssh-session-token-file/, 'YES\n'],
    [/python3 -c/, ''],
    [/printf '%s\\n'/, ''],
    [/setsid|nohup/, '4242\n']
  ])

  await spawnRemoteDashboard(ssh, { hermesPath: '/x/hermes', profile: '', token: 'tk', ownershipId: OWNERSHIP_ID })
  const spawn = ssh.calls.find(c => /setsid|nohup/.test(c))
  assert.match(spawn, /serve --isolated/)
  assert.doesNotMatch(spawn, /\bdashboard\b/)
})

test('READY_RE accepts both serve and dashboard sentinels', () => {
  assert.equal(READY_RE.exec('HERMES_BACKEND_READY port=4321')?.[1], '4321')
  assert.equal(READY_RE.exec('HERMES_DASHBOARD_READY port=8765')?.[1], '8765')
})

test('spawnRemoteDashboard rejects when no pid is returned', async () => {
  const ssh = fakeSsh([
    [/grep -q ssh-session-token-file/, 'YES\n'],
    [/python3 -c/, ''],
    [/printf '%s\\n'/, ''],
    [/setsid|nohup/, 'not-a-pid']
  ])

  await assert.rejects(
    () => spawnRemoteDashboard(ssh, { hermesPath: '/x/hermes', profile: '', token: 't', ownershipId: OWNERSHIP_ID }),
    (err: any) => {
      assert.equal(err.kind, 'spawn-failed')

      return true
    }
  )
})

test('scrapeReadyPort reads only the named spawn log', async () => {
  const logPath = spawnLogPath(OWNERSHIP_ID, SPAWN_NONCE)
  const ssh = fakeSsh([[/cat/, 'some noise\nHERMES_DASHBOARD_READY port=51234\n']])
  const port = await scrapeReadyPort(ssh, logPath, { timeoutMs: 1000 })
  assert.equal(port, 51234)
  assert.ok(ssh.calls.every(call => !call.includes('desktop-ssh.log')))
})

test('scrapeReadyPort times out and reports a dead spawn', async () => {
  // never emits a READY line
  const ssh = fakeSsh([[/cat .*\.log/, 'still starting...']])
  await assert.rejects(
    () => scrapeReadyPort(ssh, spawnLogPath(OWNERSHIP_ID, SPAWN_NONCE), { timeoutMs: 60 }),
    (err: any) => {
      assert.equal(err.kind, 'ready-timeout')

      return true
    }
  )
  // dead process before announcement → spawn-failed
  await assert.rejects(
    () =>
      scrapeReadyPort(fakeSsh([[/cat/, '']]), spawnLogPath(OWNERSHIP_ID, SPAWN_NONCE), {
        timeoutMs: 1000,
        isAlive: async () => false
      }),
    (err: any) => {
      assert.equal(err.kind, 'spawn-failed')

      return true
    }
  )
})

function connectDeps(ssh, over: any = {}) {
  return {
    ssh,
    ownershipId: OWNERSHIP_ID,
    profile: '',
    forward: async () => {},
    cancelForward: async () => {},
    pickLocalPort: async () => 50001,
    waitForHermes: async () => {},
    probeReuseProof: async () => 'authenticated-ok',
    adoptServedToken: async (_baseUrl, spawn) => spawn || 'served-token',
    rememberLog: () => {},
    readyTimeoutMs: 2000,
    ...over
  }
}

test('connect() spawns fresh when there is no lockfile, adopts the served token', async () => {
  const ssh = fakeSsh([
    [/uname/, 'Linux\nx86_64'],
    [/\[ -x/, 'OK'],
    [/cat .*lock\.json/, ''], // no lockfile
    [/grep -q ssh-session-token-file/, 'YES\n'],
    [/python3 -c/, ''], // token file write
    [/printf '%s\\n'/, ''],
    [/setsid/, '777\n'],
    [/kill -0 777/, 'ALIVE'],
    [/cat .*\.log/, 'HERMES_DASHBOARD_READY port=51999\n']
  ])

  const result = await connect(connectDeps(ssh, { adoptServedToken: async () => 'the-served-token' }))
  assert.equal(result.reused, false)
  assert.equal(result.remotePort, 51999)
  assert.equal(result.localPort, 50001)
  assert.equal(result.pid, 777)
  assert.equal(result.token, 'the-served-token')
  assert.equal(result.baseUrl, 'http://127.0.0.1:50001')
  assert.equal(result.tokenFingerprint, fingerprintToken('the-served-token'))
})

test('connect() reuses a healthy dashboard when fingerprint + probe pass', async () => {
  const reuseToken = 'stored-token'
  const lock = ownedLock({ tokenFingerprint: fingerprintToken(reuseToken) })

  const ssh = fakeSsh([
    [/uname/, 'Linux\nx86_64'],
    [/\[ -x/, 'OK'],
    [/cat .*lock\.json/, JSON.stringify(lock)],
    [/kill -0/, 'ALIVE'],
    [/print\("OWNED"/, 'OWNED\n']
  ])

  const result = await connect(connectDeps(ssh, { reuseToken, adoptServedToken: async (_b, t) => t }))
  assert.equal(result.reused, true)
  assert.equal(result.pid, 333)
  assert.equal(result.remotePort, 40000)
  // never spawned
  assert.ok(!ssh.calls.some(c => /setsid/.test(c)), 'reuse path must not spawn a new dashboard')
})

test('connect() respawns when the lockfile hermesPath differs from the resolved path', async () => {
  const reuseToken = 'stored-token'
  const lock = ownedLock({ hermesPath: '/old/stale/hermes', tokenFingerprint: fingerprintToken(reuseToken) })

  const ssh = fakeSsh([
    [/uname/, 'Linux\nx86_64'],
    [/\[ -x/, 'OK'],
    [/cat .*lock\.json/, JSON.stringify(lock)],
    [/kill -0/, 'ALIVE'],
    [/print\("OWNED"/, 'FOREIGN\n'],
    [/--version/, 'Hermes Agent v0.18.2\n'],
    [/grep -q ssh-session-token-file/, 'YES\n'],
    [/python3 -c/, ''],
    [/setsid/, '890\n'],
    [/cat .*\.log/, 'HERMES_DASHBOARD_READY port=52050\n']
  ])

  const result = await connect(
    connectDeps(ssh, { reuseToken, remoteHermesPath: '/new/hermes', adoptServedToken: async () => 'fresh' })
  )

  assert.equal(result.reused, false, 'must respawn, not reuse the old-path dashboard')
  assert.ok(
    ssh.calls.some(c => /setsid/.test(c)),
    'a fresh dashboard must be spawned'
  )
})

test('connect() respawns when the lockfile protocolVersion is incompatible', async () => {
  const reuseToken = 'stored-token'

  const lock = {
    schemaVersion: LOCKFILE_SCHEMA_VERSION,
    protocolVersion: PROTOCOL_VERSION + 99,
    pid: 333,
    port: 40000,
    tokenFingerprint: fingerprintToken(reuseToken)
  }

  const ssh = fakeSsh([
    [/uname/, 'Linux\nx86_64'],
    [/\[ -x/, 'OK'],
    [/cat .*lock\.json/, JSON.stringify(lock)],
    [/kill -0 333/, 'ALIVE'],
    [/print\("OWNED"/, 'FOREIGN\n'],
    [/grep -q ssh-session-token-file/, 'YES\n'],
    [/python3 -c/, ''],
    [/setsid/, '901\n'],
    [/kill -0 901/, 'ALIVE'],
    [/cat .*\.log/, 'HERMES_DASHBOARD_READY port=44100\n']
  ])

  const result = await connect(connectDeps(ssh, { reuseToken, adoptServedToken: async () => 'fresh' }))
  assert.equal(result.reused, false, 'incompatible protocol must force a fresh spawn, not a reattach')
  assert.equal(result.pid, 901)
})

test('connect() fresh spawn writes hermesHome + protocolVersion into the lockfile', async () => {
  const writes: string[] = []

  const ssh = fakeSsh([
    [/uname/, 'Linux\nx86_64'],
    [/\[ -x/, 'OK'],
    [/cat .*lock\.json/, ''], // no lockfile
    [/HERMES_HOME/, '/home/alice/.hermes\n'],
    [/grep -q ssh-session-token-file/, 'YES\n'],
    [/python3 -c/, ''],
    [/printf '%s\\n'/, ''],
    [/setsid/, '700\n'],
    [/kill -0 700/, 'ALIVE'],
    [/cat .*\.log/, 'HERMES_DASHBOARD_READY port=45500\n'],
    [
      /printf '%s' '/,
      c => {
        writes.push(c)

        return ''
      }
    ]
  ])

  await connect(connectDeps(ssh, { adoptServedToken: async () => 'fresh' }))
  const lockWrite = writes.find(c => c.includes('schemaVersion')) || ''
  assert.match(lockWrite, new RegExp(`"protocolVersion":${PROTOCOL_VERSION}`))
  assert.match(lockWrite, /"hermesHome":"\/home\/alice\/\.hermes"/)
})

test('connect() respawns when the lockfile pid is dead (killed dashboard)', async () => {
  const lock = ownedLock({ tokenFingerprint: fingerprintToken('t') })

  const ssh = fakeSsh([
    [/uname/, 'Linux\nx86_64'],
    [/\[ -x/, 'OK'],
    [/cat .*lock\.json/, JSON.stringify(lock)],
    [/kill -0 333/, 'DEAD'],
    [/print\("OWNED"/, 'FOREIGN\n'],
    [/grep -q ssh-session-token-file/, 'YES\n'],
    [/python3 -c/, ''],
    [/setsid/, '888\n'],
    [/kill -0 888/, 'ALIVE'],
    [/cat .*\.log/, 'HERMES_DASHBOARD_READY port=42000\n']
  ])

  const result = await connect(connectDeps(ssh, { reuseToken: 't', adoptServedToken: async () => 'fresh' }))
  assert.equal(result.reused, false)
  assert.equal(result.pid, 888)
  assert.equal(result.remotePort, 42000)
  assert.ok(
    !ssh.calls.some(command => command.includes('pid=333') && command.includes('print("OWNED"')),
    'a dead pid has no process identity to verify'
  )
})

test('connect() respawns when the dashboard is wedged (alive pid, probe fails)', async () => {
  const reuseToken = 'stored'

  const lock = {
    schemaVersion: LOCKFILE_SCHEMA_VERSION,
    protocolVersion: PROTOCOL_VERSION,
    pid: 333,
    port: 40000,
    tokenFingerprint: fingerprintToken(reuseToken)
  }

  const ssh = fakeSsh([
    [/uname/, 'Linux\nx86_64'],
    [/\[ -x/, 'OK'],
    [/cat .*lock\.json/, JSON.stringify(lock)],
    [/kill -0/, 'ALIVE'],
    [/print\("OWNED"/, 'FOREIGN\n'],
    [/grep -q ssh-session-token-file/, 'YES\n'],
    [/python3 -c/, ''],
    [/setsid/, '999\n'],
    [/kill -0 999/, 'ALIVE'],
    [/cat .*\.log/, 'HERMES_DASHBOARD_READY port=43000\n']
  ])

  const result = await connect(
    connectDeps(ssh, {
      reuseToken,
      probeReuseProof: async () => 'authenticated-stale',
      adoptServedToken: async () => 'fresh'
    })
  )

  assert.equal(result.reused, false)
  assert.equal(result.pid, 999)
  assert.equal(result.remotePort, 43000)
})

test('connect() aborts on an unsupported remote platform before doing anything else', async () => {
  const ssh = fakeSsh([[/uname/, 'SunOS\nsun4v']])
  await assert.rejects(
    () => connect(connectDeps(ssh)),
    (err: any) => {
      assert.equal(err.kind, 'unsupported-platform')

      return true
    }
  )
  assert.ok(!ssh.calls.some(c => /setsid/.test(c)))
})

test('openForward retries bind collisions only', async () => {
  const ports = [41001, 41002]
  const calls: number[] = []

  const localPort = await openForward(
    {
      pickLocalPort: async () => ports.shift(),
      forward: async port => {
        calls.push(port)

        if (calls.length === 1) {
          throw new Error('bind: Address already in use')
        }
      }
    },
    9119
  )

  assert.equal(localPort, 41002)
  assert.deepEqual(calls, [41001, 41002])
  assert.equal(isForwardBindCollision(new Error('Permission denied')), false)
})

test('connect() preserves an owned backend when a reuse transport throws', async () => {
  const reuseToken = 'stored-token'
  const lock = ownedLock({ tokenFingerprint: fingerprintToken(reuseToken) })

  const ssh = fakeSsh([
    [/uname/, 'Linux\nx86_64'],
    [/\[ -x/, 'OK'],
    [/cat .*lock\.json/, JSON.stringify(lock)],
    [/kill -0/, 'ALIVE'],
    [/print\("OWNED"/, 'OWNED\n']
  ])

  await assert.rejects(
    () =>
      connect(
        connectDeps(ssh, {
          reuseToken,
          forward: async () => {
            throw new Error('network reset')
          }
        })
      ),
    /network reset/
  )
  assert.ok(!ssh.calls.some(cmd => /kill 333\b/.test(cmd)))
})

test('validateRemotePath accepts absolute POSIX paths', () => {
  assert.doesNotThrow(() => validateRemotePath('/usr/bin/hermes'))
  assert.doesNotThrow(() => validateRemotePath('/home/user/.hermes/hermes-agent/venv/bin/hermes'))
})

test('validateRemotePath accepts ~/ prefix paths', () => {
  assert.doesNotThrow(() => validateRemotePath('~/bin/hermes'))
  assert.doesNotThrow(() => validateRemotePath('~/.hermes/logs/desktop-ssh.log'))
  assert.doesNotThrow(() => validateRemotePath('~'))
})

test('validateRemotePath accepts paths with spaces and quotes', () => {
  assert.doesNotThrow(() => validateRemotePath('/home/user/my project/hermes'))
  assert.doesNotThrow(() => validateRemotePath("~/path with 'quotes'/file"))
  assert.doesNotThrow(() => validateRemotePath('/path with "double quotes"/file'))
})

test('validateRemotePath rejects relative paths', () => {
  assert.throws(() => validateRemotePath('hermes'), /absolute|relative/i)
  assert.throws(() => validateRemotePath('./bin/hermes'), /absolute|relative/i)
  assert.throws(() => validateRemotePath('../etc/passwd'), /absolute|relative/i)
})

test('validateRemotePath rejects NUL and newline', () => {
  assert.throws(() => validateRemotePath('/usr/bin/hermes\x00'), /unsafe/i)
  assert.throws(() => validateRemotePath('/usr/bin/hermes\n'), /unsafe/i)
  assert.throws(() => validateRemotePath('/usr/bin/hermes\r'), /unsafe/i)
})

test('validateRemotePath preserves shell metacharacters as path data', () => {
  for (const p of ['/usr/$(whoami)/hermes', '/usr/`id`/hermes', '/usr/a;b|c&d<e>f']) {
    assert.doesNotThrow(() => validateRemotePath(p))
    assert.match(expandRemotePath(p), /^'/)
  }
})

test('expandRemotePath expands ~/ to "$HOME"/', () => {
  const result = expandRemotePath('~/.hermes/logs/desktop-ssh.log')
  assert.match(result, /\$HOME/)
  assert.ok(!result.includes('eval'), 'must not use eval')
  assert.ok(!result.includes('echo'), 'must not use echo for expansion')
})

test('expandRemotePath returns quoted absolute paths unchanged', () => {
  const result = expandRemotePath('/usr/local/bin/hermes')
  assert.ok(result.includes('/usr/local/bin/hermes'))
  assert.ok(!result.includes('eval'))
})

test('expandRemotePath preserves spaces as data', () => {
  const result = expandRemotePath('/home/user/my project/hermes')
  assert.ok(result.includes('my project'), 'spaces must be preserved, not split')
})

test('buildSpawnCommand does not embed the token in the command string', () => {
  const cmd = buildSpawnCommand('/x/hermes', 'work', { logPath: spawnLogPath(OWNERSHIP_ID, SPAWN_NONCE) })
  assert.ok(!cmd.includes('super_secret_token_value'), 'token must not appear in the spawn command')
  assert.ok(!cmd.includes('HERMES_DASHBOARD_SESSION_TOKEN'), 'env var name must not appear')
})

test('buildSpawnCommand includes --ssh-session-token-file when tokenFilePath is provided', () => {
  const cmd = buildSpawnCommand('/x/hermes', 'work', {
    tokenFilePath: `~/.hermes/desktop-ssh/${OWNERSHIP_ID}/${SPAWN_NONCE}.token`,
    logPath: spawnLogPath(OWNERSHIP_ID, SPAWN_NONCE),
    spawnNonce: SPAWN_NONCE
  })

  assert.match(cmd, /--ssh-session-token-file/)
  assert.match(cmd, /\.hermes\/desktop-ssh\//)
})

test('buildSpawnCommand always uses serve, never dashboard', () => {
  const cmd = buildSpawnCommand('/x/hermes', '', { logPath: spawnLogPath(OWNERSHIP_ID, SPAWN_NONCE) })
  assert.match(cmd, /serve --isolated/)
  assert.doesNotMatch(cmd, /\bdashboard\b/)
  assert.doesNotMatch(cmd, /--skip-build/)
  assert.doesNotMatch(cmd, /--no-open/)
})

test('spawnRemoteDashboard removes a token file when upload reporting fails', async () => {
  const failure = new Error('channel closed')

  const ssh = fakeSsh([
    [/grep -q ssh-session-token-file/, 'YES\n'],
    [command => /python3 -c/.test(command) && !/rm -f/.test(command), failure],
    [/rm -f/, '']
  ])

  await assert.rejects(
    () => spawnRemoteDashboard(ssh, { hermesPath: '/x/hermes', profile: '', token: 'tok', ownershipId: OWNERSHIP_ID }),
    /channel closed/
  )
  assert.ok(ssh.calls.some(command => /rm -f .*\.token/.test(command)))
})

test('spawnRemoteDashboard streams the token over stdin, not argv/env', async () => {
  const stdinCalls: string[] = []
  const calls: string[] = []

  const ssh = {
    calls,
    async exec(cmd, opts?) {
      calls.push(cmd)

      if (opts?.stdinData) {
        stdinCalls.push(opts.stdinData)
      }

      if (/grep -q ssh-session-token-file/.test(cmd)) {
        return 'YES\n'
      }

      if (/python3 -c/.test(cmd)) {
        return ''
      }

      if (/setsid|nohup/.test(cmd)) {
        return '4242\n'
      }

      if (/printf '%s\\n'/.test(cmd)) {
        return ''
      }

      return ''
    }
  }

  const { pid } = await spawnRemoteDashboard(ssh as any, {
    hermesPath: '/x/hermes',
    profile: '',
    token: 'secret_token_val',
    ownershipId: OWNERSHIP_ID
  })

  assert.equal(pid, 4242)
  assert.ok(stdinCalls.length > 0, 'token must be sent via stdin')
  assert.ok(
    stdinCalls.some(d => d === 'secret_token_val'),
    'stdin must contain the token'
  )

  for (const cmd of calls) {
    assert.ok(!cmd.includes('secret_token_val'), `token leaked into command: ${cmd}`)
  }
})

test('spawnRemoteDashboard upload uses exclusive-create and O_NOFOLLOW', async () => {
  const calls: string[] = []

  const ssh = {
    calls,
    async exec(cmd, opts?) {
      calls.push(cmd)

      if (/grep -q ssh-session-token-file/.test(cmd)) {
        return 'YES\n'
      }

      if (/python3 -c/.test(cmd)) {
        return ''
      }

      if (/setsid|nohup/.test(cmd)) {
        return '4242\n'
      }

      if (/printf '%s\\n'/.test(cmd)) {
        return ''
      }

      return ''
    }
  }

  await spawnRemoteDashboard(ssh as any, {
    hermesPath: '/x/hermes',
    profile: '',
    token: 'tk',
    ownershipId: OWNERSHIP_ID
  })
  const uploadCmd = calls.find(c => /python3 -c/.test(c))
  assert.ok(uploadCmd, 'must use python3 -c for token upload')
  assert.match(uploadCmd, /O_EXCL/, 'upload must use O_EXCL to reject existing files')
  assert.match(uploadCmd, /O_NOFOLLOW/, 'upload must use O_NOFOLLOW to reject symlinks')
  assert.match(uploadCmd, /O_WRONLY/, 'upload must open write-only')
  assert.match(uploadCmd, /dir_fd=dd/, 'upload must create relative to the opened parent directory')
  assert.match(uploadCmd, /os\.fstat\(dd\)/, 'upload must validate the opened parent directory')
  assert.ok(!uploadCmd.includes('tk'), 'token must not appear in the upload command')
})

test('readLockfile rejects lock with non-integer pid', async () => {
  const lock = { schemaVersion: LOCKFILE_SCHEMA_VERSION, pid: 'not-a-number', port: 8080 }
  assert.equal(await readLockfile(fakeSsh([[/cat/, JSON.stringify(lock)]]), OWNERSHIP_ID), null)
})

test('readLockfile rejects lock with pid <= 0', async () => {
  const lock = { schemaVersion: LOCKFILE_SCHEMA_VERSION, pid: -1, port: 8080 }
  assert.equal(await readLockfile(fakeSsh([[/cat/, JSON.stringify(lock)]]), OWNERSHIP_ID), null)
})

test('readLockfile rejects lock with port out of range', async () => {
  const lock = { schemaVersion: LOCKFILE_SCHEMA_VERSION, pid: 100, port: 99999 }
  assert.equal(await readLockfile(fakeSsh([[/cat/, JSON.stringify(lock)]]), OWNERSHIP_ID), null)
  const lock2 = { schemaVersion: LOCKFILE_SCHEMA_VERSION, pid: 100, port: 0 }
  assert.equal(await readLockfile(fakeSsh([[/cat/, JSON.stringify(lock2)]]), OWNERSHIP_ID), null)
})

test('readLockfile accepts a complete owned lock', async () => {
  const lock = ownedLock({ pid: 42, port: 51234 })
  const result = await readLockfile(fakeSsh([[/cat/, JSON.stringify(lock)]]), OWNERSHIP_ID)
  assert.deepEqual(result, lock)
})

test('connect() reuse path does not write a token file', async () => {
  const reuseToken = 'stored-token'
  const lock = ownedLock({ tokenFingerprint: fingerprintToken(reuseToken) })

  const ssh = fakeSsh([
    [/uname/, 'Linux\nx86_64'],
    [/\[ -x/, 'OK'],
    [/cat .*lock\.json/, JSON.stringify(lock)],
    [/kill -0/, 'ALIVE'],
    [/print\("OWNED"/, 'OWNED\n']
  ])

  const result = await connect(connectDeps(ssh, { reuseToken, adoptServedToken: async (_b, t) => t }))
  assert.equal(result.reused, true)
  assert.ok(!ssh.calls.some(c => /sys\.stdin\.buffer\.read/.test(c)), 'reuse must not upload a token file')
})

test('spawnRemoteDashboard fails with update-required when remote lacks --ssh-session-token-file', async () => {
  const ssh = fakeSsh([[/--ssh-session-token-file/, 'NO\n']])

  await assert.rejects(
    () => spawnRemoteDashboard(ssh, { hermesPath: '/x/hermes', profile: '', token: 'tk', ownershipId: OWNERSHIP_ID }),
    (err: any) => {
      assert.match(err.message, /update|upgrade/i)
      assert.equal(err.kind, 'update-required')

      return true
    }
  )
})

test('readLockfile rejects a log path outside the exact ownership and spawn path', async () => {
  const lock = ownedLock({ logPath: '~/.hermes/desktop-ssh/other.log' })
  const ssh = fakeSsh([[/cat .*lock\.json/, JSON.stringify(lock)]])
  assert.equal(await readLockfile(ssh, OWNERSHIP_ID), null)
})

test('cleanupStale never deletes a lock-supplied unexpected log path', async () => {
  const ssh = fakeSsh([[/print\("OWNED"/, 'OWNED\n']])
  await cleanupStale(ssh, OWNERSHIP_ID, ownedLock({ logPath: '~/.hermes/unrelated.log' }))
  assert.ok(!ssh.calls.some(command => command.includes('unrelated.log')))
})

test('pidIsOurDashboard requires an exact nonce option value', async () => {
  const prefix = `/x/hermes serve --isolated --ssh-owner-nonce ${SPAWN_NONCE}ff`
  const suffix = `/x/hermes serve --isolated --ssh-owner-nonce xx${SPAWN_NONCE}`
  assert.equal(await pidIsOurDashboard(fakeSsh([[/print\("OWNED"/, 'FOREIGN\n']]), 5, SPAWN_NONCE, '/x/hermes'), false)
  assert.equal(await pidIsOurDashboard(fakeSsh([[/print\("OWNED"/, 'FOREIGN\n']]), 5, SPAWN_NONCE, '/x/hermes'), false)
})

test('connect removes the token file when a fresh backend fails after returning a pid', async () => {
  const ssh = fakeSsh([
    [/uname/, 'Linux\nx86_64'],
    [/\[ -x/, 'OK'],
    [/cat .*lock\.json/, ''],
    [/grep -q ssh-session-token-file/, 'YES\n'],
    [/python3 -c/, ''],
    [/setsid/, '999\n'],
    [/kill -0 999/, 'DEAD']
  ])

  await assert.rejects(() => connect(connectDeps(ssh)), /exited before announcing/i)
  assert.ok(ssh.calls.some(command => /rm -f .*\.token/.test(command)))
})

test('connect preserves an exact-owned backend when reuse proof transport fails', async () => {
  const reuseToken = 'stored-token'
  const lock = ownedLock({ tokenFingerprint: fingerprintToken(reuseToken) })

  const ssh = fakeSsh([
    [/uname/, 'Linux\nx86_64'],
    [/\[ -x/, 'OK'],
    [/cat .*lock\.json/, JSON.stringify(lock)],
    [/kill -0/, 'ALIVE'],
    [/print\("OWNED"/, 'OWNED\n']
  ])

  await assert.rejects(
    () =>
      connect(
        connectDeps(ssh, {
          reuseToken,
          probeReuseProof: async () => {
            throw new Error('connection reset')
          }
        })
      ),
    (error: any) => error.kind === 'transient-transport-error'
  )
  assert.ok(!ssh.calls.some(command => /kill 333\b/.test(command)))
  assert.ok(!ssh.calls.some(command => /rm -f .*backend\.lock\.json/.test(command)))
})

test('connect replaces an exact-owned backend only after authenticated stale proof', async () => {
  const reuseToken = 'stored-token'
  const lock = ownedLock({ tokenFingerprint: fingerprintToken(reuseToken) })

  const ssh = fakeSsh([
    [/uname/, 'Linux\nx86_64'],
    [/\[ -x/, 'OK'],
    [/cat .*lock\.json/, JSON.stringify(lock)],
    [/kill -0 333/, 'ALIVE'],
    [/print\("OWNED"/, 'OWNED\n'],
    [/grep -q ssh-session-token-file/, 'YES\n'],
    [/python3 -c/, ''],
    [/setsid/, '999\n'],
    [/kill -0 999/, 'ALIVE'],
    [/cat .*\.log/, 'HERMES_DASHBOARD_READY port=43000\n']
  ])

  const result = await connect(
    connectDeps(ssh, {
      reuseToken,
      probeReuseProof: async (_baseUrl, token, nonce) => {
        assert.equal(token, reuseToken)
        assert.equal(nonce, SPAWN_NONCE)

        return 'authenticated-stale'
      },
      adoptServedToken: async () => 'fresh'
    })
  )

  assert.equal(result.reused, false)
  assert.ok(ssh.calls.some(command => /kill 333\b/.test(command)))
})

test('remote SSH ownership capability requires both secure bootstrap flags', async () => {
  let helpProbe = ''

  const supported = fakeSsh([
    [
      /serve --help/,
      command => {
        helpProbe = command

        return 'YES\n'
      }
    ]
  ])

  assert.equal(await remoteSupportsSshOwnership(supported, '/x/hermes'), true)
  assert.match(helpProbe, /ssh-session-token-file/)
  assert.match(helpProbe, /ssh-owner-nonce/)

  const unsupported = fakeSsh([[/serve --help/, 'NO\n']])
  assert.equal(await remoteSupportsSshOwnership(unsupported, '/x/hermes'), false)
})
