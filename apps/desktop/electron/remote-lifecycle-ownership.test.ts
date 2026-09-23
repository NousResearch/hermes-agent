import assert from 'node:assert/strict'
import { exec as execCallback } from 'node:child_process'
import { mkdir, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import { promisify } from 'node:util'

import { test } from 'vitest'

import {
  buildSpawnCommand,
  cleanupStale,
  connect,
  fingerprintToken,
  isLockfileSkew,
  LOCKFILE_SCHEMA_VERSION,
  pidIsOurDashboard,
  probeHermesVersion,
  readLockfile,
  remoteSupportsSshOwnership,
  spawnLogPath,
  spawnRemoteDashboard,
  spawnTokenPath
} from './remote-lifecycle'
import { connectDeps, fakeSsh, ownedLock, OWNERSHIP_ID, SPAWN_NONCE } from './remote-lifecycle.test-helpers'

const exec = promisify(execCallback)

test('buildSpawnCommand includes --ssh-session-token-file when tokenFilePath is provided', () => {
  const cmd = buildSpawnCommand('/x/hermes', 'work', {
    tokenFilePath: `~/.hermes/desktop-ssh/${OWNERSHIP_ID}/${SPAWN_NONCE}.token`,
    logPath: spawnLogPath(OWNERSHIP_ID, SPAWN_NONCE),
    spawnNonce: SPAWN_NONCE
  })

  assert.match(cmd, /--ssh-session-token-file/)
  assert.match(cmd, /\.hermes\/desktop-ssh\//)
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

      if (/python3 -c/.test(cmd) && !/fcntl\.flock/.test(cmd)) {
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

      if (/python3 -c/.test(cmd) && !/fcntl\.flock/.test(cmd)) {
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
  const uploadCmd = calls.find(c => /python3 -c/.test(c) && !/fcntl\.flock/.test(c))
  assert.ok(uploadCmd, 'must use python3 -c for token upload')
  assert.match(uploadCmd, /O_EXCL/, 'upload must use O_EXCL to reject existing files')
  assert.match(uploadCmd, /O_NOFOLLOW/, 'upload must use O_NOFOLLOW to reject symlinks')
  assert.match(uploadCmd, /O_WRONLY/, 'upload must open write-only')
  assert.match(uploadCmd, /dir_fd=dd/, 'upload must create relative to the opened parent directory')
  assert.match(uploadCmd, /os\.fstat\(dd\)/, 'upload must validate the opened parent directory')
  assert.ok(!uploadCmd.includes('tk'), 'token must not appear in the upload command')
})

test('readLockfile treats a lock with non-integer pid as skew', async () => {
  const lock = { schemaVersion: LOCKFILE_SCHEMA_VERSION, pid: 'not-a-number', port: 8080 }
  assert.equal(isLockfileSkew(await readLockfile(fakeSsh([[/cat/, JSON.stringify(lock)]]), OWNERSHIP_ID)), true)
})

test('readLockfile treats a lock with pid <= 0 as skew', async () => {
  const lock = { schemaVersion: LOCKFILE_SCHEMA_VERSION, pid: -1, port: 8080 }
  assert.equal(isLockfileSkew(await readLockfile(fakeSsh([[/cat/, JSON.stringify(lock)]]), OWNERSHIP_ID)), true)
})

test('readLockfile treats a lock with port out of range as skew', async () => {
  const lock = { schemaVersion: LOCKFILE_SCHEMA_VERSION, pid: 100, port: 99999 }
  assert.equal(isLockfileSkew(await readLockfile(fakeSsh([[/cat/, JSON.stringify(lock)]]), OWNERSHIP_ID)), true)
  const lock2 = { schemaVersion: LOCKFILE_SCHEMA_VERSION, pid: 100, port: 0 }
  assert.equal(isLockfileSkew(await readLockfile(fakeSsh([[/cat/, JSON.stringify(lock2)]]), OWNERSHIP_ID)), true)
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

test('readLockfile treats a log path outside the exact ownership and spawn path as skew', async () => {
  const lock = ownedLock({ logPath: '~/.hermes/desktop-ssh/other.log' })
  const ssh = fakeSsh([[/cat .*lock\.json/, JSON.stringify(lock)]])
  assert.equal(isLockfileSkew(await readLockfile(ssh, OWNERSHIP_ID)), true)
})

test('cleanupStale never deletes a lock-supplied unexpected log path', async () => {
  const ssh = fakeSsh([
    [/print\("OWNED"/, 'OWNED\n'],
    [cmd => /pidfd_open/.test(cmd), 'TERMINATED\n']
  ])

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
    [cmd => /pidfd_open/.test(cmd), 'TERMINATED\n'],
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
  // The kill goes through main's cleanupStale (ownership-proved SIGTERM with
  // SIGKILL escalation, #91668) — the PR's python re-proof command shape is
  // used by the managed-update path (terminateOwnedDashboardForUpdate), not
  // by connect's stale replacement. Assert the CONTRACT: the owned pid was
  // signalled and the record reclaimed.
  assert.ok(ssh.calls.some(command => /kill 333\b/.test(command)))
  assert.ok(ssh.calls.some(command => /rm -f .*backend\.lock\.json/.test(command)))
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

test.skipIf(process.platform === 'win32')(
  'capability probe survives a zsh login shell on the remote (#111949)',
  async t => {
    // sshd runs the remote command under the account's LOGIN shell. A bare
    // `set -m` is fatal in a non-interactive zsh, so the watchdog-wrapped probe
    // used to return nothing and a current remote was reported as unsupported.
    const zsh = await exec('command -v zsh || true').then(r => r.stdout.trim())

    // CI installs zsh (js-tests.yml); locally a missing zsh must show as a
    // skip, not a pass, or a wrapper regression stays green unnoticed.
    if (!zsh) {
      t.skip('zsh not installed')

      return
    }

    const dir = await mkdtemp(path.join(os.tmpdir(), 'hermes-zsh-probe-'))

    try {
      const hermes = path.join(dir, 'hermes')
      await writeFile(hermes, '#!/bin/sh\necho "--ssh-session-token-file --ssh-owner-nonce"\n', { mode: 0o700 })

      const ssh = { exec: async (command: string) => (await exec(command, { shell: zsh })).stdout }

      assert.equal(await remoteSupportsSshOwnership(ssh, hermes), true)
    } finally {
      await rm(dir, { recursive: true, force: true })
    }
  }
)

test('probes run under the remote watchdog so a hung CLI cannot orphan (#110478)', async () => {
  let versionProbe = ''

  const versionSsh = fakeSsh([
    [
      /--version/,
      (cmd: string) => {
        versionProbe = cmd

        return 'Hermes Agent v0.18.2 (abc123)\n'
      }
    ]
  ])

  assert.equal(await probeHermesVersion(versionSsh, '/x/hermes'), 'Hermes Agent v0.18.2 (abc123)')
  assert.ok(versionProbe.includes('kill -9'), 'version probe wrapped in the remote watchdog')

  let helpProbe = ''

  const helpSsh = fakeSsh([
    [
      /serve --help/,
      (cmd: string) => {
        helpProbe = cmd

        return 'YES\n'
      }
    ]
  ])

  assert.equal(await remoteSupportsSshOwnership(helpSsh, '/x/hermes'), true)
  assert.ok(helpProbe.includes('kill -9'), 'ownership probe wrapped in the remote watchdog')
  assert.ok(
    /\$\(.*\(.*serve --help.*\) <\/dev\/null &/.test(helpProbe),
    'watchdog nested around the inner serve --help'
  )
})

test('cleanupStale escalates to SIGKILL when the backend survives the graceful wait (#91668 quit-during-active-turn)', async () => {
  // A serve mid-turn (in-flight LLM call, live MCP children) can ride out
  // SIGTERM well past the 5s graceful wait. Before-quit races the whole
  // teardown against 6s and then closes SSH — so a give-up here reparents
  // the still-running backend to pid 1: exactly the #91668 leak. The
  // graceful-wait failure must escalate to SIGKILL and still drop the lock.
  const ssh = fakeSsh([
    [/print\("OWNED"/, 'OWNED\n'],
    [(cmd: string) => /kill 9 &&/.test(cmd), new Error('exit 1: pid alive after graceful wait')]
  ])

  await cleanupStale(ssh, OWNERSHIP_ID, {
    pid: 9,
    spawnNonce: SPAWN_NONCE,
    hermesPath: '/x/hermes',
    logPath: spawnLogPath(OWNERSHIP_ID, SPAWN_NONCE)
  })

  assert.ok(
    ssh.calls.some(c => /kill -9 9\b/.test(c)),
    'must escalate to SIGKILL after the graceful wait fails'
  )
  assert.ok(
    ssh.calls.some(c => /rm -f .*backend\.lock\.json/.test(c)),
    'lockfile must still be dropped after the forced kill'
  )
})

test('cleanupStale keeps the lockfile when even SIGKILL cannot confirm the pid died', async () => {
  const ssh = fakeSsh([
    [/print\("OWNED"/, 'OWNED\n'],
    [(cmd: string) => /kill 9 &&/.test(cmd), new Error('exit 1: pid alive after graceful wait')],
    [(cmd: string) => /kill -9 9\b/.test(cmd), new Error('exit 1: unkillable (D-state)')]
  ])

  await assert.rejects(
    cleanupStale(ssh, OWNERSHIP_ID, {
      pid: 9,
      spawnNonce: SPAWN_NONCE,
      hermesPath: '/x/hermes',
      logPath: spawnLogPath(OWNERSHIP_ID, SPAWN_NONCE)
    }),
    /Could not terminate/
  )

  // The record must survive so the next connect's reap pass retries.
  assert.ok(!ssh.calls.some(c => /rm -f .*backend\.lock\.json/.test(c)))
})
test.skipIf(process.platform === 'win32')(
  'buildSpawnCommand quotes expandRemotePath fragments exactly once (real sh parse)',
  async () => {
    // expandRemotePath() output is pre-quoted; a second shq() ships literal quote
    // characters to the remote python. Parse the composed command with a real sh,
    // as the remote login shell does, and require every path to come out clean.
    const cmd = buildSpawnCommand('/x/hermes', 'work', {
      hermesHome: '~/.hermes',
      logPath: spawnLogPath(OWNERSHIP_ID, SPAWN_NONCE),
      ownershipId: OWNERSHIP_ID,
      reservationNonce: SPAWN_NONCE,
      spawnNonce: SPAWN_NONCE,
      tokenFilePath: spawnTokenPath(OWNERSHIP_ID, SPAWN_NONCE),
      lockMetadata: { ownershipId: OWNERSHIP_ID, spawnNonce: SPAWN_NONCE }
    })

    // Capture the argv a remote shell would hand to python3, via a shim on PATH.
    const root = await mkdtemp(path.join(os.tmpdir(), 'hermes-argv-shim-'))

    try {
      const shimDir = path.join(root, 'shim')
      const fakeHome = path.join(root, 'home')
      await mkdir(shimDir)
      await mkdir(fakeHome)
      const argvFile = path.join(shimDir, 'argv')
      await writeFile(path.join(shimDir, 'python3'), `#!/bin/sh\nprintf '%s\\0' "$@" > '${argvFile}'\n`, {
        mode: 0o755
      })
      await exec(cmd, {
        env: { ...process.env, PATH: `${shimDir}:${process.env.PATH}`, HOME: fakeHome }
      })
      const argv = (await readFile(argvFile, 'utf8')).split('\0')

      // argv: ['-c', <mutex script>, <mutex path>, <payload>]
      assert.equal(
        argv[2],
        `${fakeHome}/.hermes/.hermes-update-in-progress.mutex`,
        'mutex path must reach python fully expanded, with no quote characters'
      )

      // The payload assigns reservation/lock/owner_file before its mkdir loop.
      // Evaluate only that prefix the way the remote sh does; never the loop itself.
      const payload = argv[3]
      const loopStart = payload.indexOf('i=0;')
      assert.ok(loopStart > 0, 'payload prefix sentinel missing')

      const { stdout } = await exec(
        `${payload.slice(0, loopStart)} printf '%s\\n' "$reservation" "$lock" "$owner_file"`,
        {
          env: { ...process.env, HOME: fakeHome }
        }
      )

      const [reservation, lock, ownerFile] = stdout.split('\n')
      const base = `${fakeHome}/.hermes/desktop-ssh/${OWNERSHIP_ID}`
      assert.equal(reservation, `${base}/.connect.lock`)
      assert.equal(lock, `${base}/backend.lock.json`)
      assert.equal(ownerFile, `${base}/.connect.lock/owner`)
    } finally {
      await rm(root, { recursive: true, force: true })
    }
  }
)

// The liveness and ownership probes answer over the same SSH channel that is
// often mid-teardown right after the served token resolved. An exec that
// returns neither sentinel is indeterminate (#111810): read as DEAD it tore
// down a live backend; read as FOREIGN it skipped the reap while removing the
// lockfile — one orphaned `serve --isolated` per failed attempt.
test('connect() does not declare a live dashboard dead when the liveness probe answers nothing once', async () => {
  let liveness = 0

  const ssh = fakeSsh([
    [/uname/, 'Linux\nx86_64'],
    [/\[ -x/, 'OK'],
    [/cat .*lock\.json/, ''],
    [/grep -q ssh-session-token-file/, 'YES\n'],
    [/python3 -c/, ''],
    [/printf '%s\\n'/, ''],
    [/setsid/, '777\n'],
    [(cmd: string) => /kill -0 777/.test(cmd) && !cmd.includes('while'), () => (liveness++ === 0 ? '' : 'ALIVE\n')],
    [/cat .*\.log/, 'HERMES_DASHBOARD_READY port=51999\n']
  ])

  const result = await connect(connectDeps(ssh, { platform: { os: 'Linux', arch: 'x86_64' } }))

  assert.equal(result.reused, false)
  assert.equal(result.pid, 777)
  assert.ok(
    !ssh.calls.some(c => /(^|[^-\d])kill(?: -\w+)? 777\b/.test(c) && !/kill -0/.test(c)),
    'must not reap a live backend'
  )
})

test('cleanupStale reaps after one lost ownership answer and keeps the lockfile when none ever settles', async () => {
  let ownership = 0

  const flaky = fakeSsh([
    [/print\("OWNED"/, () => (ownership++ === 0 ? '' : 'OWNED\n')],
    [/kill 777 &&/, 'TERMINATED\n']
  ])

  await cleanupStale(flaky, OWNERSHIP_ID, ownedLock({ pid: 777 }))
  assert.ok(
    flaky.calls.some(c => /kill 777 &&/.test(c)),
    'must reap the owned backend'
  )
  assert.ok(flaky.calls.some(c => /rm -f .*backend\.lock\.json/.test(c)))

  const silent = fakeSsh([[/print\("OWNED"/, '']])

  await assert.rejects(
    cleanupStale(silent, OWNERSHIP_ID, ownedLock({ pid: 777 })),
    (error: any) => error.kind === 'transient-transport-error'
  )

  assert.ok(
    !silent.calls.some(c => /(^|[^-\d])kill(?: -\w+)? 777\b/.test(c) && !/kill -0/.test(c)),
    'must not kill unproven'
  )
  assert.ok(
    !silent.calls.some(c => /rm -f .*backend\.lock\.json/.test(c)),
    'record must survive for the next connect to reap'
  )
})

test('connect() post-spawn cleanup that cannot prove ownership keeps the original boot error', async () => {
  const boot: any = new Error('dashboard never answered')
  boot.kind = 'boot-failed'

  const ssh = fakeSsh([
    [/uname/, 'Linux\nx86_64'],
    [/\[ -x/, 'OK'],
    [/cat .*lock\.json/, ''],
    [/grep -q ssh-session-token-file/, 'YES\n'],
    [/python3 -c/, ''],
    [/printf '%s\\n'/, ''],
    [/setsid/, '777\n'],
    [/kill -0 777/, 'ALIVE\n'],
    [/cat .*\.log/, 'HERMES_DASHBOARD_READY port=51999\n'],
    [/print\("OWNED"/, '']
  ])

  await assert.rejects(
    connect(
      connectDeps(ssh, {
        platform: { os: 'Linux', arch: 'x86_64' },
        waitForHermes: async () => {
          throw boot
        }
      })
    ),
    (error: any) => error === boot && error.cleanupCause?.kind === 'transient-transport-error'
  )

  assert.ok(
    !ssh.calls.some(c => /rm -f .*backend\.lock\.json/.test(c)),
    'record must survive for the next connect to reap'
  )
})
