import { execFileSync } from 'node:child_process'
import { mkdtempSync, rmSync, writeFileSync } from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { describe, expect, it, vi } from 'vitest'

import * as recovery from './pooled-ssh-dispatch-recovery'
import { connect, fingerprintToken, spawnLogPath } from './remote-lifecycle'
import {
  ensureHealthyPooledRemoteBackendForDispatch,
  RemoteLivenessTracker,
  revalidatePooledRemoteBackends
} from './remote-liveness'

const ownershipId = '0123456789abcdef0123456789abcdef'
const spawnNonce = '0123456789abcdef'

function ownedLock(overrides: Record<string, unknown> = {}) {
  return {
    schemaVersion: 2,
    protocolVersion: 1,
    ownershipId,
    spawnNonce,
    pid: 333,
    port: 40000,
    profile: '',
    hermesPath: '/home/alice/.local/bin/hermes',
    hermesHome: '/home/alice/.hermes',
    logPath: spawnLogPath(ownershipId, spawnNonce),
    tokenFingerprint: fingerprintToken('stored-token'),
    startedAt: '2026-07-14T00:00:00.000Z',
    creationTime: 'linux:123456',
    ...overrides
  }
}

// Command-selection fixture only: no remote server or real signals are used.
function remote(overrides: Record<string, unknown> = {}, verdict = 'ALIVE') {
  let lock: any = ownedLock(overrides)
  const effects: string[] = []

  const ssh = {
    exec: vi.fn(async (command: string) => {
      if (/setsid|nohup/.test(command)) {
        effects.push('spawn')

        return '777\n'
      }

      if (/^kill (?!-0)/.test(command)) {
        effects.push('signal')

        return ''
      }

      if (command.startsWith('rm -f')) {
        effects.push('unlink')

        return ''
      }

      const staticReply = [
        ['.hermes-update-in-progress', 'CLEAR'],
        ['HERMES_HOME', '/home/alice/.hermes\n'],
        ['command -v hermes', '/home/alice/.local/bin/hermes\n'],
        ['[ -x', 'OK'],
        ['--version', 'Hermes Agent v0.18.2\n']
      ].find(([needle]) => command.includes(needle))

      if (staticReply) {
        return staticReply[1]
      }

      if (command.includes('cat ') && command.includes('backend.lock.json')) {
        return lock === null ? '' : JSON.stringify(lock)
      }

      if (command.includes('os.kill(pid,0)')) {
        return verdict
      }

      if (command.includes('kill -0 333')) {
        return verdict
      }

      if (command.includes('kill -0')) {
        return 'ALIVE'
      }

      if (command.includes('print("OWNED"')) {
        return 'OWNED'
      }

      if (command.includes('value="linux:"')) {
        return 'linux:123456'
      }

      if (command.includes('grep -q ssh-session-token-file')) {
        return 'YES'
      }

      if (command.includes('cat ') && command.includes('.log')) {
        return 'HERMES_DASHBOARD_READY port=51999\n'
      }

      if (command.includes('mv -f')) {
        effects.push('publish')
      }

      return ''
    })
  }

  const deps = {
    ssh,
    platform: { os: 'Linux', arch: 'x86_64' },
    ownershipId,
    reuseToken: 'stored-token',
    automaticRecovery: true,
    requireOwnerRecord: true,
    forward: vi.fn(async () => {}),
    cancelForward: vi.fn(async () => {}),
    pickLocalPort: async () => 50001,
    waitForHermes: async () => {},
    probeReuseProof: vi.fn(async () => 'authenticated-stale'),
    adoptServedToken: async (_baseUrl: string, token: string) => token,
    readyTimeoutMs: 100
  }

  return {
    deps,
    effects,
    ssh,
    replaceLock: (replacement: any) => {
      lock = replacement
    }
  }
}

describe('background SSH pool retirement', () => {
  it.each([false, true])(
    'awaits stopped-owner permission before dropping after three failures: %s',
    async permitted => {
      const connection = { baseUrl: 'http://127.0.0.1:50001', remoteKind: 'ssh' }
      const entry = { connectionPromise: Promise.resolve(connection), remoteBaseUrl: connection.baseUrl }
      const pool = new Map([['scope', entry]])
      const state = { pid: 333 }
      const sshStates = new Map([['scope', state]])
      const releaseTransport = vi.fn(async () => {})

      const stopBackend = vi.fn(async (key: string) => {
        await Promise.resolve()
        pool.delete(key)
      })

      const forEntry = recovery.createPooledBackendRecovery({
        pool,
        sshStates,
        scopeFor: key => key,
        proveStopped: async () => permitted,
        markRecovery: () => {},
        retireEntry: stopBackend,
        releaseTransport,
        drain: async (_key, cleanup) => cleanup()
      })

      const canRetire = vi.fn(async (key, probed, resolved) =>
        forEntry(key, probed, resolved?.remoteKind === 'ssh').canRetire()
      )

      const tracker = new RemoteLivenessTracker()

      const probe = vi.fn(async () => {
        throw new Error('probe timed out')
      })

      const run = () =>
        revalidatePooledRemoteBackends({
          entries: pool.entries(),
          currentEntry: key => pool.get(key),
          canRetire,
          probe,
          stopBackend,
          tracker,
          log: () => {}
        })

      await expect(run()).resolves.toEqual({ dropped: [] })
      await expect(run()).resolves.toEqual({ dropped: [] })
      await expect(run()).resolves.toEqual({ dropped: permitted ? ['scope'] : [] })
      expect(canRetire).toHaveBeenCalledWith('scope', entry, connection)
      expect(probe).toHaveBeenCalledTimes(3)
      expect(pool.get('scope')).toBe(permitted ? undefined : entry)
      expect(stopBackend).toHaveBeenCalledTimes(permitted ? 1 : 0)
      expect(sshStates.get('scope')).toBe(state)
      expect(releaseTransport).not.toHaveBeenCalled()
    }
  )

  it('does not drop a replacement installed while awaiting retirement permission', async () => {
    const connection = { baseUrl: 'http://127.0.0.1:50001' }
    const entry = { connectionPromise: Promise.resolve(connection), remoteBaseUrl: connection.baseUrl }
    const replacement = { ...entry }
    const pool = new Map([['scope', entry]])
    const stopBackend = vi.fn()
    const tracker = new RemoteLivenessTracker()

    const canRetire = vi.fn(async () => {
      await Promise.resolve()
      pool.set('scope', replacement)

      return true
    })

    const run = () =>
      revalidatePooledRemoteBackends({
        entries: pool.entries(),
        currentEntry: key => pool.get(key),
        canRetire,
        probe: async () => {
          throw new Error('probe timed out')
        },
        stopBackend,
        tracker,
        log: () => {}
      })

    await run()
    await run()
    await expect(run()).resolves.toEqual({ dropped: [] })
    expect(canRetire).toHaveBeenCalledOnce()
    expect(pool.get('scope')).toBe(replacement)
    expect(stopBackend).not.toHaveBeenCalled()
  })
})

describe('pooled SSH dispatch retirement', () => {
  it('wires a rejected dispatch through the registry guard without evicting its live owner', async () => {
    const connection = { baseUrl: 'http://127.0.0.1:50001', mode: 'remote' }
    const entry = { connectionPromise: Promise.resolve(connection) }
    const state = { pid: 333 }
    const pool = new Map([['scope', entry]])
    const sshStates = new Map([['scope', state]])
    const releaseTransport = vi.fn(async () => {})
    const failure = new Error('dispatch health timed out')

    const forEntry = recovery.createPooledBackendRecovery({
      pool,
      sshStates,
      scopeFor: key => key,
      proveStopped: async () => false,
      markRecovery: () => {},
      retireEntry: async key => {
        pool.delete(key)
      },
      releaseTransport,
      drain: async (_key, cleanup) => cleanup()
    })

    const reconnect = vi.fn()

    await expect(
      ensureHealthyPooledRemoteBackendForDispatch({
        connectionPromise: entry.connectionPromise,
        currentConnectionPromise: () => pool.get('scope')?.connectionPromise || null,
        probe: async () => {
          throw failure
        },
        reconnect,
        ...forEntry('scope', entry, true)
      })
    ).rejects.toBe(failure)
    expect(pool.get('scope')).toBe(entry)
    expect(sshStates.get('scope')).toBe(state)
    expect(releaseTransport).not.toHaveBeenCalled()
    expect(reconnect).not.toHaveBeenCalled()
  })

  it('preserves a published master while releasing a failed reuse attempt forward', async () => {
    const ssh = { cancelForward: vi.fn(async () => {}), close: vi.fn(async () => {}) }
    await recovery.releaseSshTransport({ ssh, localPort: 50002, remotePort: 40000 }, { preserveMaster: true })
    expect(ssh.cancelForward).toHaveBeenCalledWith(50002, 40000)
    expect(ssh.close).not.toHaveBeenCalled()
  })

  it('releases only the captured transport after a stopped-owner proof', async () => {
    const ssh = { exec: vi.fn(), cancelForward: vi.fn(async () => {}), close: vi.fn(async () => {}) }
    const state = { ssh, localPort: 50001, remotePort: 40000 }

    const guard = recovery.createPooledSshDispatchRecovery({
      state,
      currentState: () => state,
      isCurrentEntry: () => true,
      proveStopped: async () => true,
      markRecovery: () => {},
      retireEntry: async () => {},
      releaseTransport: captured => recovery.releaseSshTransport(captured),
      drain: async cleanup => cleanup()
    })

    expect(await guard.canRetire()).toBe(true)
    await guard.retire()
    expect(ssh.cancelForward).toHaveBeenCalledWith(50001, 40000)
    expect(ssh.close).toHaveBeenCalledOnce()
    expect(ssh.exec).not.toHaveBeenCalled()
  })

  it.each([undefined, { pid: 444 }])('does not mark a disconnected or replaced SSH state: %j', async replacement => {
    const state = { pid: 333 }
    let current: typeof state | undefined = state
    const markRecovery = vi.fn()
    const proveStopped = vi.fn(async () => true)
    const retireEntry = vi.fn(async () => {})
    const releaseTransport = vi.fn(async () => {})

    const guard = recovery.createPooledSshDispatchRecovery({
      state,
      currentState: () => current,
      isCurrentEntry: () => true,
      proveStopped,
      markRecovery,
      retireEntry,
      releaseTransport,
      drain: async cleanup => cleanup()
    })

    current = replacement
    expect(await guard.canRetire()).toBe(false)
    await guard.retire()
    expect(markRecovery).not.toHaveBeenCalled()
    expect(proveStopped).not.toHaveBeenCalled()
    expect(retireEntry).not.toHaveBeenCalled()
    expect(releaseTransport).not.toHaveBeenCalled()
  })

  it('keeps the pool entry and transport when the captured owner is not confirmed stopped', async () => {
    const state = { pid: 333 }
    const markRecovery = vi.fn()
    const retireEntry = vi.fn(async () => {})
    const releaseTransport = vi.fn(async () => {})
    const drain = vi.fn(async cleanup => cleanup())

    const guard = recovery.createPooledSshDispatchRecovery({
      state,
      currentState: () => state,
      isCurrentEntry: () => true,
      proveStopped: async () => false,
      markRecovery,
      retireEntry,
      releaseTransport,
      drain
    })

    expect(await guard.canRetire()).toBe(false)
    await guard.retire()
    expect(markRecovery).toHaveBeenCalledOnce()
    expect(retireEntry).not.toHaveBeenCalled()
    expect(releaseTransport).not.toHaveBeenCalled()
  })
})

describe('automatic SSH dispatch recovery', () => {
  it('permits an initial reservation only while its ownership record is absent', () => {
    const root = mkdtempSync(path.join(os.tmpdir(), 'ssh-initial-reservation-'))
    const record = path.join(root, 'backend.lock.json')

    const run = () =>
      execFileSync(
        process.env.HERMES_PYTHON || 'python3',
        ['-c', recovery.stoppedOwnerReservationProgram(null), record],
        { encoding: 'utf8' }
      ).trim()

    try {
      expect(run()).toBe('ABSENT')
      writeFileSync(record, JSON.stringify(ownedLock()))
      expect(run()).toBe('REFUSED')
    } finally {
      rmSync(root, { recursive: true, force: true })
    }
  })

  it('remembers a live owner rejected on the first connect before its record disappears', async () => {
    const test = remote({ profile: 'another-profile' })
    const observed = new Set<string>()

    const deps = {
      ...test.deps,
      requireOwnerRecord: false,
      onOwnerObserved: () => observed.add('scope')
    }

    await expect(connect(deps)).rejects.toMatchObject({ kind: 'remote-owner-preserved' })
    test.replaceLock(null)
    deps.requireOwnerRecord = observed.has('scope')
    await expect(connect(deps)).rejects.toMatchObject({ kind: 'remote-owner-preserved' })
    expect(observed.has('scope')).toBe(true)
    expect(test.effects).toEqual([])
    expect(test.deps.forward).not.toHaveBeenCalled()
  })

  it('allows a first connection with no previously observed owner to spawn', async () => {
    const test = remote()
    test.replaceLock(null)
    test.deps.requireOwnerRecord = false

    await expect(connect(test.deps)).resolves.toMatchObject({ reused: false, pid: 777 })
  })

  it('refuses a replacement owner at the reservation even when its PID is inaccessible', () => {
    const root = mkdtempSync(path.join(os.tmpdir(), 'ssh-recovery-'))
    const record = path.join(root, 'backend.lock.json')
    const replacement = ownedLock({ pid: 444, spawnNonce: 'fedcba9876543210' })
    writeFileSync(record, JSON.stringify(replacement))

    try {
      const program = recovery.stoppedOwnerReservationProgram(ownedLock())

      const output = execFileSync(
        process.env.HERMES_PYTHON || 'python3',
        [
          '-c',
          'import os,sys\nfrom unittest.mock import patch\n' +
            `sys.argv=["probe",${JSON.stringify(record)}]\n` +
            'with patch("os.kill",side_effect=PermissionError("EPERM")):\n' +
            ` exec(${JSON.stringify(program)})\n`
        ],
        { encoding: 'utf8' }
      )

      expect(output.trim()).toBe('REFUSED')
    } finally {
      rmSync(root, { recursive: true, force: true })
    }
  })

  it.each([null, ownedLock({ protocolVersion: 2 }), ownedLock({ creationTime: 'invalid' })])(
    'does not treat a missing or incompatible record as permission to spawn: %j',
    async lock => {
      const test = remote()
      test.replaceLock(lock)

      await expect(connect(test.deps)).rejects.toMatchObject({ kind: 'remote-owner-preserved' })
      expect(test.effects).toEqual([])
    }
  )

  it('rebuilds a confirmed stopped owner without stale cleanup before the spawn reservation', async () => {
    const test = remote({}, 'DEAD')

    await expect(connect(test.deps)).resolves.toMatchObject({ pid: 777, reused: false })
    expect(test.effects).toEqual(['spawn', 'publish', 'publish'])
  })

  it.each([
    { tokenFingerprint: fingerprintToken('another-token') },
    { profile: 'another-profile' },
    { hermesPath: '/other/hermes' },
    { hermesHome: '/other/home' },
    { devServer: 'http://other-renderer' },
    { port: 0 }
  ])('preserves a live owner whose settings cannot be reused: %j', async overrides => {
    const test = remote(overrides)

    await expect(connect(test.deps)).rejects.toMatchObject({ kind: 'remote-owner-preserved' })
    expect(test.effects).toEqual([])
    expect(test.deps.forward).not.toHaveBeenCalled()
  })

  it('preserves an alive owned dashboard when the authenticated reuse proof is stale', async () => {
    const test = remote()

    await expect(connect(test.deps)).rejects.toMatchObject({ kind: 'remote-owner-preserved' })
    expect(test.effects).toEqual([])
    expect(test.deps.cancelForward).toHaveBeenCalledWith(50001, 40000)
  })
})
