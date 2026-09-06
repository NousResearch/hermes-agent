import assert from 'node:assert/strict'

import { test } from 'vitest'

import {
  requireUpdaterHandoff,
  runUpdaterHandoffTransaction,
  stopAndRecordPluginHost
} from './windows-update-orchestration'

test('observes immediately after spawn before awaiting authentication', async () => {
  const events: string[] = []
  let releaseAuthentication!: () => void
  const authentication = new Promise<void>(resolve => {
    releaseAuthentication = resolve
  })

  const pending = runUpdaterHandoffTransaction({
    spawn: () => {
      events.push('spawn')

      return {}
    },
    observe: async () => {
      events.push('observe')

      return { ok: true, message: 'ok' }
    },
    authenticate: async () => {
      events.push('authenticate')
      await authentication

      return true
    },
    commit: () => {
      events.push('commit')

      return 'done'
    },
    restore: () => {
      events.push('restore')
    },
    authenticationError: 'not adopted'
  })

  await Promise.resolve()
  assert.deepEqual(events, ['spawn', 'observe', 'authenticate'])
  releaseAuthentication()
  assert.deepEqual(await pending, { ok: true, value: 'done' })
  assert.deepEqual(events, ['spawn', 'observe', 'authenticate', 'commit'])
})

test('awaits restoration before returning a spawn failure', async () => {
  const events: string[] = []
  let finishRestore!: () => void
  let markRestoreStarted!: () => void
  const restoring = new Promise<void>(resolve => {
    finishRestore = resolve
  })
  const restoreStarted = new Promise<void>(resolve => {
    markRestoreStarted = resolve
  })

  const pending = runUpdaterHandoffTransaction({
    spawn: () => ({}),
    observe: async () => ({ ok: false, message: 'early exit' }),
    authenticate: async () => true,
    commit: () => 'unreachable',
    restore: async () => {
      events.push('restore-start')
      markRestoreStarted()
      await restoring
      events.push('restore-end')
    },
    authenticationError: 'not adopted'
  })

  await restoreStarted
  assert.deepEqual(events, ['restore-start'])
  finishRestore()
  assert.deepEqual(await pending, { ok: false, error: 'updater-spawn-failed', message: 'early exit' })
  assert.deepEqual(events, ['restore-start', 'restore-end'])
})

test('waits for successor-marker clearance before restoring backends', async () => {
  let successorMarker = true
  const events: string[] = []

  const pending = runUpdaterHandoffTransaction({
    spawn: () => ({}),
    observe: async () => ({ ok: true, message: 'ok' }),
    authenticate: async () => false,
    commit: () => 'unreachable',
    restore: async () => {
      events.push('restore-wait')

      while (successorMarker) {
        await new Promise(resolve => setTimeout(resolve, 0))
      }
      events.push('backends-restored')
    },
    authenticationError: 'not adopted'
  })

  await new Promise(resolve => setTimeout(resolve, 0))
  assert.deepEqual(events, ['restore-wait'])
  successorMarker = false
  assert.deepEqual(await pending, { ok: false, error: 'update-handoff-unacknowledged', message: 'not adopted' })
  assert.deepEqual(events, ['restore-wait', 'backends-restored'])
})

test('restores after a thrown spawn error and rethrows it', async () => {
  let restored = false
  await assert.rejects(
    runUpdaterHandoffTransaction({
      spawn: () => {
        throw new Error('spawn failed')
      },
      observe: async () => ({ ok: true, message: 'ok' }),
      authenticate: async () => true,
      commit: () => 'unreachable',
      restore: async child => {
        assert.equal(child, null)
        restored = true
      },
      authenticationError: 'not adopted'
    }),
    /spawn failed/
  )
  assert.equal(restored, true)
})

test('does not retry a restoration operation that throws', async () => {
  let restores = 0
  await assert.rejects(
    runUpdaterHandoffTransaction({
      spawn: () => ({}),
      observe: async () => ({ ok: false, message: 'early exit' }),
      authenticate: async () => true,
      commit: () => 'unreachable',
      restore: async () => {
        restores += 1
        throw new Error('restore failed')
      },
      authenticationError: 'not adopted'
    }),
    /restore failed/
  )
  assert.equal(restores, 1)
})

test('a failed started handoff cannot fall through to an in-process updater', () => {
  assert.throws(
    () =>
      requireUpdaterHandoff({
        ok: false,
        error: 'update-handoff-unacknowledged',
        message: 'child ownership is ambiguous'
      }),
    /child ownership is ambiguous/
  )
  assert.equal(requireUpdaterHandoff({ ok: true, value: 'quit desktop' }), 'quit desktop')
})

test('compensates a stopped plugin host when its recovery record cannot be persisted', async () => {
  const host = { pid: 42 }
  const compensated: unknown[] = []

  assert.equal(
    await stopAndRecordPluginHost({
      terminate: async () => ({ terminated: true, host }),
      record: () => false,
      compensate: async value => {
        compensated.push(value)

        return true
      }
    }),
    false
  )
  assert.deepEqual(compensated, [host])
})

test('reports when both the plugin-host recovery record and compensating restart fail', async () => {
  const host = { pid: 42 }
  const failures: unknown[] = []

  assert.equal(
    await stopAndRecordPluginHost({
      terminate: async () => ({ terminated: true, host }),
      record: () => false,
      compensate: async () => false,
      onRecoveryFailure: value => {
        failures.push(value)
      }
    }),
    false
  )
  assert.deepEqual(failures, [host])
})
