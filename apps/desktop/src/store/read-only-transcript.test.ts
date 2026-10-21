import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { $connectionsRegistry } from './connections'
import { $profiles } from './profile'
import {
  $cronRunReadOnlyVerdicts,
  $readOnlyStoredTranscripts,
  clearStoredTranscriptReadOnly,
  isCronRunReadOnly,
  isReadOnlyRuntimeId,
  isStoredTranscriptReadOnly,
  markStoredTranscriptReadOnly,
  readOnlyRuntimeIdFor,
  recordCronRunVerdict,
  resumeWithStoredTranscriptFallback
} from './read-only-transcript'
import { assertSessionOwnerResolved } from './session-owner-resolution'

const registry = (...ids: string[]) =>
  ({
    connections: ids.map(id => ({ id })),
    lastUsed: ids[0] ?? null,
    launchMode: 'primary',
    primary: ids[0] ?? null
  }) as never

beforeEach(() => {
  $connectionsRegistry.set(null)
  $profiles.set([])
  $readOnlyStoredTranscripts.set(new Set())
  $cronRunReadOnlyVerdicts.set(new Map())
})

afterEach(() => {
  delete (window as unknown as { hermesDesktop?: unknown }).hermesDesktop
})

describe('read-only stored-transcript resume (#94724 no-owner recovery)', () => {
  it('opens the stored transcript read-only when the owner fails closed on a 2-connection topology', async () => {
    // The reporter's shape: registry topology with two registered
    // connections, a legacy NULL-owner row — the production gate throws
    // SessionOwnerResolutionError for session.resume (Error B).
    $connectionsRegistry.set(registry('gw-a', 'gw-b'))
    $profiles.set([{ name: 'default' }, { name: 'researcher' }] as never)

    const gatewayDispatch = vi.fn()
    const transcript = { messages: [{ content: 'intact history', role: 'user' }], session_id: 'legacy-1' }

    const outcome = await resumeWithStoredTranscriptFallback(
      'legacy-1',
      async () => {
        // The REAL fail-closed gate, not a re-implementation: an unknown
        // owner under registry topology throws before any dispatch.
        assertSessionOwnerResolved(undefined, { method: 'session.resume', sessionId: 'legacy-1' })
        gatewayDispatch()

        return { session_id: 'runtime-1' }
      },
      async () => transcript
    )

    expect(outcome.mode).toBe('read-only')

    if (outcome.mode === 'read-only') {
      expect(outcome.transcript).toBe(transcript)
      expect(outcome.error.name).toBe('SessionOwnerResolutionError')
    }

    // The whole point of the recovery path: NO gateway routing happened.
    expect(gatewayDispatch).not.toHaveBeenCalled()
    expect(isStoredTranscriptReadOnly('legacy-1')).toBe(true)
  })

  it('stays live (and clears the latch) when the owner resolves', async () => {
    $connectionsRegistry.set(registry('gw-a', 'gw-b'))
    $profiles.set([{ name: 'default' }] as never)
    markStoredTranscriptReadOnly('legacy-2')

    const outcome = await resumeWithStoredTranscriptFallback(
      'legacy-2',
      async () => {
        // Backfilled row: a bare profile is a routable owner in every topology.
        assertSessionOwnerResolved('default', { method: 'session.resume', sessionId: 'legacy-2' })

        return { session_id: 'runtime-2' }
      },
      async () => {
        throw new Error('stored read must not run on the live path')
      }
    )

    expect(outcome.mode).toBe('live')
    expect(isStoredTranscriptReadOnly('legacy-2')).toBe(false)
  })

  it('rethrows non-owner-resolution errors without marking read-only', async () => {
    const boom = new Error('backend exploded')

    await expect(
      resumeWithStoredTranscriptFallback(
        'legacy-3',
        async () => {
          throw boom
        },
        async () => ({ messages: [] })
      )
    ).rejects.toBe(boom)

    expect(isStoredTranscriptReadOnly('legacy-3')).toBe(false)
  })

  it('rethrows the ORIGINAL owner error when even the stored read fails', async () => {
    $connectionsRegistry.set(registry('gw-a', 'gw-b'))
    $profiles.set([{ name: 'default' }, { name: 'researcher' }] as never)

    await expect(
      resumeWithStoredTranscriptFallback(
        'legacy-4',
        async () => {
          assertSessionOwnerResolved(null, { method: 'session.resume', sessionId: 'legacy-4' })

          return {}
        },
        async () => {
          throw new Error('404 stored row missing')
        }
      )
    ).rejects.toMatchObject({ name: 'SessionOwnerResolutionError' })

    expect(isStoredTranscriptReadOnly('legacy-4')).toBe(false)
  })

  it('mints collision-proof synthetic runtime ids and round-trips the latch', () => {
    const id = readOnlyRuntimeIdFor('stored-9')

    expect(isReadOnlyRuntimeId(id)).toBe(true)
    expect(isReadOnlyRuntimeId('stored-9')).toBe(false)
    expect(isReadOnlyRuntimeId(null)).toBe(false)

    markStoredTranscriptReadOnly('stored-9')
    expect(isStoredTranscriptReadOnly('stored-9')).toBe(true)
    clearStoredTranscriptReadOnly('stored-9')
    expect(isStoredTranscriptReadOnly('stored-9')).toBe(false)
  })
})

describe('read-only cron runs (#88443 zombie cron session)', () => {
  it('blocks writes on a cron run whose verdict is read-only', () => {
    recordCronRunVerdict('cron_job-1_20260929_120000', true)

    expect(isCronRunReadOnly('cron_job-1_20260929_120000')).toBe(true)
    // The submit path's gate: one answer for every write surface.
    expect(isStoredTranscriptReadOnly('cron_job-1_20260929_120000')).toBe(true)
    // Unrelated ids stay writable.
    expect(isStoredTranscriptReadOnly('cron_job-1_20260929_130000')).toBe(false)
    expect(isStoredTranscriptReadOnly(null)).toBe(false)
  })

  it('a fresh verdict reopens the run — nothing latches', () => {
    recordCronRunVerdict('cron-flip', true)
    recordCronRunVerdict('cron-flip', false)

    expect(isStoredTranscriptReadOnly('cron-flip')).toBe(false)
  })

  it('survives a live resume clearing the owner-recovery flag', () => {
    recordCronRunVerdict('cron-keep', true)
    markStoredTranscriptReadOnly('cron-keep')

    // Exactly what a successful resume does to the #94724 flag — the cron
    // verdict must NOT be collateral damage, or the guard evaporates the
    // moment the transcript paints.
    clearStoredTranscriptReadOnly('cron-keep')

    expect(isStoredTranscriptReadOnly('cron-keep')).toBe(true)
  })

  it('ignores blank ids', () => {
    recordCronRunVerdict('   ', true)

    expect(isCronRunReadOnly('')).toBe(false)
    expect($cronRunReadOnlyVerdicts.get().size).toBe(0)
  })
})
