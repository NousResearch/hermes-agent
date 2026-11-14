import { act, cleanup } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { getApiRequestProfile, setApiRequestConnection, setApiRequestProfile } from '@/api/client'
import { CronRunGateSupersededError, refreshCronRunWriteGate } from '@/app/cron/open-cron-run'
import { getSession } from '@/hermes'
import { $composerDraft, setComposerDraft } from '@/store/composer'
import { $notifications } from '@/store/notifications'
import { $cronRunReadOnlyVerdicts, hasCronRunVerdict } from '@/store/read-only-transcript'
import type { SessionOwnerScope } from '@/store/session-request-router'

import { resolveSessionOwner } from '../use-session-actions/utils'

import { actRender, Harness, type HarnessHandle, sessionInfo } from './index-test-utils'
import { clearSingleFlightSessionResumeState } from './single-flight-resume'

vi.mock('@/hermes', async importOriginal => ({
  ...(await importOriginal<Record<string, unknown>>()),
  getSession: vi.fn()
}))

vi.mock('../use-session-actions/utils', async importOriginal => ({
  ...(await importOriginal<Record<string, unknown>>()),
  resolveSessionOwner: vi.fn()
}))

const storedId = 'cron_job-1_20260929_120000'

const deferred = <T,>() => {
  let resolve!: (value: T) => void
  const promise = new Promise<T>(r => (resolve = r))

  return { promise, resolve }
}

const liveRun = () => sessionInfo({ ended_at: null, id: storedId, is_active: true, scheduler_owned: true, source: 'cron' })
const deadRun = () => sessionInfo({ ended_at: null, id: storedId, scheduler_owned: false, source: 'cron' })

// What the backend does with a /api/sessions/{id}?profile= read: a bare-profile read must name the profile the
// ambient connection serves, or it is refused (profile_scope_mismatch). An explicit connection pin carries its own.
const invalidRequests: unknown[] = []

function serveSession(row: ReturnType<typeof sessionInfo>) {
  vi.mocked(getSession).mockImplementation(async (_id, scope) => {
    if (typeof scope === 'string' && scope !== getApiRequestProfile()) {
      invalidRequests.push(scope)
    }

    return row
  })
}

describe('usePromptActions cron write gate vs scope switch during owner resolution', () => {
  const requestGateway = vi.fn(async (_method: string, _params?: Record<string, unknown>, _timeoutMs?: number) => ({}) as never)

  const render = async () => {
    let handle: HarnessHandle | null = null

    await actRender(
      <Harness
        onReady={h => (handle = h)}
        refreshSessions={async () => undefined}
        requestGateway={requestGateway as never}
        storedSessionId={storedId}
      />
    )

    return handle!
  }

  const promptSubmits = () => requestGateway.mock.calls.filter(([method]) => method === 'prompt.submit')

  beforeEach(() => {
    clearSingleFlightSessionResumeState()
    invalidRequests.length = 0
    requestGateway.mockClear()
    vi.mocked(getSession).mockReset()
    vi.mocked(resolveSessionOwner).mockReset()
    $cronRunReadOnlyVerdicts.set(new Map())
    $notifications.set([])
    setComposerDraft('keep this draft')
    setApiRequestConnection(null)
    setApiRequestProfile('alpha')
  })

  afterEach(() => {
    cleanup()
    setApiRequestProfile(null)
    setApiRequestConnection(null)
    $cronRunReadOnlyVerdicts.set(new Map())
    $notifications.set([])
    setComposerDraft('')
  })

  // One awaited act() holds the whole sequence: the submit starts, we wait for it to be parked on the pending
  // owner, switch scope, release the owner, and let the submit settle. No state update escapes act().
  const switchDuringOwnerResolution = async (
    owner: SessionOwnerScope,
    doSwitch: () => void
  ): Promise<{ result: boolean }> => {
    const pendingOwner = deferred<SessionOwnerScope>()
    vi.mocked(resolveSessionOwner).mockReturnValue(pendingOwner.promise)
    const handle = await render()
    let result = false

    await act(async () => {
      const submitted = handle.submitTextRaw('into the cron session')

      await vi.waitFor(() => expect(resolveSessionOwner).toHaveBeenCalledWith(storedId))
      doSwitch()
      pendingOwner.resolve(owner)
      result = await submitted
    })

    return { result }
  }

  it.each([
    ['profile', () => setApiRequestProfile('beta')],
    ['connection', () => setApiRequestConnection('conn-2')]
  ])('a %s switch while the owner resolves sends no read, no prompt, records no verdict and keeps the draft', async (_kind, doSwitch) => {
    serveSession(liveRun())
    const toastsBefore = $notifications.get().length

    const { result } = await switchDuringOwnerResolution('alpha', doSwitch)

    expect(result).toBe(false)
    expect(getSession).not.toHaveBeenCalled()
    expect(invalidRequests).toEqual([])
    expect(promptSubmits()).toHaveLength(0)
    expect(hasCronRunVerdict(storedId)).toBe(false)
    expect($composerDraft.get()).toBe('keep this draft')
    expect($notifications.get()).toHaveLength(toastsBefore)
  })

  it('an unchanged scope still takes the normal write-gate path', async () => {
    serveSession(liveRun())

    const { result } = await switchDuringOwnerResolution('alpha', () => undefined)

    expect(getSession).toHaveBeenCalledWith(storedId, 'alpha')
    expect(invalidRequests).toEqual([])
    expect(result).toBe(true)
    expect(promptSubmits()).toHaveLength(1)
    expect(hasCronRunVerdict(storedId)).toBe(true)
  })

  it('a genuinely read-only run stays blocked', async () => {
    serveSession(deadRun())

    const { result } = await switchDuringOwnerResolution('alpha', () => undefined)

    expect(result).toBe(false)
    expect(promptSubmits()).toHaveLength(0)
    expect($cronRunReadOnlyVerdicts.get().get(storedId)).toBe(true)
    expect($notifications.get().some(note => note.kind === 'info')).toBe(true)
  })

  it('an ordinary fetch failure with no verdict still fails closed', async () => {
    vi.mocked(getSession).mockRejectedValue(new Error('network down'))

    const { result } = await switchDuringOwnerResolution('alpha', () => undefined)

    expect(result).toBe(false)
    expect(promptSubmits()).toHaveLength(0)
    expect($cronRunReadOnlyVerdicts.get().get(storedId)).toBe(true)
  })

  it('an ordinary fetch failure keeps a known writable verdict', async () => {
    $cronRunReadOnlyVerdicts.set(new Map([[storedId, false]]))
    vi.mocked(getSession).mockRejectedValue(new Error('network down'))

    const { result } = await switchDuringOwnerResolution('alpha', () => undefined)

    expect(result).toBe(true)
    expect($cronRunReadOnlyVerdicts.get().get(storedId)).toBe(false)
  })

  it('an explicit owner pin that differs from the selected profile is not rejected for that alone', async () => {
    serveSession(liveRun())
    const owner = { connectionId: 'conn-2', profile: 'beta' }

    const { result } = await switchDuringOwnerResolution(owner, () => undefined)

    expect(getSession).toHaveBeenCalledWith(storedId, owner)
    expect(result).toBe(true)
  })

  it('a pinned owner is not superseded by an ambient switch: its read carries its own connection', async () => {
    serveSession(liveRun())
    const owner = { connectionId: 'conn-2', profile: 'beta' }

    const { result } = await switchDuringOwnerResolution(owner, () => setApiRequestProfile('gamma'))

    expect(getSession).toHaveBeenCalledWith(storedId, owner)
    expect(result).toBe(true)
  })
})

describe('refreshCronRunWriteGate superseded fetch', () => {
  beforeEach(() => $cronRunReadOnlyVerdicts.set(new Map()))
  afterEach(() => $cronRunReadOnlyVerdicts.set(new Map()))

  it('rethrows CronRunGateSupersededError and records no verdict when none existed', async () => {
    await expect(
      refreshCronRunWriteGate(storedId, async () => {
        throw new CronRunGateSupersededError()
      })
    ).rejects.toBeInstanceOf(CronRunGateSupersededError)

    expect(hasCronRunVerdict(storedId)).toBe(false)
  })

  it.each([true, false])('rethrows without changing an existing %s verdict', async verdict => {
    $cronRunReadOnlyVerdicts.set(new Map([[storedId, verdict]]))

    await expect(
      refreshCronRunWriteGate(storedId, async () => {
        throw new CronRunGateSupersededError()
      })
    ).rejects.toBeInstanceOf(CronRunGateSupersededError)

    expect($cronRunReadOnlyVerdicts.get().get(storedId)).toBe(verdict)
  })

  it('still swallows an ordinary failure: unknown run fails closed', async () => {
    await expect(
      refreshCronRunWriteGate(storedId, async () => {
        throw new Error('network down')
      })
    ).resolves.toBe(true)

    expect($cronRunReadOnlyVerdicts.get().get(storedId)).toBe(true)
  })

  it('still swallows an ordinary failure: a known verdict is kept', async () => {
    $cronRunReadOnlyVerdicts.set(new Map([[storedId, false]]))

    await expect(
      refreshCronRunWriteGate(storedId, async () => {
        throw new Error('network down')
      })
    ).resolves.toBe(false)
  })

  it('does not call fetchRow for a session that is not a cron run', async () => {
    const fetchRow = vi.fn()

    await expect(refreshCronRunWriteGate('plain-session', fetchRow)).resolves.toBe(false)
    expect(fetchRow).not.toHaveBeenCalled()
  })
})
