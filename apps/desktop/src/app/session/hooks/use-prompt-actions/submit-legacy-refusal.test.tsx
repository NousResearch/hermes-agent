import { cleanup } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { en } from '@/i18n/en'
import { $notifications, clearNotifications } from '@/store/notifications'

import { actRender, Harness, type HarnessHandle, RUNTIME_SESSION_ID } from './index-test-utils'
import { clearStoredJournal, storedJournal } from './prepared-submissions-test-utils'
import { clearSingleFlightSessionResumeState } from './single-flight-resume'

vi.mock('@/hermes', () => ({
  getLatestSessionMessages: vi.fn(async () => ({ messages: [], session_id: 'session' })),
  getProfiles: vi.fn(async () => ({ profiles: [] })),
  getSession: vi.fn(),
  PROMPT_SUBMIT_REQUEST_TIMEOUT_MS: 1_800_000,
  setApiRequestProfile: vi.fn(),
  transcribeAudio: vi.fn()
}))

vi.mock('@/store/gateway', async importOriginal => ({
  ...(await importOriginal<Record<string, unknown>>()),
  requestGatewayForAgent: vi.fn()
}))

const FRESH_RUNTIME_ID = `${RUNTIME_SESSION_ID}-fresh`

// The legacy `prompt.submit` handler's answer for a runtime id it no longer holds. It looks the
// session up before it claims a slot or starts a turn, so nothing was admitted.
const SESSION_NOT_FOUND = () => {
  throw Object.assign(new Error('session not found'), { code: 4001 })
}

const lostAck = () => {
  throw new Error('request timed out: prompt.submit')
}

const admitted = () => ({ status: 'streaming' })

interface LegacyServe {
  /** One answer per identityless send, in order. */
  identityless: Array<() => unknown>
  resume?: () => unknown
}

// A legacy `hermes serve`: its strict contract refuses `submission_id` (4000) before any handler
// runs, so every send is followed by an identityless one.
function legacyServe({ identityless, resume = () => ({ session_id: FRESH_RUNTIME_ID }) }: LegacyServe) {
  const calls: string[] = []
  const sessionIds: unknown[] = []

  const requestGateway = vi.fn(async (method: string, params?: Record<string, unknown>) => {
    if (method === 'session.resume') {
      calls.push('resume')

      return resume() as never
    }

    if (method !== 'prompt.submit') {
      return {} as never
    }

    if (params && 'submission_id' in params) {
      calls.push('submit(id)')

      throw Object.assign(
        new Error('invalid params for prompt.submit: submission_id: Extra inputs are not permitted'),
        { code: 4000 }
      )
    }

    calls.push('submit')
    sessionIds.push(params?.session_id)

    return identityless.shift()!() as never
  })

  return { calls, requestGateway, sessionIds }
}

async function mount(requestGateway: ReturnType<typeof legacyServe>['requestGateway']): Promise<HarnessHandle> {
  let handle: HarnessHandle | null = null
  await actRender(
    <Harness
      onReady={h => (handle = h)}
      rawAdmissionReceipts
      refreshSessions={async () => undefined}
      requestGateway={requestGateway}
    />
  )

  return handle!
}

const shownNotices = () => $notifications.get().map(notice => notice.message)

// #154787: the identityless send to a legacy backend has no deduplication identity, so recovery
// must not repeat it after a lost acknowledgement. `session not found` is not a lost
// acknowledgement: the backend answered before admitting anything, so the resume may send again.
describe('identityless legacy send refused with session not found', () => {
  beforeEach(() => {
    clearSingleFlightSessionResumeState()
    clearStoredJournal()
    clearNotifications()
  })

  afterEach(cleanup)

  it('resumes a dropped runtime session and sends again', async () => {
    const serve = legacyServe({ identityless: [SESSION_NOT_FOUND, admitted] })
    const handle = await mount(serve.requestGateway)

    expect(await handle.submitText('follow-up')).toBe(true)
    expect(serve.calls).toEqual(['submit(id)', 'submit', 'resume', 'submit(id)', 'submit'])
    expect(serve.sessionIds).toEqual([RUNTIME_SESSION_ID, FRESH_RUNTIME_ID])
    expect(handle.state().messages.some(message => message.error)).toBe(false)
    expect(storedJournal()).toEqual({})
  })

  it('still fails closed when the send after the refusal loses its acknowledgement', async () => {
    // A third identityless send would be admitted: it must never go out.
    const serve = legacyServe({ identityless: [SESSION_NOT_FOUND, lostAck, admitted] })
    const handle = await mount(serve.requestGateway)
    const options = { submission_id: 'lost-after-refusal' }

    expect(await handle.submitText('follow-up', options)).toBe(false)
    expect(serve.calls.filter(call => call === 'submit')).toHaveLength(2)
    expect(Object.values(storedJournal()).map(entry => entry.legacyAttempted)).toEqual([true])

    // The explicit retry of that uncertain send is refused, visibly, without sending.
    expect(await handle.submitText('follow-up', options)).toBe(false)
    expect(serve.calls.filter(call => call === 'submit')).toHaveLength(2)
    expect(shownNotices()).toContain(en.promptNotices.legacySendUnconfirmed)
  })

  it('leaves a refused send retryable when the resume fails', async () => {
    const serve = legacyServe({
      identityless: [SESSION_NOT_FOUND, admitted],
      resume: () => {
        throw new Error('gateway not connected')
      }
    })

    const handle = await mount(serve.requestGateway)
    const options = { submission_id: 'refused-then-resent' }

    expect(await handle.submitText('follow-up', options)).toBe(false)
    expect(serve.calls).toEqual(['submit(id)', 'submit', 'resume'])
    // The backend refused it: the retained entry is not an uncertain send.
    expect(Object.values(storedJournal()).map(entry => Boolean(entry.legacyAttempted))).toEqual([false])

    // The next Enter on the restored draft sends, with no "may already have run" notice.
    expect(await handle.submitText('follow-up', options)).toBe(true)
    expect(serve.calls).toEqual(['submit(id)', 'submit', 'resume', 'submit(id)', 'submit'])
    expect(shownNotices()).not.toContain(en.promptNotices.legacySendUnconfirmed)
    expect(storedJournal()).toEqual({})
  })
})
