import { cleanup } from '@testing-library/react'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import { actRender, Harness, type HarnessHandle } from './index-test-utils'
import { clearStoredJournal } from './prepared-submissions-test-utils'
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

beforeEach(() => {
  clearSingleFlightSessionResumeState()
  clearStoredJournal()
  window.localStorage.removeItem('hermes.desktop.composerQueue.v1')
})

afterEach(cleanup)

// A message held in the queue while its profile's backend restarts (use-composer-submit) drains the
// moment the backend is back, still naming the chat's runtime id from before the restart: the new
// backend process does not know it. A backend on the `hermes serve` contract also refuses
// `submission_id` (4000), so the send goes: with the id -> 4000, without it -> "session not found"
// (4001). Both are refused before admission, so nothing ran: the resumed session takes the message.
it('a queued send to a restarted legacy backend recovers the session and is sent once', async () => {
  const STORED_SESSION_ID = 'stored-chat'
  const RESTARTED_RUNTIME_ID = 'rt-after-restart'
  const submits: Array<Record<string, unknown>> = []

  const requestGateway = vi.fn(async (method: string, params?: Record<string, unknown>) => {
    if (method === 'session.resume') {
      return { session_id: RESTARTED_RUNTIME_ID } as never
    }

    if (method !== 'prompt.submit') {
      return {} as never
    }

    submits.push(params!)

    if ('submission_id' in params!) {
      throw Object.assign(
        new Error(
          'invalid params for prompt.submit: submission_id: Extra inputs are not permitted — the client and the Hermes backend are out of sync (different versions)'
        ),
        { code: 4000 }
      )
    }

    if (params!.session_id !== RESTARTED_RUNTIME_ID) {
      throw Object.assign(new Error('session not found'), { code: 4001 })
    }

    return { status: 'streaming' } as never
  })

  let handle: HarnessHandle | null = null
  await actRender(
    <Harness
      onReady={h => (handle = h)}
      rawAdmissionReceipts
      refreshSessions={async () => undefined}
      requestGateway={requestGateway}
      storedSessionId={STORED_SESSION_ID}
    />
  )

  expect(await handle!.submitText('try again', { fromQueue: true, submission_id: 'queued-1' })).toBe(true)

  const admitted = submits.filter(params => params.session_id === RESTARTED_RUNTIME_ID && !('submission_id' in params))
  expect(admitted).toHaveLength(1)
  expect(admitted[0]).toMatchObject({ text: 'try again' })
})
