import { cleanup } from '@testing-library/react'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import { $heldPrompt, takeHeldPrompt } from '@/store/held-prompt'
import { $shellChoice } from '@/store/shell-variant'
import { $visitorAccess } from '@/store/visitor-access'

import { actRender, Harness, type HarnessHandle } from './index-test-utils'

vi.mock('@/hermes', () => ({
  getLatestSessionMessages: vi.fn(async () => ({ messages: [], session_id: 'session' })),
  getProfiles: vi.fn(async () => ({ profiles: [] })),
  getSession: vi.fn(),
  PROMPT_SUBMIT_REQUEST_TIMEOUT_MS: 1_800_000,
  setApiRequestProfile: vi.fn(),
  transcribeAudio: vi.fn()
}))

beforeEach(() => {
  window.localStorage.clear()
  takeHeldPrompt()
  $shellChoice.set(true)
  $visitorAccess.set({ kind: 'open' })
})

afterEach(() => {
  cleanup()
  $visitorAccess.set({ kind: 'open' })
})

async function send(text: string) {
  let handle: HarnessHandle | null = null
  const requestGateway = vi.fn(async () => ({ status: 'queued' }) as never)
  const createBackendSessionForSend = vi.fn(async () => 'rt-new')

  await actRender(
    <Harness
      activeSessionId={null}
      createBackendSessionForSend={createBackendSessionForSend}
      onReady={h => (handle = h)}
      refreshSessions={async () => undefined}
      requestGateway={requestGateway}
    />
  )

  const accepted = await handle!.submitText(text)

  return { accepted, createBackendSessionForSend, requestGateway }
}

it('a visitor who must sign up sends nothing anywhere: the message is held, and the composer may clear', async () => {
  $visitorAccess.set({ kind: 'sign_up_required', signUpUrl: 'https://portal.example/sign-up' })

  const { accepted, createBackendSessionForSend, requestGateway } = await send('  Clean up my inbox  ')

  expect(accepted).toBe(true)
  expect(requestGateway).not.toHaveBeenCalled()
  expect(createBackendSessionForSend).not.toHaveBeenCalled()
  expect($heldPrompt.get()).toBe('Clean up my inbox')
})

it('a slash command from such a visitor is held too, never run', async () => {
  $visitorAccess.set({ kind: 'sign_up_required', signUpUrl: 'https://portal.example/sign-up' })

  const { requestGateway } = await send('/help')

  expect(requestGateway).not.toHaveBeenCalled()
})

it('anyone else’s message goes to their agent as always, and nothing is held', async () => {
  const { createBackendSessionForSend, requestGateway } = await send('Clean up my inbox')

  expect(createBackendSessionForSend.mock.calls.length + requestGateway.mock.calls.length).toBeGreaterThan(0)
  expect($heldPrompt.get()).toBeNull()
})
