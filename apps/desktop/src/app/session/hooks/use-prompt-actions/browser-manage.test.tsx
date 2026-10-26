import { cleanup } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { $connection, setSessions } from '@/store/session'

import { actRender, Harness, type HarnessHandle, RUNTIME_SESSION_ID, sessionInfo } from './index-test-utils'
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

describe('usePromptActions /browser connect|disconnect', () => {
  beforeEach(() => {
    clearSingleFlightSessionResumeState()
    setSessions(() => [sessionInfo()])
  })

  afterEach(() => {
    cleanup()
    $connection.set(null)
    vi.restoreAllMocks()
  })

  // #134889 TUI-14: on a local connection /browser connect|disconnect go to the gateway as
  // browser.manage (never slash.exec); the gateway owns any refusal.
  it.each([
    ['/browser connect', { action: 'connect', session_id: RUNTIME_SESSION_ID, url: 'http://127.0.0.1:9222' }],
    ['/browser disconnect', { action: 'disconnect', session_id: RUNTIME_SESSION_ID }]
  ])('%s on a local connection is sent as browser.manage', async (text, params) => {
    $connection.set({ connectionId: 'local', mode: 'local' } as never)
    const requestGateway = vi.fn(async () => ({ connected: false }) as never)

    let handle: HarnessHandle | null = null
    await actRender(
      <Harness
        onReady={h => (handle = h)}
        refreshSessions={vi.fn(async () => undefined)}
        requestGateway={requestGateway}
      />
    )

    await handle!.submitText(text)

    expect(requestGateway).toHaveBeenCalledWith('browser.manage', expect.objectContaining(params))
    expect(requestGateway).not.toHaveBeenCalledWith('slash.exec', expect.anything())
  })
})
