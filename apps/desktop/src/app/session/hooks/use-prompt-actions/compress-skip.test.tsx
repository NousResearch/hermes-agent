import { cleanup } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { $notifications, clearNotifications } from '@/store/notifications'
import { setCurrentUsage, setMessages, setSessions } from '@/store/session'

import { actRender, Harness, type HarnessHandle, sessionInfo } from './index-test-utils'
import { clearSingleFlightSessionResumeState } from './single-flight-resume'

// Split out of index.test.tsx on purpose: that file is already over the
// code-health line cap, and new behaviour never grows an over-cap file.
beforeEach(() => {
  clearSingleFlightSessionResumeState()
  setSessions(() => [sessionInfo()])
})

afterEach(() => {
  cleanup()
  clearNotifications()
  setCurrentUsage({ calls: 0, input: 0, output: 0, total: 0 })
  setMessages([])
  vi.restoreAllMocks()
})

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

describe('usePromptActions /compress outcome contract', () => {
  it('reports a lock-held compression as the backend reason, never as success', async () => {
    const requestGateway = vi.fn(async (method: string) => {
      if (method === 'session.compress') {
        // The in-process route's lock skip: answered, nothing compressed.
        return {
          compressed: false,
          lock_held: true,
          message: 'another process holds the compression lock for this session'
        } as never
      }

      return {} as never
    })

    let handle: HarnessHandle | null = null
    await actRender(
      <Harness onReady={h => (handle = h)} refreshSessions={async () => undefined} requestGateway={requestGateway} />
    )

    await handle!.submitText('/compress')

    expect($notifications.get()).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          kind: 'info',
          message: 'another process holds the compression lock for this session'
        })
      ])
    )
    expect($notifications.get()).not.toEqual(expect.arrayContaining([expect.objectContaining({ kind: 'success' })]))
  })
})
