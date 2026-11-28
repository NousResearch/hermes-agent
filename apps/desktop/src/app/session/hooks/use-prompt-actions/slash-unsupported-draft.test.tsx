import { JsonRpcGatewayError } from '@hermes/shared'
import { act, cleanup } from '@testing-library/react'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import { setSessions } from '@/store/session'
import { dropSessionState } from '@/store/session-states'

import type { CreateBackendSessionForSend } from '../use-session-actions/create-overrides'

import { actRender, Harness, type HarnessHandle, RUNTIME_SESSION_ID, sessionInfo } from './index-test-utils'

vi.mock('@/hermes', () => ({
  getLatestSessionMessages: vi.fn(async () => ({ messages: [], session_id: 'session' })),
  getProfiles: vi.fn(async () => ({ profiles: [] })),
  getSession: vi.fn(),
  PROMPT_SUBMIT_REQUEST_TIMEOUT_MS: 1_800_000,
  setApiRequestProfile: vi.fn(),
  transcribeAudio: vi.fn()
}))

beforeEach(() => {
  setSessions(() => [sessionInfo()])
})

afterEach(() => {
  cleanup()
  dropSessionState(RUNTIME_SESSION_ID)
  setSessions(() => [])
})

const unsupported = new JsonRpcGatewayError('unsupported_command', {
  code: 4001,
  data: { reason: 'unsupported_command' }
})

it.each([
  { case: 'returns the draft to the newly created chat after both routes refuse the command', execError: unsupported, submitted: false },
  { case: 'avoids restoring a retry after slash.exec times out', execError: new Error('slash worker timed out'), submitted: true }
])('$case', async ({ execError, submitted }) => {
  const requestGateway = vi.fn(async (method: string) => {
    if (method === 'slash.exec') {
      throw execError
    }

    if (method === 'command.dispatch') {
      throw unsupported
    }

    return {} as never
  })

  const onComposerScopeAssigned = vi.fn()

  const createBackendSessionForSend = vi.fn<CreateBackendSessionForSend>(async (_preview, overrides) => {
    overrides?.onComposerScopeAssigned?.('stored-unknown-command')

    return RUNTIME_SESSION_ID
  })

  let handle: HarnessHandle | null = null
  await actRender(
    <Harness
      activeSessionId={null}
      createBackendSessionForSend={createBackendSessionForSend}
      onReady={value => { handle = value }}
      refreshSessions={async () => undefined}
      requestGateway={requestGateway}
      storedSessionId={null}
    />
  )

  const arg = '这段草稿需要保留。'.repeat(160)
  await act(async () => {
    expect(await handle!.submitTextRaw(`/grill ${arg}`, { onComposerScopeAssigned })).toBe(submitted)
  })
  expect(onComposerScopeAssigned).toHaveBeenCalledWith('stored-unknown-command')
  expect(requestGateway).toHaveBeenCalledWith('slash.exec', expect.objectContaining({ command: `grill ${arg}` }))
  expect(requestGateway).toHaveBeenCalledWith('command.dispatch', expect.objectContaining({ name: 'grill', arg }))
  expect(requestGateway.mock.calls.some(([method]) => method === 'prompt.submit')).toBe(false)
})
