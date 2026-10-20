// Every Desktop RPC that carries a session_id must reach the backend that OWNS
// that session. On a multiplexed local gateway the window's ambient socket is
// the launch profile's authority, which refuses a secondary profile's session:
// llm.oneshot and the session YOLO toggle failed outright, session.title fell
// back to REST and model.options to the launch profile's catalog.
import type { ModelOptionsResult } from '@hermes/shared'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { cleanup, render, waitFor } from '@testing-library/react'
import { atom } from 'nanostores'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { I18nProvider } from '@/i18n'
import type * as GatewayStore from '@/store/gateway'
import { _resetSessionOwnerHintsForTests, setSessionOwnerHint } from '@/store/session'
import { $sessionStates } from '@/store/session-states'
import { stubResizeObserver } from '@/test/jsdom'

const mocks = vi.hoisted(() => {
  // The ambient (launch-profile) socket: it refuses another profile's session.
  const ambient = {
    request: vi.fn(async () => {
      throw new Error('not_found')
    })
  }

  return { ambient, requestGatewayForAgent: vi.fn(), requestModelOptions: vi.fn() }
})

vi.mock('@/store/gateway', async importOriginal => ({
  ...(await importOriginal<typeof GatewayStore>()),
  $gateway: atom(mocks.ambient),
  activeGateway: () => mocks.ambient,
  requestGatewayForAgent: mocks.requestGatewayForAgent
}))

vi.mock('@/lib/model-options', async importOriginal => ({
  ...(await importOriginal<Record<string, unknown>>()),
  requestModelOptions: mocks.requestModelOptions
}))

const { requestOneShot } = await import('./oneshot')
const { setSessionYolo } = await import('./yolo-session')
const { renameSessionPreferringRpc } = await import('@/app/chat/sidebar/session-actions-menu')
const { ModelVisibilityDialog } = await import('@/components/model-visibility-dialog')

stubResizeObserver()

const RUNTIME = 'rt-research-1'
const STORED = 'stored-research-1'
const OWNER = { connectionId: 'local', profile: 'research' }

beforeEach(() => {
  mocks.requestGatewayForAgent.mockImplementation(async (_connection: string, _profile: string, method: string) =>
    method === 'llm.oneshot'
      ? { text: ' commit: research ' }
      : method === 'config.set'
        ? { value: '1' }
        : method === 'session.title'
          ? { title: 'Renamed on owner' }
          : {}
  )
  mocks.requestModelOptions.mockResolvedValue({ providers: [] } satisfies ModelOptionsResult)
  $sessionStates.set({ [RUNTIME]: { storedSessionId: STORED } as never })
  setSessionOwnerHint(STORED, OWNER)
})

afterEach(() => {
  cleanup()
  _resetSessionOwnerHintsForTests()
  $sessionStates.set({})
  vi.clearAllMocks()
})

describe('session_id RPCs go to the session owner, not the ambient socket', () => {
  it('llm.oneshot borrows the secondary-profile session model on its owner', async () => {
    await expect(requestOneShot({ sessionId: RUNTIME, template: 'commit_message' })).resolves.toBe('commit: research')

    expect(mocks.ambient.request).not.toHaveBeenCalled()
    expect(mocks.requestGatewayForAgent).toHaveBeenCalledWith(
      'local',
      'research',
      'llm.oneshot',
      expect.objectContaining({ session_id: RUNTIME, template: 'commit_message' })
    )
  })

  it('the session YOLO toggle lands on the owner, even when handed the ambient requester', async () => {
    const ambientRequest = vi.fn(async () => {
      throw new Error('not_found')
    })

    await expect(setSessionYolo(ambientRequest as never, RUNTIME, true)).resolves.toBe(true)

    expect(ambientRequest).not.toHaveBeenCalled()
    expect(mocks.requestGatewayForAgent).toHaveBeenCalledWith('local', 'research', 'config.set', {
      key: 'yolo',
      session_id: RUNTIME,
      value: '1'
    })
  })

  it('session.title renames the live runtime on its owner instead of dropping to REST', async () => {
    await expect(renameSessionPreferringRpc(STORED, 'Renamed')).resolves.toEqual({ title: 'Renamed on owner' })

    expect(mocks.ambient.request).not.toHaveBeenCalled()
    expect(mocks.requestGatewayForAgent).toHaveBeenCalledWith('local', 'research', 'session.title', {
      session_id: RUNTIME,
      title: 'Renamed'
    })
  })

  it('Edit Models reads model.options for a secondary-profile session from its owner', async () => {
    render(
      <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
        <I18nProvider>
          <ModelVisibilityDialog
            gw={mocks.ambient as never}
            onOpenChange={() => {}}
            onOpenProviders={() => {}}
            open
            profile="default"
            sessionId={RUNTIME}
          />
        </I18nProvider>
      </QueryClientProvider>
    )

    await waitFor(() => expect(mocks.requestModelOptions).toHaveBeenCalled())

    const { request } = mocks.requestModelOptions.mock.calls[0][0] as {
      request?: (method: string, params?: Record<string, unknown>) => Promise<unknown>
    }

    expect(request).toBeTypeOf('function')
    await request!('model.options', { session_id: RUNTIME })
    expect(mocks.ambient.request).not.toHaveBeenCalled()
    expect(mocks.requestGatewayForAgent).toHaveBeenCalledWith('local', 'research', 'model.options', {
      session_id: RUNTIME
    })
  })
})
