// The inline message-edit composer is a second `/` palette: it must reach the
// edited session's OWNER exactly like the main composer does (#141583). Before
// it passed its sessionId, its catalog went out on the ambient socket with no
// session at all, so a non-primary-profile chat's edit palette listed the
// launch profile's commands (or nothing).
import { ExportedMessageRepository, type ThreadMessage } from '@assistant-ui/react'
import { AssistantRuntimeProvider } from '@assistant-ui/react'
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { HermesGateway } from '@/hermes'
import { useIncrementalExternalStoreRuntime } from '@/lib/incremental-external-store-runtime'
import { queryClient } from '@/lib/query-client'
import type * as GatewayStore from '@/store/gateway'
import { _resetSessionOwnerHintsForTests, setSessionOwnerHint } from '@/store/session'

import { assistantMessage, stubThreadEnvironment, stubThreadViewportSize, userMessage } from '../test-utils'

import { Thread } from '.'

const routed = vi.hoisted(() => ({ requestGatewayForAgent: vi.fn() }))

vi.mock('@/store/gateway', async importOriginal => ({
  ...(await importOriginal<typeof GatewayStore>()),
  requestGatewayForAgent: routed.requestGatewayForAgent
}))

stubThreadEnvironment()
stubThreadViewportSize()

const SESSION = 'sess-research-edit'
const OWNER = { connectionId: 'local', profile: 'research' }

const CATALOG = {
  categories: [{ name: 'Session', pairs: [['/new', 'Start a new session']] }],
  pairs: [['/new', 'Start a new session']]
}

function Harness({ gateway }: { gateway: HermesGateway }) {
  const runtime = useIncrementalExternalStoreRuntime<ThreadMessage>({
    messageRepository: ExportedMessageRepository.fromArray([userMessage(), assistantMessage()]),
    isRunning: false,
    setMessages: () => {},
    onNew: async () => {},
    onEdit: async () => {},
    onCancel: async () => {},
    onReload: async () => {}
  })

  return (
    <AssistantRuntimeProvider runtime={runtime}>
      <Thread gateway={gateway} sessionId={SESSION} />
    </AssistantRuntimeProvider>
  )
}

beforeEach(() => {
  routed.requestGatewayForAgent.mockResolvedValue(CATALOG)
  setSessionOwnerHint(SESSION, OWNER)
})

afterEach(() => {
  cleanup()
  queryClient.clear()
  _resetSessionOwnerHintsForTests()
  routed.requestGatewayForAgent.mockReset()
})

describe('message-edit composer completions', () => {
  it('fetch the `/` catalog from the edited session’s owner, not the ambient socket', async () => {
    const gateway = { request: vi.fn(async () => CATALOG) } as unknown as HermesGateway & {
      request: ReturnType<typeof vi.fn>
    }

    render(<Harness gateway={gateway} />)
    fireEvent.click(await screen.findByRole('button', { name: 'Edit message' }))
    await screen.findByRole('textbox', { name: 'Edit message' })

    await waitFor(() =>
      expect(routed.requestGatewayForAgent).toHaveBeenCalledWith('local', 'research', 'commands.catalog', {
        session_id: SESSION
      })
    )
    expect(gateway.request).not.toHaveBeenCalledWith('commands.catalog', expect.anything())
  })
})
