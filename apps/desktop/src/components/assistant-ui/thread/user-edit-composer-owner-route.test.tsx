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
import { $sessions, _resetSessionOwnerHintsForTests, setSessionOwnerHint } from '@/store/session'

import { assistantMessage, stubThreadEnvironment, stubThreadViewportSize, userMessage } from '../test-utils'

import { Thread } from '.'

const routed = vi.hoisted(() => ({ requestGatewayForAgent: vi.fn(), requestGatewayForProfile: vi.fn() }))

vi.mock('@/store/gateway', async importOriginal => ({
  ...(await importOriginal<typeof GatewayStore>()),
  requestGatewayForAgent: routed.requestGatewayForAgent,
  requestGatewayForProfile: routed.requestGatewayForProfile
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
  $sessions.set([])
  routed.requestGatewayForAgent.mockReset()
  routed.requestGatewayForProfile.mockReset()
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

describe('message-edit composer attachment drops', () => {
  // A session row that names only its profile is a bare-string owner. Its
  // upload still belongs on that profile's socket, never the ambient one.
  it('stage a dropped image on a bare-profile owner, not the ambient socket', async () => {
    _resetSessionOwnerHintsForTests()
    $sessions.set([{ id: SESSION, profile: 'research' }] as never)
    routed.requestGatewayForProfile.mockImplementation(async (_profile: string, method: string) =>
      method === 'image.attach' ? { attached: true, path: '/research/cache/shot.png' } : CATALOG
    )
    const previous = window.hermesDesktop
    Object.defineProperty(window, 'hermesDesktop', {
      configurable: true,
      value: { ...previous, getPathForFile: () => '/Users/me/shot.png' }
    })

    const gateway = { request: vi.fn(async () => CATALOG) } as unknown as HermesGateway & {
      request: ReturnType<typeof vi.fn>
    }

    try {
      render(<Harness gateway={gateway} />)
      fireEvent.click(await screen.findByRole('button', { name: 'Edit message' }))
      const editor = await screen.findByRole('textbox', { name: 'Edit message' })
      const file = new File(['png'], 'shot.png', { type: 'image/png' })

      fireEvent.drop(editor, {
        dataTransfer: {
          files: Object.assign([file], { item: () => file }),
          getData: () => '',
          items: [{ getAsFile: () => file, kind: 'file', webkitGetAsEntry: () => ({ isDirectory: false }) }],
          types: ['Files']
        }
      })

      await waitFor(() =>
        expect(routed.requestGatewayForProfile).toHaveBeenCalledWith(
          'research',
          'image.attach',
          { path: '/Users/me/shot.png', session_id: SESSION },
          undefined,
          undefined
        )
      )
      expect(gateway.request).not.toHaveBeenCalledWith('image.attach', expect.anything())
    } finally {
      Object.defineProperty(window, 'hermesDesktop', { configurable: true, value: previous })
    }
  })
})
