// @vitest-environment jsdom
// A plugin's card (`ctx.post_card`) is a durable `plugin_card` row: it renders the
// transcript directive the plugin's Desktop half registered, with the row's attrs;
// a name nobody claims is the card's text; the live `plugin.card` event appends the
// same row without a reload.
import { AssistantRuntimeProvider, type ThreadMessage, useExternalStoreRuntime } from '@assistant-ui/react'
import { cleanup, render } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { handlePluginCardEvent } from '@/app/session/hooks/use-message-stream/gateway-event/plugin-card'
import type { GatewayEventContext } from '@/app/session/hooks/use-message-stream/gateway-event/types'
import { registry } from '@/contrib/registry'
import { type ChatMessage, toChatMessages } from '@/lib/chat-messages'
import { toRuntimeMessage } from '@/lib/chat-runtime'
import { TRANSCRIPT_DIRECTIVE_AREA, type TranscriptDirectiveContribution } from '@/lib/transcript-directives'
import type { SessionMessage } from '@/types/hermes'

import { stubThreadEnvironment } from '../test-utils'

import { Thread } from '.'

vi.mock('@/store/native-notifications', () => ({ dispatchNativeNotification: vi.fn() }))

stubThreadEnvironment()
afterEach(cleanup)

const CARD = {
  card_id: 'share-1',
  directive: { attrs: { mode: 'ask', skill: 'deploy-checklist' }, name: 'wisdom-share' },
  plugin_id: 'cwplug'
}

const TEXT = 'Share "deploy-checklist" with your team?'

function storedRow(card: unknown = CARD): SessionMessage {
  return {
    content: '[cwplug posted a card share-1]',
    display_kind: 'plugin_card',
    display_metadata: { completion_note: ['x'], display_text: TEXT, plugin_card: card },
    role: 'user',
    timestamp: 1_700_000_000
  } as unknown as SessionMessage
}

function Harness({ messages }: { messages: ChatMessage[] }) {
  const runtime = useExternalStoreRuntime<ThreadMessage>({
    isRunning: false,
    messages: messages.map(toRuntimeMessage),
    onNew: async () => {}
  })

  return (
    <AssistantRuntimeProvider runtime={runtime}>
      <Thread />
    </AssistantRuntimeProvider>
  )
}

function claimShare() {
  return registry.register({
    area: TRANSCRIPT_DIRECTIVE_AREA,
    data: {
      name: 'wisdom-share',
      render: ({ attrs }) => (
        <div data-testid="share-card">
          {attrs.skill}:{attrs.mode}
        </div>
      )
    } satisfies TranscriptDirectiveContribution,
    id: 'test:wisdom-share',
    source: 'plugin:cwplug'
  })
}

describe('plugin_card rows', () => {
  it('hydrate as a system row carrying the card, never as the user', () => {
    const [message] = toChatMessages([storedRow()])

    expect(message.role).toBe('system')
    expect(message.pluginCard).toEqual({
      cardId: 'share-1',
      directive: { attrs: { mode: 'ask', skill: 'deploy-checklist' }, name: 'wisdom-share' },
      pluginId: 'cwplug'
    })
    expect(message.parts).toMatchObject([{ text: TEXT, type: 'text' }])
  })

  it("render the plugin's registered directive with the row's attrs", () => {
    const dispose = claimShare()

    try {
      const { getByTestId } = render(<Harness messages={toChatMessages([storedRow()])} />)

      expect(getByTestId('share-card').textContent).toBe('deploy-checklist:ask')
    } finally {
      dispose()
    }
  })

  it('fall back to the text when no plugin claims the directive', () => {
    const { container, queryByTestId } = render(<Harness messages={toChatMessages([storedRow()])} />)

    expect(queryByTestId('share-card')).toBeNull()
    expect(container.querySelector('[data-role="system"]')?.textContent).toContain(TEXT)
  })

  it('show only the text for a card with no directive', () => {
    const dispose = claimShare()

    try {
      const rows = toChatMessages([storedRow({ card_id: 'note-1', plugin_id: 'cwplug' })])
      const { container, queryByTestId } = render(<Harness messages={rows} />)

      expect(queryByTestId('share-card')).toBeNull()
      expect(container.querySelector('[data-role="system"]')?.textContent).toContain(TEXT)
    } finally {
      dispose()
    }
  })
})

describe('the plugin.card event', () => {
  function liveContext(payload: Record<string, unknown>, initialMessages: ChatMessage[] = []) {
    let messages: ChatMessage[] = initialMessages

    const updateSessionState = vi.fn((_sid: string, update: (state: { messages: ChatMessage[] }) => unknown) => {
      messages = (update({ messages }) as { messages: ChatMessage[] }).messages
    })

    const ctx = {
      deps: {
        compactedTurnRef: { current: new Set<string>() },
        failAssistantMessage: vi.fn(),
        flushQueuedDeltas: vi.fn(),
        hydrateFromStoredSession: vi.fn(),
        queryClient: { invalidateQueries: vi.fn() },
        sessionStateByRuntimeIdRef: { current: new Map() },
        updateSessionState
      },
      event: { payload, session_id: 'sess-1', type: 'plugin.card' },
      explicitSid: 'sess-1',
      fromActiveSource: () => true,
      isActiveEvent: true,
      occurredAt: 1_700_000_100,
      payload,
      scheduleConfigRefresh: vi.fn(),
      sessionId: 'sess-1'
    } as unknown as GatewayEventContext

    return { ctx, messages: () => messages }
  }

  it('appends the card row without a reload, once per card', () => {
    const { ctx, messages } = liveContext({ ...CARD, text: TEXT })

    expect(handlePluginCardEvent(ctx)).toBe(true)
    handlePluginCardEvent(ctx)

    expect(messages()).toHaveLength(1)
    expect(messages()[0]).toMatchObject({
      pluginCard: { cardId: 'share-1', directive: CARD.directive, pluginId: 'cwplug' },
      role: 'system'
    })

    const dispose = claimShare()

    try {
      const { getByTestId } = render(<Harness messages={messages()} />)

      expect(getByTestId('share-card').textContent).toBe('deploy-checklist:ask')
    } finally {
      dispose()
    }
  })

  it('does not append a replayed event after the durable row hydrated', () => {
    const hydrated = toChatMessages([storedRow()])
    const { ctx, messages } = liveContext({ ...CARD, text: TEXT }, hydrated)

    expect(handlePluginCardEvent(ctx)).toBe(true)
    expect(messages()).toEqual(hydrated)
  })
})
