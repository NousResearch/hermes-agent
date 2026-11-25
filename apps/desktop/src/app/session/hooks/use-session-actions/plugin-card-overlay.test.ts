import { describe, expect, it, vi } from 'vitest'

import { handlePluginCardEvent } from '@/app/session/hooks/use-message-stream/gateway-event/plugin-card'
import type { GatewayEventContext } from '@/app/session/hooks/use-message-stream/gateway-event/types'
import { type ChatMessage, toChatMessages } from '@/lib/chat-messages'
import type { SessionMessage } from '@/types/hermes'

import { overlayConcurrentMessageChanges } from './utils'

const card = { card_id: 'card-1', plugin_id: 'example' }

const stored = (descriptor = card): ChatMessage[] =>
  toChatMessages([
    {
      id: 123,
      content: '[example posted a card 0123456789abcdef]',
      display_kind: 'plugin_card',
      display_metadata: { display_text: 'A card', plugin_card: descriptor },
      role: 'user',
      timestamp: 1_700_000_000
    } as unknown as SessionMessage
  ])

function eventDuringFetch() {
  let state = { messages: [] as ChatMessage[] }
  const baseline = state.messages
  handlePluginCardEvent({
    deps: {
      flushQueuedDeltas: vi.fn(),
      updateSessionState: (_id: string, update: (previous: typeof state) => typeof state) => {
        state = update(state)
      }
    },
    event: { type: 'plugin.card' },
    occurredAt: 1_700_000_000,
    payload: { ...card, text: 'A card' },
    sessionId: 'session-1'
  } as unknown as GatewayEventContext)

  return { baseline, current: state.messages }
}

describe('plugin cards during an in-flight history fetch', () => {
  it('keeps the durable row once when the live event arrives before the history response', () => {
    const { baseline, current } = eventDuringFetch()
    const page = stored()
    expect(current).toHaveLength(1)
    expect(current[0].id).not.toBe(page[0].id)

    const merged = overlayConcurrentMessageChanges(page, baseline, current)

    expect(merged).toBe(page)
    expect(merged.map(message => message.id)).toEqual([page[0].id])
  })

  it.each([
    { card_id: 'card-1', plugin_id: 'another-plugin' },
    { card_id: 'card-2', plugin_id: 'example' }
  ])('preserves distinct card identity %j', descriptor => {
    const { baseline, current } = eventDuringFetch()
    const merged = overlayConcurrentMessageChanges(stored(descriptor), baseline, current)

    expect(merged).toHaveLength(2)
    expect(merged.map(message => message.pluginCard)).toEqual([
      { cardId: descriptor.card_id, pluginId: descriptor.plugin_id },
      { cardId: card.card_id, pluginId: card.plugin_id }
    ])
  })

  it('keeps an event not yet present in the fetched page', () => {
    const { baseline, current } = eventDuringFetch()
    expect(overlayConcurrentMessageChanges([], baseline, current)).toEqual(current)
  })
})
