import { describe, expect, it } from 'vitest'

import type { ChatMessage, ChatMessagePart } from './chat-messages'
import { preserveMcpUiCards } from './chat-messages'

// MCP Apps ui payloads are live-only (single-use pop; never persisted), so a
// hydrate-from-storage rebuild must re-attach them by toolCallId or the card
// iframe vanishes from the transcript (D1).
describe('preserveMcpUiCards', () => {
  const ui = { server: 'utp', uri: 'ui://utp/catalog-search', html: '<html>', csp: null }

  const liveMessages: ChatMessage[] = [
    {
      id: 'assistant-stream-1',
      role: 'assistant',
      parts: [
        {
          type: 'tool-call',
          toolCallId: 'call_1',
          toolName: 'mcp_utp_utp_cart_list',
          args: {} as never,
          argsText: '{}',
          result: { result: '3 items', ui }
        }
      ]
    }
  ]

  it('re-attaches live-only ui payloads onto hydrated tool parts by toolCallId (D1)', () => {
    const hydrated: ChatMessage[] = [
      {
        id: 'stored-9',
        role: 'assistant',
        parts: [
          {
            type: 'tool-call',
            toolCallId: 'call_1',
            toolName: 'mcp_utp_utp_cart_list',
            args: {} as never,
            argsText: '{}',
            result: { result: '3 items' }
          }
        ]
      }
    ]

    const merged = preserveMcpUiCards(hydrated, liveMessages)
    const part = merged[0].parts[0] as Extract<ChatMessagePart, { type: 'tool-call' }>

    expect((part.result as { ui?: unknown }).ui).toEqual(ui)
  })

  it('leaves messages untouched when nothing carries ui', () => {
    const hydrated: ChatMessage[] = [{ id: 'x', role: 'assistant', parts: [] }]

    expect(preserveMcpUiCards(hydrated, [{ id: 'y', role: 'assistant', parts: [] }])).toBe(hydrated)
  })
})
