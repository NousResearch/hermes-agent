import type { PersistedTurn } from '@hermes/shared'
import { describe, expect, it } from 'vitest'

import { graftRefreshedTailOntoBackfill } from '@/app/chat/transcript-backfill'
import { type ChatMessage, chatMessageText, toChatMessages } from '@/lib/chat-messages'
import type { SessionMessage } from '@/types/hermes'

/**
 * A turn interrupted before it produced any assistant text. Its completion does carry a receipt
 * (measured: `{"row_ids":[103341..103349],"complete":false,...}`), but the turn's last stored row is
 * the empty `hidden` marker `hidden_interrupt_placeholder_row()` writes, which the page never paints,
 * and the only assistant bubble on screen is the tool card. That bubble has no stored id, so the
 * merge has to recognise it by the receipt alone — while a bubble that does carry text must still
 * stay, because the page may be missing exactly that answer.
 */

const at = (id: number) => 1_000 + id

const user = (id: number, text: string): SessionMessage =>
  ({ id, role: 'user', content: text, timestamp: at(id) }) as SessionMessage

const toolRound = (id: number, call: string): SessionMessage[] => [
  ({
    id,
    role: 'assistant',
    content: '',
    timestamp: at(id),
    tool_calls: [{ id: call, type: 'function', function: { name: 'terminal', arguments: '{"command":"ls"}' } }]
  }) as SessionMessage,
  ({ id: id + 1, role: 'tool', content: 'ok', tool_call_id: call, tool_name: 'terminal', timestamp: at(id + 1) }) as SessionMessage
]

const hiddenMarker = (id: number): SessionMessage =>
  ({ id, role: 'assistant', content: '', display_kind: 'hidden', timestamp: at(id) }) as SessionMessage

const answer = (id: number, text: string): SessionMessage =>
  ({ id, role: 'assistant', content: text, timestamp: at(id) }) as SessionMessage

/** The page's newest window starts on a stored row the on-screen window does not hold. */
const olderRow = (): SessionMessage => answer(103299, 'older answer')

/** Three tool rounds, the interrupt marker, then the next turn's prompt and reply. */
const page = (): SessionMessage[] => [
  user(103341, 'hermes desktop ma blad w wyswietlaniu danych sesji.'),
  ...toolRound(103342, 'c1'),
  ...toolRound(103345, 'c2'),
  ...toolRound(103347, 'c3'),
  hiddenMarker(103349),
  user(103350, 'tylko w renderze'),
  answer(103351, 'Jasne — czyli dane sa OK.')
]

/** The receipt `persisted_turn_receipt()` publishes for that turn (status `interrupted`). */
const partialReceipt: PersistedTurn = {
  row_ids: [103341, 103342, 103343, 103345, 103346, 103347, 103348, 103349],
  complete: false,
  user_row_id: 103341,
  user_row_ids: [103341]
}

const prompt = (rowId?: number): ChatMessage => ({
  id: 'u-103341',
  role: 'user',
  parts: [{ type: 'text', text: 'hermes desktop ma blad w wyswietlaniu danych sesji.' }],
  ...(rowId !== undefined ? { rowId } : {})
})

const toolCard = (extra: Partial<ChatMessage> = {}): ChatMessage => ({
  id: 'live-tools',
  role: 'assistant',
  parts: [
    { type: 'tool-call', toolCallId: 'c1', toolName: 'terminal', args: { command: 'ls' }, argsText: '{}', result: 'ok' },
    { type: 'tool-call', toolCallId: 'c2', toolName: 'terminal', args: { command: 'ls' }, argsText: '{}', result: 'ok' },
    { type: 'tool-call', toolCallId: 'c3', toolName: 'terminal', args: { command: 'ls' }, argsText: '{}', result: 'ok' }
  ],
  ...extra
})

const cards = (messages: ChatMessage[]) =>
  messages.filter(message => message.role === 'assistant' && message.parts.some(part => part.type === 'tool-call')).length

describe('a turn interrupted before any output', () => {
  it('draws its tool card once when the refreshed page holds the turn', () => {
    // The page starts on a row the window lacks, so the graft merges by stored id and runs the
    // receipt drop instead of replacing the window outright.
    const merged = graftRefreshedTailOntoBackfill(toChatMessages([olderRow(), ...page()]), [
      prompt(103341),
      toolCard({ persistedTurn: partialReceipt })
    ])

    expect(cards(merged)).toBe(1)
  })

  it('keeps a live answer the page never painted, even with a partial receipt (control)', () => {
    // The final answer failed to store: the receipt's last row is a tool row, the page paints no
    // text for it, and the window's bubble is the only copy of the answer.
    const failedStore: PersistedTurn = {
      row_ids: [103341, 103342, 103343],
      complete: false,
      user_row_id: 103341,
      user_row_ids: [103341]
    }

    const merged = graftRefreshedTailOntoBackfill(toChatMessages([olderRow(), ...page().slice(0, 4)]), [
      prompt(103341),
      {
        id: 'live-answer',
        role: 'assistant',
        parts: [{ type: 'text', text: 'FINAL A — only on screen' }],
        persistedTurn: failedStore
      }
    ])

    expect(merged.map(message => chatMessageText(message)).join('\n')).toContain('FINAL A — only on screen')
  })
})
