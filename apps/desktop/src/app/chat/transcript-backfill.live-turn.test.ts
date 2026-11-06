import { describe, expect, it } from 'vitest'

import { graftRefreshedTailOntoBackfill } from '@/app/chat/transcript-backfill'
import {
  preserveLocalPendingTurnMessages,
  reconcileDurableHistory
} from '@/app/session/hooks/use-session-actions/utils'
import { type ChatMessage, chatMessageText, toChatMessages } from '@/lib/chat-messages'
import type { SessionMessage } from '@/types/hermes'

/**
 * A running turn stores its prompt and each finished tool round as it goes, before any completion
 * receipt exists. A refresh that lands mid-turn (reopening the chat, a background read) brings
 * those rows while the window still draws the turn live (#145333).
 */

const at = (id: number) => 1_000 + id

const user = (id: number, text: string, submission?: string): SessionMessage =>
  ({
    id,
    role: 'user',
    content: text,
    timestamp: at(id),
    ...(submission ? { display_metadata: { submission_id: submission } } : {})
  }) as SessionMessage

const narration = (id: number, text: string, call: string): SessionMessage =>
  ({
    id,
    role: 'assistant',
    content: text,
    timestamp: at(id),
    tool_calls: [{ id: call, type: 'function', function: { name: 'terminal', arguments: '{"command":"echo"}' } }]
  }) as SessionMessage

const tool = (id: number, call: string): SessionMessage =>
  ({ id, role: 'tool', content: 'ok', tool_call_id: call, tool_name: 'terminal', timestamp: at(id) }) as SessionMessage

const answer = (id: number, text: string): SessionMessage =>
  ({ id, role: 'assistant', content: text, timestamp: at(id) }) as SessionMessage

const olderTurn = (first: number, tag: string): SessionMessage[] => [
  user(first, `${tag} prompt`),
  narration(first + 1, `${tag} look one`, `${tag}-1`),
  tool(first + 2, `${tag}-1`),
  narration(first + 3, `${tag} look two`, `${tag}-2`),
  tool(first + 4, `${tag}-2`),
  answer(first + 5, `${tag} answer`)
]

const PROMPT = 'tidy the release notes'

const liveCall = (toolCallId: string) => ({
  type: 'tool-call' as const,
  toolCallId,
  toolName: 'terminal',
  args: { command: 'echo' } as never,
  argsText: '{"command":"echo"}',
  result: 'ok',
  completedAt: 2_000
})

/** The window as the live stream drew the running turn: the send, then the streaming reply. */
const liveTurn = (): ChatMessage[] => [
  { id: 'user-sub-live', role: 'user', parts: [{ type: 'text', text: PROMPT }], timestamp: at(13), attachmentRefs: [] },
  {
    id: 'assistant-stream-1',
    role: 'assistant',
    pending: true,
    parts: [
      { type: 'text', text: 'reading the notes' },
      liveCall('live-1'),
      { type: 'text', text: 'checking the links' },
      liveCall('live-2')
    ]
  }
]

/** What the store holds of that turn so far: the prompt under its send id and two finished tool rounds. */
const storedTurn = [
  user(13, PROMPT, 'sub-live'),
  narration(14, 'reading the notes', 'live-1'),
  tool(15, 'live-1'),
  narration(16, 'checking the links', 'live-2'),
  tool(17, 'live-2')
]

const history = [...olderTurn(1, 'old-a'), ...olderTurn(7, 'old-b')]

const count = (messages: ChatMessage[], needle: string) =>
  messages.filter(message => chatMessageText(message).includes(needle)).length

const compositions = {
  // use-background-sync's sessions.changed / post-turn reads.
  'a background refresh': (page: ChatMessage[], window: ChatMessage[]) =>
    preserveLocalPendingTurnMessages(graftRefreshedTailOntoBackfill(page, window), window),
  // The warm reopen of the chat (use-session-actions' session.activate branch).
  'a warm reopen': (page: ChatMessage[], window: ChatMessage[]) =>
    reconcileDurableHistory(graftRefreshedTailOntoBackfill(page, window), window)
}

describe('a refresh that lands while the turn runs', () => {
  describe.each(Object.entries(compositions))('through %s', (_label, refresh) => {
    it.each([
      // Long chat: the newest page starts inside an older turn (stored-id merge).
      ['a page that starts inside an older turn', history.slice(3)],
      // Short chat: the page is the whole conversation (taken as is).
      ['a page that holds the whole chat', history]
    ])('draws the prompt and the reply once after %s', (_page, pageHead) => {
      const window = [...toChatMessages(history), ...liveTurn()]
      const merged = refresh(toChatMessages([...pageHead, ...storedTurn]), window)

      for (const text of [PROMPT, 'reading the notes', 'checking the links']) {
        expect(count(merged, text), text).toBe(1)
      }

      expect(merged.findLastIndex(message => message.role === 'user')).toBeLessThan(
        merged.findIndex(message => chatMessageText(message).includes('reading the notes'))
      )
    })
  })

  it('keeps a live send whose words the page stores under another send', () => {
    // The same words sent earlier and stored; this send has no row on the page yet.
    const earlier = [user(13, PROMPT, 'sub-earlier'), answer(14, 'notes tidied')]
    const window = [...toChatMessages([...history, ...earlier]), ...liveTurn()]
    const merged = graftRefreshedTailOntoBackfill(toChatMessages([...history.slice(3), ...earlier]), window)

    expect(count(merged, PROMPT)).toBe(2)
    expect(merged.at(-2)?.id).toBe('user-sub-live')
    expect(merged.at(-1)?.id).toBe('assistant-stream-1')
  })
})
