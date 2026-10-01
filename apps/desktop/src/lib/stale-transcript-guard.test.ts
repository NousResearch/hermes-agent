import { describe, expect, it } from 'vitest'

import type { ChatMessage } from '@/lib/chat-messages'
import type { SessionMessage } from '@/types/hermes'

import { toChatMessages } from './chat-messages'
import { messagesIfTranscriptBehind } from './stale-transcript-guard'

/**
 * The guard refuses a send when the authoritative latest page holds MORE
 * transcript than the window does, rather than forking the session (#65047).
 * It measures that difference with `remoteChat.length > localMessages.length`
 * after `toChatMessages`.
 *
 * A backend-authored NOTICE also arrives as a `ChatMessage` (`model changed`,
 * `background agent work finished`): `tui_gateway/server.py` persists
 * `display_kind=model_switch` with `role=user` on an in-place model switch, and
 * hydration renders it as a system row. Nothing about it means another window
 * exists — but it inflates the page by one message per event, so the guard
 * reported "This window was behind another view of the same chat" to a user who
 * had only switched models, refused the send, and said the same thing on every
 * retry. Staleness must be measured in AUTHORED content, not array length.
 */

const row = (over: Partial<SessionMessage> & Pick<SessionMessage, 'role'>): SessionMessage => ({
  content: '',
  timestamp: 1_700_000_000,
  ...over
})

const userTurn = (id: number, text: string): SessionMessage =>
  row({ content: text, id, role: 'user', timestamp: 1_700_000_000 + id })

const assistantTurn = (id: number, text: string): SessionMessage =>
  row({ content: text, id, role: 'assistant', timestamp: 1_700_000_000 + id })

/** What an in-place model switch persists, exactly as the gateway writes it. */
const modelSwitchNotice = (id: number): SessionMessage =>
  row({
    content: 'switched to another model',
    display_kind: 'model_switch',
    id,
    role: 'user',
    timestamp: 1_700_000_000 + id
  })

describe('messagesIfTranscriptBehind', () => {
  it('does not treat a backend-authored notice as another view being ahead', () => {
    const windowMessages = toChatMessages([userTurn(1, 'ask'), assistantTurn(2, 'answer')])
    const page = toChatMessages([userTurn(1, 'ask'), assistantTurn(2, 'answer'), modelSwitchNotice(3)])

    // The notice is a real message in the page: this is what skews a length compare.
    expect(page).toHaveLength(3)
    expect(page.map(message => message.role)).toEqual(['user', 'assistant', 'system'])

    expect(messagesIfTranscriptBehind(windowMessages, page)).toBeNull()
  })

  it('still refuses when another view advanced the chat with a real turn', () => {
    const windowMessages = toChatMessages([userTurn(1, 'ask'), assistantTurn(2, 'answer')])

    const page = toChatMessages([
      userTurn(1, 'ask'),
      assistantTurn(2, 'answer'),
      userTurn(4, 'sent from another window')
    ])

    expect(messagesIfTranscriptBehind(windowMessages, page)).not.toBeNull()
  })

  it('still refuses when a notice arrives alongside a reply this window never saw', () => {
    const windowMessages = toChatMessages([userTurn(1, 'ask'), assistantTurn(2, 'answer')])

    const page = toChatMessages([
      userTurn(1, 'ask'),
      assistantTurn(2, 'answer'),
      modelSwitchNotice(3),
      assistantTurn(4, 'a reply this window never rendered')
    ])

    expect(messagesIfTranscriptBehind(windowMessages, page)).not.toBeNull()
  })

  it('is current when both sides hold the same notices and the same turns', () => {
    const rows = [userTurn(1, 'ask'), assistantTurn(2, 'answer'), modelSwitchNotice(3)]

    expect(messagesIfTranscriptBehind(toChatMessages(rows), toChatMessages(rows))).toBeNull()
  })

  it('is current when the page is empty, and refreshes a window that holds nothing', () => {
    const rows = [userTurn(1, 'ask'), assistantTurn(2, 'answer')]

    expect(messagesIfTranscriptBehind(toChatMessages(rows), [])).toBeNull()
    expect(messagesIfTranscriptBehind([], toChatMessages(rows))).toEqual(toChatMessages(rows))
  })

  // #125975: a tool-using turn folds into one bubble, and the two paths that
  // build it bind different ends of the span. Live settle stamps the turn's
  // FINAL row (withPersistedIdentity → rowId + the last text part's
  // sourceRowId); hydration keeps the folded bubble's FIRST row and stamps
  // every text part with its own sourceRowId. When the latest page starts
  // mid-turn, its first durable row is one the window's settled bubble never
  // carried — so a rowId-only tip compare misses, the graft appends, and the
  // count reports "behind": the first send after a tool-using turn bounces
  // with "Chat out of date" in a single window. The tip must read the highest
  // durable address the bubble carries (rowId or any text part's sourceRowId).
  it('matches tips when the live-settled bubble binds the final row and the page starts mid-turn', () => {
    const text = (rowId: number, body: string) => ({ sourceRowId: rowId, text: body, type: 'text' as const })
    const streamed = (body: string) => ({ text: body, type: 'text' as const })
    const tool = () => ({ toolCallId: 'call-1', toolName: 'shell', type: 'tool-call' as const })

    // The window settled both tool turns live: each folded bubble is addressed
    // by its FINAL row (withPersistedIdentity), and only the final text part
    // carries a durable sourceRowId — the streamed pre-tool text has none.
    const localWindow: ChatMessage[] = [
      { id: 'u1', parts: [streamed('ask')], role: 'user', rowId: 1 },
      {
        id: 'a-fold1-live',
        parts: [streamed('checking'), tool(), text(5, 'final one')],
        role: 'assistant',
        rowId: 5
      },
      { id: 'u3', parts: [streamed('run the tests')], role: 'user', rowId: 6 },
      {
        id: 'a-fold2-live',
        parts: [streamed('verifying'), tool(), text(10, 'final two')],
        role: 'assistant',
        rowId: 10
      }
    ]

    // The refreshed latest page starts INSIDE the first folded turn: hydration
    // addresses each folded bubble by its FIRST row (3, 8), the final rows
    // surviving only as text-part sourceRowIds. The window's settled bubbles
    // never carried rows 3 or 8, so a rowId-only tip compare sees 10 vs 8,
    // the graft finds no anchor, and the merge reports the window behind —
    // the first send after a tool-using turn bounces with "Chat out of date"
    // in a single window (#125975).
    const latestPage: ChatMessage[] = [
      {
        id: 'a-fold1-page',
        parts: [text(3, 'checking'), tool(), text(5, 'final one')],
        role: 'assistant',
        rowId: 3
      },
      { id: 'u3', parts: [streamed('run the tests')], role: 'user', rowId: 6 },
      {
        id: 'a-fold2-page',
        parts: [text(8, 'verifying'), tool(), text(10, 'final two')],
        role: 'assistant',
        rowId: 8
      }
    ]

    expect(messagesIfTranscriptBehind(localWindow, latestPage)).toBeNull()
  })

  it('correctly identifies identical durable rows despite missing synthetic row', () => {
    const text = (rowId: number, body: string) => ({ sourceRowId: rowId, text: body, type: 'text' as const })
    const tool = () => ({ toolCallId: 'call-1', toolName: 'shell', type: 'tool-call' as const })
    
    const localMessages: ChatMessage[] = [
      { id: 'msg1', parts: [text(10, 'hello')], role: 'user', rowId: 10 },
      {
        id: 'msg2',
        parts: [text(11, 'tool-call'), tool(), text(12, 'response')],
        role: 'assistant',
        rowId: 11
      }
    ]

    const remoteChat: ChatMessage[] = [
      { id: 'msg1', parts: [text(10, 'hello')], role: 'user', rowId: 10 },
      {
        id: 'synthetic',
        parts: [text(11, 'tool-call'), tool(), text(12, 'response')],
        role: 'assistant'
      } as ChatMessage
    ]

    expect(messagesIfTranscriptBehind(localMessages, remoteChat)).toBeNull()
  })
})
