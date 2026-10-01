import { describe, expect, it } from 'vitest'

import type { ChatMessage } from '@/lib/chat-messages'
import type { SessionMessage } from '@/types/hermes'

import { toChatMessages } from './chat-messages'
import { messagesIfTranscriptBehind, surplusIsCompetingView } from './stale-transcript-guard'

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
})

function competingViewUserMessage(id: string, text: string, rowId?: number): ChatMessage {
  return {
    id,
    role: 'user',
    parts: [{ type: 'text', text }],
    ...(rowId !== undefined && { rowId })
  }
}

function competingViewAssistantMessage(id: string, text: string, rowId?: number): ChatMessage {
  return {
    id,
    role: 'assistant',
    parts: [{ type: 'text', text }],
    ...(rowId !== undefined && { rowId })
  }
}

describe('surplusIsCompetingView', () => {
  // #124005: a turn that died on an approval timeout leaves its user row plus
  // tool/assistant residue server-side under durable ids the window never saw —
  // the window holds only its optimistic copy under a local id with no rowId.
  // That residue is this window's own turn aftermath, not a competing view.
  it('turn-death residue with an unreconciled optimistic user row is not a competing view', () => {
    const local = [competingViewUserMessage('opt-old', 'deploy the schema')]
    const refreshed = [
      competingViewUserMessage('msg-10', 'deploy the schema', 10),
      competingViewAssistantMessage('msg-11', '⌛ Approval timed out after 5 minutes', 11)
    ]

    expect(surplusIsCompetingView(local, refreshed, { optimisticMessageId: 'opt-old' })).toBe(false)
  })

  it('a plain completed tool turn single-window surplus is not a competing view', () => {
    // Issue comment 2: a successful plain tool turn reproduces the refusal
    // single-window — the server folded rows this window never streamed back.
    const local = [competingViewUserMessage('opt-old', 'run the tests')]
    const refreshed = [
      competingViewUserMessage('msg-10', 'run the tests', 10),
      competingViewAssistantMessage('msg-11', 'checking', 11),
      competingViewAssistantMessage('msg-12', 'all green', 12)
    ]

    expect(surplusIsCompetingView(local, refreshed, { optimisticMessageId: 'opt-old' })).toBe(false)
  })

  it('rejects a repeated prompt from another view even when the text matches', () => {
    const local = [competingViewUserMessage('prior', 'run the tests')]
    const refreshed = [
      competingViewUserMessage('prior', 'run the tests', 10),
      competingViewUserMessage('peer', 'run the tests', 12)
    ]

    expect(surplusIsCompetingView(local, refreshed)).toBe(true)
  })

  it('a user row with unknown text is a competing view', () => {
    const local = [competingViewUserMessage('opt-old', 'deploy the schema')]
    const refreshed = [
      competingViewUserMessage('msg-10', 'deploy the schema', 10),
      competingViewUserMessage('msg-12', 'typed in another window'),
      competingViewAssistantMessage('msg-13', 'ok', 13)
    ]

    expect(surplusIsCompetingView(local, refreshed)).toBe(true)
  })

  it('user rows matching by rowId or id are known', () => {
    const byRowId = [competingViewUserMessage('msg-10', 'deploy the schema', 10)]
    expect(surplusIsCompetingView(byRowId, [
      competingViewUserMessage('msg-10', 'deploy the schema', 10),
      competingViewAssistantMessage('msg-11', 'done', 11)
    ])).toBe(false)

    const byId = [competingViewUserMessage('msg-10', 'deploy the schema')]
    expect(surplusIsCompetingView(byId, [
      competingViewUserMessage('msg-10', 'deploy the schema'),
      competingViewAssistantMessage('msg-11', 'done')
    ])).toBe(false)
  })

  it('assistant/tool-only surplus is never a competing view', () => {
    const local = [competingViewUserMessage('opt-old', 'deploy the schema')]
    const refreshed = [
      competingViewUserMessage('msg-10', 'deploy the schema', 10),
      competingViewAssistantMessage('msg-11', 'running the command', 11),
      competingViewAssistantMessage('msg-12', '⌛ Approval timed out', 12)
    ]

    expect(surplusIsCompetingView(local, refreshed, { optimisticMessageId: 'opt-old' })).toBe(false)
  })
})
