import { describe, expect, it } from 'vitest'

import type { ChatMessage } from '@/lib/chat-messages'
import { messagesIfTranscriptBehind, surplusIsCompetingView } from '@/lib/stale-transcript-guard'

function userMessage(id: string, text: string, rowId?: number): ChatMessage {
  return {
    id,
    role: 'user',
    parts: [{ type: 'text', text }],
    ...(rowId !== undefined && { rowId })
  }
}

function assistantMessage(id: string, text: string, rowId?: number): ChatMessage {
  return {
    id,
    role: 'assistant',
    parts: [{ type: 'text', text }],
    ...(rowId !== undefined && { rowId })
  }
}

describe('surplusIsCompetingView', () => {
  it('turn-death residue with an unreconciled optimistic user row is not a competing view', () => {
    // #124005: a turn died on an approval timeout — the server persisted the
    // user row (ids the window never received) plus the tool/notice rows, while
    // the window holds the optimistic copy under its own id with no rowId. The
    // local view being behind is the expected aftermath, not a fork.
    const local = [userMessage('opt-old', 'deploy the schema')]
    const refreshed = [
      userMessage('msg-10', 'deploy the schema', 10),
      assistantMessage('msg-11', '⌛ Approval timed out after 5 minutes', 11)
    ]

    expect(surplusIsCompetingView(local, refreshed)).toBe(false)
  })

  it('a user row with unknown text is a competing view', () => {
    const local = [userMessage('opt-old', 'deploy the schema')]
    const refreshed = [
      userMessage('msg-10', 'deploy the schema', 10),
      userMessage('msg-12', 'typed in another window'),
      assistantMessage('msg-13', 'ok', 13)
    ]

    expect(surplusIsCompetingView(local, refreshed)).toBe(true)
  })

  it('user rows matching by rowId or id are known', () => {
    const byRowId = [userMessage('msg-10', 'deploy the schema', 10)]
    expect(surplusIsCompetingView(byRowId, [
      userMessage('msg-10', 'deploy the schema', 10),
      assistantMessage('msg-11', 'done', 11)
    ])).toBe(false)

    const byId = [userMessage('msg-10', 'deploy the schema')]
    expect(surplusIsCompetingView(byId, [
      userMessage('msg-10', 'deploy the schema'),
      assistantMessage('msg-11', 'done')
    ])).toBe(false)
  })

  it('assistant/tool-only surplus is never a competing view', () => {
    const local = [userMessage('opt-old', 'deploy the schema')]
    const refreshed = [
      userMessage('msg-10', 'deploy the schema', 10),
      assistantMessage('msg-11', 'running the command', 11),
      assistantMessage('msg-12', '⌛ Approval timed out', 12)
    ]

    expect(surplusIsCompetingView(local, refreshed)).toBe(false)
  })
})

describe('messagesIfTranscriptBehind', () => {
  it('a remote page ahead of the local view is returned for the guard', () => {
    const local = [userMessage('opt-old', 'deploy the schema')]
    const remote = [userMessage('msg-10', 'deploy the schema', 10), assistantMessage('msg-11', 'done', 11)]

    expect(messagesIfTranscriptBehind(local, remote)).toEqual(remote)
  })
})
