import { describe, expect, it } from 'vitest'

import type { ChatMessage } from '@/lib/chat-messages'

import { continueStoppedReply } from '../use-message-stream/interrupted-reply'

import { chatMessagesEquivalent } from './message-equivalence'
import { overlayConcurrentMessageChanges } from './utils'

// `continued` hides the stopped mark of a reply the next turn went on with
// (use-message-stream/interrupted-reply.ts). It is set once, at that turn's
// message.start, and nothing sets it again, so every comparison that decides
// whether a row changed has to see it.
describe('a continued reply in the transcript comparisons', () => {
  const stopped = {
    id: 'assistant-reply',
    interrupted: true,
    parts: [{ toolCallId: 'call-read', toolName: 'read_file', type: 'tool-call' }],
    role: 'assistant'
  } as ChatMessage

  it('is a change', () => {
    expect(chatMessagesEquivalent(stopped, { ...stopped, continued: true })).toBe(false)
    expect(chatMessagesEquivalent(stopped, { ...stopped, continued: false })).toBe(true)
    expect(chatMessagesEquivalent({ ...stopped, continued: true }, { ...stopped, continued: true })).toBe(true)
  })

  // A folder grant restarts the backend: the chat resumes, and the continuation's
  // message.start can arrive while that resume still waits for its transcript page.
  it('survives a resume that was waiting when the continuation started', () => {
    const baseline = [stopped]
    const current = continueStoppedReply(baseline, {})
    const page = [{ ...stopped }]

    const [reply] = overlayConcurrentMessageChanges(page, baseline, current)

    expect(current[0].continued).toBe(true)
    expect(reply.continued).toBe(true)
  })
})
