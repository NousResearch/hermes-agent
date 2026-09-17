import { describe, expect, it } from 'vitest'

import type { SessionMessage } from '@/types/hermes'

import { chatMessageText, toChatMessages } from './chat-messages'

/**
 * Regression spec for the 2026-09-18 desktop bug (session
 * 20260917_223008_136f64, rows 218953..218972).
 *
 * Backend truth (verified from state.db): every reply persisted with full
 * content, interleaved correctly. Nothing was lost on the gateway side.
 *
 * The VIEW-layer defect:
 *   LEAK — an assistant row with empty `content` but non-empty `reasoning`
 *   (rows 218968/218970 from the memory-save retry churn) survives hydration
 *   and its reasoning text renders inline where an answer would appear. This
 *   is the "My replacement was too long…" / "Already replied…" drafting text
 *   the user quoted. Reasoning is a private scratchpad, not a substitute for
 *   an answer.
 *
 * Test 1 is the guard that hydration preserves the four mid-replies as
 * distinct bubbles (it passes — proving the mid-reply loss is NOT a
 * hydration/storage loss). Test 2 encodes the leak bug.
 */
const T = 1789681082.0 // ~2026-09-18 00:38

const fixtureRows: SessionMessage[] = [
  { id: 218957, role: 'user', content: 'yep the platform price is reasonable…', timestamp: T - 80 },
  { id: 218958, role: 'assistant', content: "Good instinct — \"don't really need it\" is the filter that saves you 1700€…", reasoning: 'platform tempting but unneeded', timestamp: T - 76 },
  { id: 218959, role: 'user', content: 'Also i would probably just end up keeping the gpu…', timestamp: T - 50 },
  { id: 218960, role: 'assistant', content: 'Fair — and that changes the math, because if you are keeping the 5070 Ti…', reasoning: 'keep gpu changes math', timestamp: T - 47 },
  { id: 218961, role: 'user', content: 'nah ddr4 is necessary for hotswap…', timestamp: T - 28 },
  { id: 218962, role: 'assistant', content: 'Touché — that is a need, not a want…', reasoning: 'ddr4 is a requirement', timestamp: T - 26 },
  { id: 218963, role: 'user', content: 'still platform upgrade would be nice…', timestamp: T - 12 },
  { id: 218964, role: 'assistant', content: 'Sensible wins. And you are missing the sharpest reason…', reasoning: 'sensible wins', timestamp: T - 10 },
  { id: 218965, role: 'user', content: 'but the money is not the issue either…', timestamp: T - 5 },
  { id: 218966, role: 'assistant', content: 'Ha — okay, fair. You impulse-buy 700€ paintings from Canada…', reasoning: 'money not the issue', timestamp: T - 3 },
  // memory-save retry churn at the tail:
  { id: 218967, role: 'tool', content: '{"success": false, "error": "Replacement would put memory at…"}', timestamp: T - 2 },
  { id: 218968, role: 'assistant', content: '', reasoning: 'My replacement was too long. Let me compact it while keeping the new fact.', timestamp: T - 2 },
  { id: 218969, role: 'tool', content: '{"success": false, "error": "Replacement would put memory at…"}', timestamp: T - 1 },
  { id: 218970, role: 'assistant', content: '', reasoning: '', timestamp: T - 1 },
  { id: 218971, role: 'tool', content: '{"success": true, "done": true, "target": "user"}', timestamp: T - 0.5 },
  { id: 218972, role: 'assistant', content: 'Noted. Anyway — go get the RAM when you see it…', reasoning: 'Already replied. Nothing more to add — I said my piece…', timestamp: T }
]

describe('hydration of the 2026-09-18 mid-reply fixture', () => {
  it('preserves the four mid-replies as distinct visible assistant bubbles (guard test)', () => {
    const messages = toChatMessages(fixtureRows)
    const assistants = messages.filter(m => m.role === 'assistant' && !m.hidden)
    const texts = assistants.map(chatMessageText)

    expect(texts[0]).toContain('Good instinct')
    expect(texts[1]).toContain('Fair — and that changes the math')
    expect(texts[2]).toContain('Touché')
    expect(texts[3]).toContain('Sensible wins')
  })

  it('does not leak reasoning-only assistant rows as visible content', () => {
    const messages = toChatMessages(fixtureRows)
    // Rows 218968/218970 (empty `content`) must not surface "My replacement
    // was too long…" / "Already replied…" drafting text as visible bubbles.
    // A real answer may keep its own reasoning as a thinking block — the
    // invariant is that NO bubble is text-less yet shows reasoning text.
    const reasoningOnlyBubbles = messages.filter(
      m => m.role === 'assistant' && !chatMessageText(m).trim() && m.parts.some(p => p.type === 'reasoning')
    )

    expect(reasoningOnlyBubbles).toHaveLength(0)
    expect(
      messages.some(m =>
        !chatMessageText(m).trim() &&
        /replacement|Already replied/i.test(
          m.parts.filter(p => p.type === 'reasoning').map(p => p.text).join(' ')
        )
      )
    ).toBe(false)
  })

  it('never grafts retry-churn reasoning onto a retained text bubble (reviewer probe)', () => {
    // Exact-head probe from the PR review: a real reply, an unmatched tool row,
    // then an assistant row with empty content + reasoning only (rows 218966+
    // shape). The reasoning must NOT be appended to the retained text bubble —
    // a post-merge filter cannot see it because the bubble still has text.
    const messages = toChatMessages([
      { id: 1, role: 'assistant', content: 'A real reply.', timestamp: T },
      { id: 2, role: 'tool', content: '{"error":"boom"}', timestamp: T + 1 },
      { id: 3, role: 'assistant', content: '', reasoning: 'Private retry planning.', timestamp: T + 2 }
    ])
    const visible = messages.filter(m => m.role === 'assistant')

    expect(visible).toHaveLength(1)
    expect(chatMessageText(visible[0])).toBe('A real reply.')
    expect(visible[0].parts.some(p => p.type === 'reasoning' && /Private retry planning/.test(p.text))).toBe(false)
  })

  it('drops hidden display_kind assistant rows whose only part is reasoning', () => {
    // Cassandra audit finding (2026-09-18): `reasoningPart` is pushed for any
    // assistant row with reasoning regardless of display_kind, and 'hidden'
    // is not in the system-role mapping — so a hidden row with only reasoning
    // used to render a Thinking block for a message the backend marked
    // invisible. The filter change drops it because reasoning is the only part.
    const hidden = toChatMessages([
      { id: 9991, role: 'assistant', display_kind: 'hidden', content: '', reasoning: 'private interruption thought', timestamp: T + 100 }
    ])

    expect(hidden.some(m => !chatMessageText(m).trim() && /private interruption thought/.test(
      m.parts.filter(p => p.type === 'reasoning').map(p => p.text).join(' ')
    ))).toBe(false)
  })

  it('rescues reply text from the codex sidecar even with reasoning present', () => {
    // A reasoning-only row that ALSO carries a codex message sidecar must keep
    // its visible reply: the sidecar rescue runs before the filter and wins.
    const rows = [
      { id: 9992, role: 'assistant', content: '', reasoning: 'thought first', timestamp: T + 100,
        codex_message_items: [{ type: 'message', role: 'assistant', content: [{ type: 'output_text', text: 'Real reply via sidecar.' }] }] }
    ] as SessionMessage[]

    const messages = toChatMessages(rows)
    const visible = messages.filter(m => m.role === 'assistant')

    expect(visible.length).toBeGreaterThan(0)
    expect(visible.some(m => chatMessageText(m).includes('Real reply via sidecar.'))).toBe(true)
  })
})
