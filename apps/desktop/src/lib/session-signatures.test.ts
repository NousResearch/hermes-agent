import { describe, expect, it } from 'vitest'

import type { SessionInfo } from '@/hermes'

import { sameCronSignature, sessionMessagesSignature } from './session-signatures'

const session = (id: string, title: string | null, extra: Partial<SessionInfo> = {}): SessionInfo =>
  ({ id, title, ...extra }) as SessionInfo

describe('sameCronSignature', () => {
  it('is false when the lengths differ', () => {
    expect(sameCronSignature([session('a', 't')], [])).toBe(false)
  })

  it.each(['cwd', 'git_repo_root', 'git_branch'] as const)('compares %s including cleared metadata', field => {
    const populated = [session('a', 't', { [field]: 'workspace' })]
    const cleared = [session('a', 't', { [field]: null })]
    expect(sameCronSignature(populated, cleared)).toBe(false)
    expect(sameCronSignature(cleared, populated)).toBe(false)
    expect(sameCronSignature(cleared, [session('a', 't', { [field]: null })])).toBe(true)
    expect(sameCronSignature(populated, [session('a', 't', { [field]: 'workspace' })])).toBe(true)
  })

  it('is true when ids and titles match in order', () => {
    const a = [session('a', 'one'), session('b', 'two')]
    const b = [session('a', 'one'), session('b', 'two')]
    expect(sameCronSignature(a, b)).toBe(true)
  })

  it('is false when a title changed', () => {
    const a = [session('a', 'one')]
    const b = [session('a', 'renamed')]
    expect(sameCronSignature(a, b)).toBe(false)
  })

  it('is false when order differs', () => {
    const a = [session('a', 't'), session('b', 't')]
    const b = [session('b', 't'), session('a', 't')]
    expect(sameCronSignature(a, b)).toBe(false)
  })

  // A pin-only page must reach $sessions: session-pin-sync treats the row as
  // authoritative and releases its write guard when a page confirms the value
  // it wrote. Gating that page out froze the row and re-pinned what the user
  // had just unpinned (#76919).
  it('is false when only the pinned flag changed', () => {
    const a = [session('a', 't', { pinned: true })]
    const b = [session('a', 't', { pinned: false })]
    expect(sameCronSignature(a, b)).toBe(false)
  })

  it('is false when only the archived flag changed', () => {
    const a = [session('a', 't', { archived: false })]
    const b = [session('a', 't', { archived: true })]
    expect(sameCronSignature(a, b)).toBe(false)
  })
})

describe('sessionMessagesSignature', () => {
  const msg = (role: string, content: string) =>
    ({ role, content }) as Parameters<typeof sessionMessagesSignature>[0][number]

  it('is stable for identical transcripts', () => {
    expect(sessionMessagesSignature([msg('user', 'hi')])).toBe(sessionMessagesSignature([msg('user', 'hi')]))
  })

  it('changes when content changes', () => {
    expect(sessionMessagesSignature([msg('user', 'hi')])).not.toBe(sessionMessagesSignature([msg('user', 'yo')]))
  })

  it('changes when a message is appended', () => {
    const one = [msg('user', 'hi')]
    expect(sessionMessagesSignature(one)).not.toBe(sessionMessagesSignature([...one, msg('assistant', 'hey')]))
  })

  // A stored correction that only moves a presentation field is authoritative
  // for the rendered transcript; the poll signature has to see it or the old
  // answer repaints never (#140396).
  const presentationDeltas: readonly { delta: Record<string, unknown>; field: string }[] = [
    { delta: { display_content: 'Corrected public answer.' }, field: 'display_content' },
    { delta: { display_kind: 'hidden' }, field: 'display_kind' },
    { delta: { display_reasoning: 'Verified reasoning.' }, field: 'display_reasoning' },
    { delta: { display_commentary: ['Public note.'] }, field: 'display_commentary' },
    { delta: { display_metadata: { reactions: [{ at: 3, author: 'user', emoji: '👍' }] } }, field: 'display_metadata' },
    { delta: { row_id: 20 }, field: 'row_id' },
    {
      delta: {
        tool_calls: [
          { function: { arguments: '{"path":"notes.md"}', name: 'read_file' }, id: 'call_read', type: 'function' }
        ]
      },
      field: 'tool_calls'
    }
  ]

  it.each(presentationDeltas)('changes when only $field changes', ({ delta }) => {
    const base = [msg('assistant', 'Original answer.')] as Parameters<typeof sessionMessagesSignature>[0]
    const corrected = [{ ...base[0], ...delta }] as Parameters<typeof sessionMessagesSignature>[0]

    expect(sessionMessagesSignature(base)).not.toBe(sessionMessagesSignature(corrected))
  })

  it('keeps an absent display_content apart from an explicit null', () => {
    const absent = [msg('assistant', 'raw scaffolding')] as Parameters<typeof sessionMessagesSignature>[0]

    const explicitNull = [
      { ...absent[0], display_content: null }
    ] as Parameters<typeof sessionMessagesSignature>[0]

    expect(sessionMessagesSignature(absent)).not.toBe(sessionMessagesSignature(explicitNull))
  })

  it('ignores fields the transcript hydration never reads', () => {
    const base = [msg('assistant', 'answer')] as Parameters<typeof sessionMessagesSignature>[0]
    const extra = [{ ...base[0], some_future_field: 'not rendered' }] as unknown as typeof base

    expect(sessionMessagesSignature(base)).toBe(sessionMessagesSignature(extra))
  })
})
