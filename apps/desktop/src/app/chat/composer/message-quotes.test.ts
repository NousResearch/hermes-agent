import { describe, expect, it, vi } from 'vitest'

import { formatRefValue } from '@/components/assistant-ui/directive-text'
import {
  $composerMessageQuotes,
  clearComposerMessageQuotes,
  freezeComposerTransportPayload,
  messageQuoteContextBlocks,
  reconcileComposerMessageQuotes,
  setComposerMessageQuote,
} from '@/store/composer'

vi.mock('@/app/chat/composer/focus', () => ({
  requestComposerInsert: vi.fn(),
}))

const { requestComposerInsert } = await import('@/app/chat/composer/focus')

const { addMessageSelectionToChat } = await import(
  '@/app/chat/composer/selection-composer-bridge'
)

describe('message quote context blocks', () => {
  it('resolves an @message:<id> ref to a quote code block', () => {
    setComposerMessageQuote('msg-abc', 'Hello from the assistant')

    const blocks = messageQuoteContextBlocks('Can you address @message:msg-abc?')

    expect(blocks).toEqual([
      '```quote<msg-abc>\nHello from the assistant\n```',
    ])

    clearComposerMessageQuotes()
  })

  it('returns empty array when draft has no @message refs', () => {
    setComposerMessageQuote('msg-xyz', 'Some text')

    expect(messageQuoteContextBlocks('Just a regular message')).toEqual([])

    clearComposerMessageQuotes()
  })

  it('returns empty array when the referenced id has no stored quote', () => {
    // Set a quote for a different id
    setComposerMessageQuote('msg-stored', 'Stored text')

    // Reference a different id that was never stored
    expect(messageQuoteContextBlocks('Check @message:msg-missing')).toEqual([])

    clearComposerMessageQuotes()
  })

  it('handles multiple distinct @message refs in one draft', () => {
    setComposerMessageQuote('msg-1', 'First quote')
    setComposerMessageQuote('msg-2', 'Second quote')

    const blocks = messageQuoteContextBlocks('@message:msg-1 and @message:msg-2')

    expect(blocks).toEqual([
      '```quote<msg-1>\nFirst quote\n```',
      '```quote<msg-2>\nSecond quote\n```',
    ])

    clearComposerMessageQuotes()
  })

  it('deduplicates repeated refs to the same message id', () => {
    setComposerMessageQuote('msg-dup', 'Single text')

    const blocks = messageQuoteContextBlocks(
      '@message:msg-dup and again @message:msg-dup'
    )

    expect(blocks).toHaveLength(1)
    expect(blocks[0]).toBe('```quote<msg-dup>\nSingle text\n```')

    clearComposerMessageQuotes()
  })

  it('handles quoted ref values (backtick-wrapped ids that need quoting)', () => {
    // IDs with special characters get wrapped by formatRefValue, and
    // messageIdsFromDraft strips the wrapping on the way back out.
    const id = 'msg:chaos-42'
    const formatted = formatRefValue(id)
    setComposerMessageQuote(id, 'Quoted id text')

    const blocks = messageQuoteContextBlocks(`See @message:${formatted}`)

    expect(blocks).toEqual([
      '```quote<msg:chaos-42>\nQuoted id text\n```',
    ])

    clearComposerMessageQuotes()
  })

  it('clearComposerMessageQuotes empties the store', () => {
    setComposerMessageQuote('msg-clear', 'Will be removed')

    expect(messageQuoteContextBlocks('@message:msg-clear')).toHaveLength(1)

    clearComposerMessageQuotes()

    expect(messageQuoteContextBlocks('@message:msg-clear')).toEqual([])
  })

  it('reconcileComposerMessageQuotes drops entries not present in the draft', () => {
    setComposerMessageQuote('msg-keep', 'Kept text')
    setComposerMessageQuote('msg-drop', 'Dropped text')

    reconcileComposerMessageQuotes('@message:msg-keep')

    // msg-drop should be gone after reconcile
    expect($composerMessageQuotes.get()['msg-drop']).toBeUndefined()
    expect($composerMessageQuotes.get()['msg-keep']).toBe('Kept text')

    clearComposerMessageQuotes()
  })

  it('reconcileComposerMessageQuotes preserves entries still referenced in the draft', () => {
    setComposerMessageQuote('msg-1', 'First')
    setComposerMessageQuote('msg-2', 'Second')

    reconcileComposerMessageQuotes('@message:msg-1 and @message:msg-2')

    expect($composerMessageQuotes.get()['msg-1']).toBe('First')
    expect($composerMessageQuotes.get()['msg-2']).toBe('Second')

    clearComposerMessageQuotes()
  })

  it('produces empty output for an empty draft', () => {
    setComposerMessageQuote('msg-1', 'Text')

    expect(messageQuoteContextBlocks('')).toEqual([])
    expect(messageQuoteContextBlocks('   ')).toEqual([])

    clearComposerMessageQuotes()
  })
})

describe('addMessageSelectionToChat', () => {
  it('keys the store and the inserted ref by the same id so the quote resolves at submit', () => {
    // Regression pin: the inserted ref text must carry the STORE KEY (the
    // message id) in exactly the wire form the store lookup parses back —
    // any drift between the two resolves to nothing and the quote silently
    // vanishes at submit.
    const id = 'msg:chaos-42'
    addMessageSelectionToChat('Quoted body', id)

    expect(requestComposerInsert).toHaveBeenCalledWith(
      `@message:${formatRefValue(id)}`,
      { mode: 'inline' },
    )
    expect($composerMessageQuotes.get()[id]).toBe('Quoted body')

    // The inserted ref must round-trip through the store lookup.
    const inserted = `@message:${formatRefValue(id)}`
    expect(messageQuoteContextBlocks(inserted)).toEqual([
      '```quote<msg:chaos-42>\nQuoted body\n```',
    ])

    clearComposerMessageQuotes()
    vi.mocked(requestComposerInsert).mockClear()
  })
})

describe('freezeComposerTransportPayload with @message quotes', () => {
  it('freezes quote refs into fenced transport and keeps the ref for display', () => {
    setComposerMessageQuote('msg-abc', 'Hello from the assistant')

    const frozen = freezeComposerTransportPayload('Can you address @message:msg-abc?')

    expect(frozen.transportText).toBe(
      '```quote<msg-abc>\nHello from the assistant\n```\n\nCan you address @message:msg-abc?'
    )
    expect(frozen.displayText).toBe('Can you address @message:msg-abc?')
    expect(frozen.missingLabels).toEqual([])

    clearComposerMessageQuotes()
  })

  it('does not expand quotes a second time when transport is already frozen', () => {
    setComposerMessageQuote('msg-abc', 'Hello')

    const once = freezeComposerTransportPayload('see @message:msg-abc')
    const twice = freezeComposerTransportPayload(once.transportText)

    expect(twice.transportText).toBe(once.transportText)
    expect((once.transportText.match(/```quote/g) ?? []).length).toBe(1)

    clearComposerMessageQuotes()
  })

  it('frozen transport carries its quotes independent of the store', () => {
    // Queue-drain safety: the freeze happens at enqueue; a later store clear
    // (e.g. another submit landing before the drain) must not drop the quote.
    setComposerMessageQuote('msg-abc', 'Hello')

    const frozen = freezeComposerTransportPayload('see @message:msg-abc')
    clearComposerMessageQuotes()

    expect(frozen.transportText).toContain('```quote<msg-abc>\nHello\n```')
  })

  it('leaves unresolved quote refs visible instead of dropping them', () => {
    const frozen = freezeComposerTransportPayload('see @message:msg-unknown')

    expect(frozen.transportText).toBe('see @message:msg-unknown')
    expect(frozen.displayText).toBe('see @message:msg-unknown')
  })

  it('stays idempotent when the quoted text contains code fences', () => {
    // The fence dedupe must match by `quote<id>` header, not by a lazy whole-
    // block regex — an inner code fence in the quoted text would truncate the
    // match and the second freeze would prepend the block again.
    setComposerMessageQuote('msg-code', 'see:\n```ts\nconst x = 1\n```\ndone')

    const once = freezeComposerTransportPayload('re @message:msg-code')
    const twice = freezeComposerTransportPayload(once.transportText)

    expect((once.transportText.match(/```quote<msg-code>/g) ?? []).length).toBe(1)
    expect(twice.transportText).toBe(once.transportText)

    clearComposerMessageQuotes()
  })

  it('freezes and dedupes ids that contain angle brackets', () => {
    // The fence header ends at `>\n`, so `quote<msg>` must not match inside a
    // `quote<msg>1>` header in either direction.
    setComposerMessageQuote('msg>1', 'Brackets')

    const once = freezeComposerTransportPayload('a @message:`msg>1` and @message:msg')

    expect(once.transportText).toContain('```quote<msg>1>\nBrackets\n```')
    // msg has no store entry yet: its ref stays visible, unfenced.
    expect(once.transportText).not.toContain('```quote<msg>\n')

    // Later the store gains msg's quote: re-freezing must still add its fence
    // even though msg>1's header shares the ` ```quote<msg> ` prefix.
    setComposerMessageQuote('msg', 'Plain')
    const twice = freezeComposerTransportPayload(once.transportText)

    expect(twice.transportText).toContain('```quote<msg>\nPlain\n```')
    expect((twice.transportText.match(/```quote<msg>1>/g) ?? []).length).toBe(1)

    clearComposerMessageQuotes()
  })

  it('expands nothing when terminal chips fail closed (missingLabels)', () => {
    // Fail-closed invariant: every caller rejects a missingLabels payload
    // wholesale, so the quote pass must leave it byte-identical rather than
    // half-freezing it.
    setComposerMessageQuote('msg-abc', 'Hello')

    const frozen = freezeComposerTransportPayload(
      'see @message:msg-abc and @terminal:`zsh:23-58`'
    )

    expect(frozen.missingLabels).toEqual(['zsh:23-58'])
    expect(frozen.transportText).toBe('see @message:msg-abc and @terminal:`zsh:23-58`')
    expect(frozen.transportText).not.toContain('```quote')

    clearComposerMessageQuotes()
  })
})
