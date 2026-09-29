import { beforeEach, describe, expect, it } from 'vitest'

import {
  type ComposerFollowUp,
  createComposerFollowUpScope,
  FOLLOW_UP_PASSAGE_MAX_CHARS,
  FOLLOW_UP_PREVIEW_MAX_CHARS,
  followUpBlockFromQuote,
  followUpPreview,
  mainComposerFollowUpScope,
  normalizeFollowUpPassage
} from './composer'

const quote = (passage: string, source: ComposerFollowUp['source'] = 'assistant'): ComposerFollowUp => ({
  passage,
  source
})

describe('normalizeFollowUpPassage', () => {
  it('trims the edges and per-line trailing whitespace', () => {
    expect(normalizeFollowUpPassage('  a line  \n b line\t\n  ')).toBe('a line\n b line')
  })

  it('collapses runs of blank lines but keeps a paragraph break', () => {
    expect(normalizeFollowUpPassage('one\n\n\n\n\ntwo')).toBe('one\n\ntwo')
  })

  it('normalizes CRLF', () => {
    expect(normalizeFollowUpPassage('one\r\ntwo')).toBe('one\ntwo')
  })

  it('caps a runaway passage and marks the cut', () => {
    const capped = normalizeFollowUpPassage('x'.repeat(FOLLOW_UP_PASSAGE_MAX_CHARS + 50))

    expect(capped).toHaveLength(FOLLOW_UP_PASSAGE_MAX_CHARS + 1)
    expect(capped.endsWith('…')).toBe(true)
  })

  it('is idempotent', () => {
    const once = normalizeFollowUpPassage('  one\n\n\n\ntwo  ')

    expect(normalizeFollowUpPassage(once)).toBe(once)
  })
})

describe('followUpBlockFromQuote', () => {
  it('is empty without a quote', () => {
    expect(followUpBlockFromQuote(null)).toBe('')
    expect(followUpBlockFromQuote(undefined)).toBe('')
    expect(followUpBlockFromQuote(quote('   '))).toBe('')
  })

  it('quotes every line of a single-line passage', () => {
    expect(followUpBlockFromQuote(quote('the answer'))).toBe('> the answer')
  })

  it('keeps a multi-paragraph passage inside ONE blockquote', () => {
    // A blank line ends a markdown blockquote, so the blank keeps its marker:
    // without it the second paragraph would read as the reader's own words.
    expect(followUpBlockFromQuote(quote('para one\n\npara two'))).toBe('> para one\n>\n> para two')
  })
})

describe('followUpPreview', () => {
  it('shows a short passage whole', () => {
    expect(followUpPreview('a short quote')).toEqual({ text: 'a short quote', truncated: false })
  })

  it('marks the cut on a passage past the preview budget', () => {
    const preview = followUpPreview('x'.repeat(FOLLOW_UP_PREVIEW_MAX_CHARS + 120))

    expect(preview.truncated).toBe(true)
    expect(preview.text).toHaveLength(FOLLOW_UP_PREVIEW_MAX_CHARS + 1)
    expect(preview.text.endsWith('…')).toBe(true)
  })

  it('normalizes before measuring, so whitespace cannot eat the budget', () => {
    const preview = followUpPreview(`  ${'y'.repeat(FOLLOW_UP_PREVIEW_MAX_CHARS)}   `)

    expect(preview).toEqual({ text: 'y'.repeat(FOLLOW_UP_PREVIEW_MAX_CHARS), truncated: false })
  })
})

describe('createComposerFollowUpScope', () => {
  beforeEach(() => {
    mainComposerFollowUpScope.clear()
  })

  it('normalizes on capture and clears on an empty passage', () => {
    const scope = createComposerFollowUpScope()

    scope.set(quote('  spaced  '))
    expect(scope.$followUp.get()).toEqual(quote('spaced'))

    scope.set(quote('   \n '))
    expect(scope.$followUp.get()).toBeNull()
  })

  it('preserves identity when the same passage is picked twice', () => {
    const scope = createComposerFollowUpScope()

    scope.set(quote('same'))
    const first = scope.$followUp.get()

    scope.set(quote('same'))
    expect(scope.$followUp.get()).toBe(first)
  })

  it('replaces the passage when another side of the transcript is quoted', () => {
    const scope = createComposerFollowUpScope()

    scope.set(quote('same', 'assistant'))
    scope.set(quote('same', 'user'))

    expect(scope.$followUp.get()).toEqual(quote('same', 'user'))
  })

  it('keeps two scopes independent', () => {
    const tile = createComposerFollowUpScope()

    tile.set(quote('tile passage'))

    expect(tile.$followUp.get()).toEqual(quote('tile passage'))
    expect(mainComposerFollowUpScope.$followUp.get()).toBeNull()
  })

  it('clear is a no-op without a pending quote', () => {
    const scope = createComposerFollowUpScope()
    let notifications = 0

    scope.$followUp.listen(() => notifications++)
    scope.clear()

    expect(notifications).toBe(0)
  })
})
