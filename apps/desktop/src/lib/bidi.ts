import { isValidElement, type ReactNode } from 'react'

/**
 * Majority-vote base direction for mixed Persian/English text.
 *
 * CSS `unicode-bidi: plaintext` and `dir="auto"` resolve a block from its
 * FIRST strong character only, so a Persian sentence that opens with an
 * English term ("Task Manager را باز کن…") renders as an LTR paragraph and
 * turns unreadable. These helpers vote over the whole string instead: the
 * dominant script wins, pure single-script text resolves exactly as before,
 * and neutral-only text (digits, punctuation, emoji) stays LTR.
 *
 * Tie-break favors RTL (`rtl >= ltr` with `rtl > 0`): real mixed sentences
 * land on ties, and an RTL base keeps the Persian majority readable while
 * the LTR terms still order correctly inside it.
 */

export type BidiDirection = 'rtl' | 'ltr'

// Hebrew + Arabic-script blocks: Hebrew/Arabic/Syriac/Thaana/NKo
// (U+0591-U+07FF), Arabic Extended-B/A (U+0870-U+08FF), Arabic presentation
// forms A/B (U+FB1D-U+FDFD, U+FE70-U+FEFC). ZWNJ/ZWJ (U+200C-U+200D) and
// combining marks are neutral and deliberately excluded — they never vote.
const RTL_PATTERN = /[֑-߿ࡰ-ࣿיִ-﷽ﹰ-ﻼ]/g
// Basic Latin + Latin extensions, Greek, Cyrillic, Armenian, Georgian.
const LTR_PATTERN = /[A-Za-zÀ-ɏͰ-ϿЀ-ӿ԰-֏ა-ჿ]/g

function countMatches(text: string, pattern: RegExp): number {
  // String.match with a /g pattern ignores lastIndex, so the shared
  // module-level patterns are safe to reuse across calls.
  return text.match(pattern)?.length ?? 0
}

// Arabic-block punctuation (U+060C ، U+061B ؛ U+061F ؟ U+061E ؞
// U+066A-U+066D ٪٫٬٭ U+06D4 ۔): weak neutrals under UAX#9, so they never
// vote — exactly like Latin punctuation, which is outside the LTR class.
// (Digits DO vote: a lone Persian "۱۹" reads naturally with an RTL base.)
const ARABIC_PUNCT_PATTERN = /[،؛؟؞٪٫٬٭۔]/g

export function voteDirection(text: string): BidiDirection {
  const rtl = countMatches(text, RTL_PATTERN) - countMatches(text, ARABIC_PUNCT_PATTERN)

  if (rtl <= 0) {
    return 'ltr'
  }

  return rtl >= countMatches(text, LTR_PATTERN) ? 'rtl' : 'ltr'
}

// Elements whose content must not vote in an ancestor's resolution — the
// browser already excludes them (they carry their own dir), so the vote
// mirrors that exclusion. Code is the load-bearing one: a paragraph like
// "`npm install` را اجرا کن" is Persian-majority once the command is out.
const NON_VOTING_ELEMENTS = new Set(['code', 'pre', 'samp', 'kbd'])

function collectVotingText(node: ReactNode): string {
  if (typeof node === 'string' || typeof node === 'number') {
    return String(node)
  }

  if (Array.isArray(node)) {
    return node.map(collectVotingText).join(' ')
  }

  if (isValidElement(node)) {
    const props = node.props as {
      children?: ReactNode
      'data-ref-text'?: unknown
      'data-slot'?: unknown
    }

    if (typeof node.type === 'string' && NON_VOTING_ELEMENTS.has(node.type)) {
      return ' '
    }

    // Reference chips (@file:, @url:, …) carry LTR paths and already sit
    // behind dir="ltr" — exclude them the same way.
    if (props['data-ref-text'] != null || props['data-slot'] === 'aui_directive-chip') {
      return ' '
    }

    return collectVotingText(props.children)
  }

  return ' '
}

/** Vote over rendered children, skipping code spans and ref chips. */
export function voteDirectionFromNodes(node: ReactNode): BidiDirection {
  return voteDirection(collectVotingText(node))
}
