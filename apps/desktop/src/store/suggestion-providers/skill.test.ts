import { describe, expect, it } from 'vitest'

import type { ChatMessage } from '@/lib/chat-messages'

import {
  collidesWithWorkspace,
  foldArabic,
  skillHit,
  skillPattern,
  skillTouchedInMessages,
  stripSlashTokens,
  triggerPattern,
  triggerPhrases
} from './skill'

// skillHit is the provider's real predicate: a whole-word match that the user
// has finished typing (at least one character follows it).
const hits = (name: string, draft: string) => skillHit(skillPattern(name), draft.toLowerCase())

// The same predicate over a declared trigger phrase, which is what lets a skill be
// offered to a user who types their own language instead of the skill's name. It runs on the
// provider's REAL haystack (slash tokens stripped, then lowercased) — a helper that skipped the
// sanitizer would pass a trigger the provider can never fire, e.g. one beginning with "/".
const hitsTrigger = (phrase: string, draft: string) =>
  skillHit(triggerPattern(phrase), foldArabic(stripSlashTokens(draft).toLowerCase()))

// Exercises the provider's real haystack sanitizer, not a copy of it.
const hitsAfterSanitize = (name: string, draft: string) =>
  skillHit(skillPattern(name), foldArabic(stripSlashTokens(draft).toLowerCase()))

describe('skillPattern + skillHit', () => {
  it('matches the exact name as a completed whole word', () => {
    expect(hits('perf', 'run the perf loop')).toBe(true)
    expect(hits('perf', 'performance is bad')).toBe(false)
  })

  it('hyphenated names also match spaced phrasing', () => {
    expect(hits('pr-ready', 'make this pr ready pls')).toBe(true)
    expect(hits('pr-ready', 'run pr-ready on it')).toBe(true)
    expect(hits('pr_ready', 'pr ready please')).toBe(true)
  })

  it('never matches inside other words', () => {
    expect(hits('read', 'i already did')).toBe(false)
    expect(hits('work', 'reworked the layout')).toBe(false)
    // Suffix boundary includes hyphen: "clean" must not fire inside "clean-up"
    // (a different skill may own that name).
    expect(hits('clean', 'do a clean-up pass')).toBe(false)
  })

  it('is case-insensitive via lowercased input', () => {
    expect(hits('Clean', 'please clean this diff')).toBe(true)
  })

  it('a name still under the caret is not a hit yet', () => {
    // The debounce fires while the word is the last thing typed — wait for
    // the next keystroke to call it intent.
    expect(hits('perf', 'run perf')).toBe(false)
    expect(hits('perf', 'run perf ')).toBe(true)
    expect(hits('perf', 'run perf.')).toBe(true)
  })

  it('an earlier completed occurrence still counts', () => {
    expect(hits('perf', 'perf first, then more perf')).toBe(true)
  })
})

describe('collidesWithWorkspace', () => {
  it('suppresses a skill named exactly like the cwd folder', () => {
    expect(collidesWithWorkspace('hermes-agent', '/Users/b/www/hermes-agent')).toBe(true)
  })

  it('suppresses inside worktree-suffixed folders too', () => {
    expect(collidesWithWorkspace('hermes-agent', '/Users/b/www/hermes-agent-suggest')).toBe(true)
  })

  it('does not suppress on substring-only overlap', () => {
    // "perf" inside "perfect-app" is not a homonym of the project.
    expect(collidesWithWorkspace('perf', '/Users/b/www/perfect-app')).toBe(false)
    expect(collidesWithWorkspace('clean', '/Users/b/www/hermes-agent')).toBe(false)
  })

  it('never collides when detached (empty cwd)', () => {
    expect(collidesWithWorkspace('hermes-agent', '')).toBe(false)
  })
})

// -- skillTouchedInMessages ---------------------------------------------------

const toolCall = (toolName: string, args?: unknown, argsText = ''): ChatMessage => ({
  id: 't',
  role: 'assistant',
  parts: [{ args: args as never, argsText, toolCallId: 'x', toolName, type: 'tool-call' }]
})

const userText = (text: string): ChatMessage => ({
  id: 'u',
  role: 'user',
  parts: [{ text, type: 'text' }]
})

describe('skillTouchedInMessages', () => {
  it('detects a skill_view load of the skill', () => {
    expect(skillTouchedInMessages('pr-ready', [toolCall('skill_view', { name: 'pr-ready' })])).toBe(true)
  })

  it('detects a skill_manage touch of the skill', () => {
    expect(skillTouchedInMessages('pr-ready', [toolCall('skill_manage', { action: 'patch', name: 'pr-ready' })])).toBe(
      true
    )
  })

  it('matches qualified skill names (category/name, plugin:name)', () => {
    expect(
      skillTouchedInMessages('hermes-agent-dev', [toolCall('skill_view', { name: 'github/hermes-agent-dev' })])
    ).toBe(true)
    expect(
      skillTouchedInMessages('writing-plans', [toolCall('skill_view', { name: 'superpowers:writing-plans' })])
    ).toBe(true)
  })

  it('falls back to argsText when args were not parsed', () => {
    expect(skillTouchedInMessages('pr-ready', [toolCall('skill_view', undefined, '{"name":"pr-ready"}')])).toBe(true)
  })

  it('detects the user loading the skill via its slash command', () => {
    expect(skillTouchedInMessages('pr-ready', [userText('/pr-ready check this branch')])).toBe(true)
    expect(skillTouchedInMessages('pr-ready', [userText('/pr-ready')])).toBe(true)
  })

  it('ignores touches of OTHER skills and non-skill tools', () => {
    expect(skillTouchedInMessages('pr-ready', [toolCall('skill_view', { name: 'clean' })])).toBe(false)
    expect(skillTouchedInMessages('pr-ready', [toolCall('read_file', { path: 'pr-ready' })])).toBe(false)
    // Slash prefix must be exact — /pr-ready-extra is a different command.
    expect(skillTouchedInMessages('pr-ready', [userText('/pr-ready-extra go')])).toBe(false)
    // Merely mentioning the name in prose is not a load.
    expect(skillTouchedInMessages('pr-ready', [userText('is pr-ready any good?')])).toBe(false)
  })

  it('is case-insensitive on the stored arg', () => {
    expect(skillTouchedInMessages('pr-ready', [toolCall('skill_view', { name: 'PR-Ready' })])).toBe(true)
  })

  it('empty transcript touches nothing', () => {
    expect(skillTouchedInMessages('pr-ready', [])).toBe(false)
  })
})

describe('draft haystack sanitization (slash-prefixed skills, #91626)', () => {
  it('strips slash commands mid-message before scanning', () => {
    // The bug: "please run /github-auth on this" re-fired the "Use skill:
    // github-auth" pill because skillHit matched "github-auth" inside the
    // slash command the user had already typed.
    expect(hitsAfterSanitize('github-auth', 'please run /github-auth on this')).toBe(false)
    expect(hitsAfterSanitize('vault', 'When I trigger /vault and /github-auth please')).toBe(false)
  })

  it('preserves URL fragments mid-prose (no leading whitespace)', () => {
    // https://example.com/api/v1 is not stripped: the regex only removes
    // whitespace-bounded slash tokens.
    expect(hitsAfterSanitize('api', 'visit https://example.com/api/v1 for api docs ')).toBe(true)
  })

  it('a real prose mention after a slash command still hits', () => {
    // Stripping the slash token must not blind the provider to a genuine
    // prose mention elsewhere in the draft.
    expect(hitsAfterSanitize('github-auth', '/github-auth loaded. Is github-auth any good? ')).toBe(true)
  })
})

describe('triggerPattern — declared phrases reach a skill by the words the user types', () => {
  it('matches a phrase in another language, which the name alone never could', () => {
    // The case that motivated this: an English-named skill and a draft written in
    // Arabic. `pdf-editing` cannot be typed as a name by a user who writes "أدمج ملفين".
    expect(hitsTrigger('دمج ملفين', 'دمج ملفين pdf وأرسلهم للعميل')).toBe(true)
    expect(hitsTrigger('ملف pdf', 'رتب لي ملف pdf هذا فضلاً')).toBe(true)
    // The name still misses the same draft — that is the gap the triggers close.
    expect(hits('pdf-editing', 'دمج ملفين pdf وأرسلهم للعميل')).toBe(false)
  })

  it('keeps the completed-word rule: a phrase still being typed does not fire', () => {
    expect(hitsTrigger('دمج ملفين', 'دمج ملفين')).toBe(false)
    expect(hitsTrigger('دمج ملفين', 'دمج ملفين ')).toBe(true)
  })

  it('requires a word boundary, so a phrase inside a longer word does not fire', () => {
    expect(hitsTrigger('دمج', 'أدمج الملفين')).toBe(false) // "أدمج" is one word
    expect(hitsTrigger('دمج', 'دمج الملفين')).toBe(true)
    expect(hitsTrigger('ملفات', 'ملفاته كثيرة هنا')).toBe(false) // a letter follows
    expect(hitsTrigger('ملفات', 'ملفات كثيرة هنا')).toBe(true)
  })

  it('treats whitespace inside a phrase as flexible', () => {
    expect(hitsTrigger('دمج   ملفين', 'دمج ملفين pdf')).toBe(true)
    expect(hitsTrigger('دمج ملفين', 'دمج    ملفين pdf')).toBe(true)
  })

  it('escapes the phrase, so punctuation is literal', () => {
    expect(hitsTrigger('a+b', 'type a+b here')).toBe(true)
    expect(hitsTrigger('a+b', 'type aab here')).toBe(false)
    expect(hitsTrigger('c++', 'use c++ now')).toBe(true)
  })

  it('an empty or whitespace phrase never matches (a blank entry is not a wildcard)', () => {
    expect(hitsTrigger('', 'anything at all')).toBe(false)
    expect(hitsTrigger('   ', 'anything at all')).toBe(false)
  })

  it('is case-insensitive on the Latin part and caller-lowercased on the rest', () => {
    expect(hitsTrigger('File PDF', 'send the file pdf now')).toBe(true)
    expect(hitsTrigger('file pdf', 'SEND THE FILE PDF NOW')).toBe(true)
  })

  it('a slash-leading trigger fires only where the draft itself starts with it', () => {
    // Measured, not assumed. stripSlashTokens removes whitespace-PREFIXED "/token" (which is why
    // "/api/v1" survives mid-prose and "run /github-auth" does not), so a phrase starting with "/"
    // has exactly one live position: the very start of the draft. It is unreliable rather than dead —
    // it fires on "/pdf please" and not on "use /pdf now".
    expect(hitsTrigger('/pdf', '/pdf please')).toBe(true)
    expect(hitsTrigger('/pdf', 'use /pdf now')).toBe(false)
  })
})

describe('triggerPhrases — the payload contract, defended', () => {
  it('accepts a list, trims it, and drops blanks', () => {
    expect(triggerPhrases(['  دمج ملفين  ', '', '   ', 'file pdf'])).toEqual(['دمج ملفين', 'file pdf'])
  })

  it('drops non-string entries instead of stringifying them', () => {
    // `5` becoming the phrase "5" is a phrase nobody wrote — the backend refuses scalars for the
    // same reason, and this is the second guard.
    expect(triggerPhrases([5, null, undefined, true, { a: 1 }, ['x'], 'kept'])).toEqual(['kept'])
  })

  it('a non-array payload yields no phrases instead of throwing', () => {
    // A string has `.length` (so it would pass a length gate) but no `.map`: this is the shape that
    // would throw inside the composer's render path, from an older or malformed backend.
    expect(triggerPhrases('short, phrases')).toEqual([])
    expect(triggerPhrases(undefined)).toEqual([])
    expect(triggerPhrases(null)).toEqual([])
    expect(triggerPhrases(7)).toEqual([])
  })

  it('caps the list', () => {
    expect(triggerPhrases(Array.from({ length: 40 }, (_, i) => `p${i}`))).toHaveLength(12)
  })
})

describe('foldArabic — the spelling variants Arabic writers type interchangeably', () => {
  // Every draft below carries a trailing word on purpose: skillHit only fires on a COMPLETED word
  // (at least one character after the match), which is the pill's real contract — the user is mid-draft,
  // not typing a bare phrase. A fixture that ended AT the phrase would assert false for the wrong reason,
  // and would have hidden a fold that wrongly merges two words.
  // The reported case, from the owner's own machine: a skill declaring `تدقيق الأرقام` was offered when
  // the draft carried the hamza and missed when it did not. Both spellings are the same phrase to a
  // reader, so they must be the same phrase to the matcher.
  it('folds every alef variant, in both directions', () => {
    for (const variant of ['أ', 'إ', 'آ', 'ٱ']) {
      expect(hitsTrigger(`تدقيق ${variant}رقام`, 'تدقيق ارقام الآن')).toBe(true)
      expect(hitsTrigger('تدقيق ارقام', `تدقيق ${variant}رقام الآن`)).toBe(true)
    }
  })

  it('drops diacritics and the tatweel', () => {
    expect(hitsTrigger('كشف حساب', 'كَشْف حِسَاب الآن')).toBe(true)
    expect(hitsTrigger('كشف حساب', 'كشــــــف حساب الآن')).toBe(true)
  })

  it('folds a decomposed spelling the same as a composed one', () => {
    const composed = 'الأرقام'
    const decomposed = composed.normalize('NFD')
    expect(decomposed).not.toBe(composed)
    expect(foldArabic(decomposed)).toBe(foldArabic(composed))
    expect(hitsTrigger(composed, `${decomposed} الآن`)).toBe(true)
  })

  // The defect this test was written to catch: folding marks before composing turns the DECOMPOSED
  // رؤية into روية, so the same words would merge in one encoding and stay apart in the other.
  it('keeps a hamza-on-waw distinct in both encodings', () => {
    const composed = 'رؤية'
    const decomposed = composed.normalize('NFD')
    expect(foldArabic(decomposed)).toBe(foldArabic(composed))
    expect(hitsTrigger(composed, `${decomposed} الآن`)).toBe(true) // same word, either spelling
    expect(hitsTrigger('روية', `${decomposed} الآن`)).toBe(false) // and never the other one
    expect(hitsTrigger('روية', `${composed} الآن`)).toBe(false)
  })

  it('is idempotent, and leaves blank input alone', () => {
    const once = foldArabic('الأرقامُ')
    expect(foldArabic(once)).toBe(once)
    expect(foldArabic('')).toBe('')
    expect(foldArabic('   ')).toBe('   ')
  })

  // The exclusions asserted, not merely documented: an unbounded fold is exactly how this feature
  // would start offering skills on unrelated prose, which is the failure the whole feature is at risk of.
  it('does NOT fold what would merge two different words', () => {
    expect(hitsTrigger('كتابة', 'كتابه الآن')).toBe(false) // ة / ه
    expect(hitsTrigger('مصطفى', 'مصطفي الآن')).toBe(false) // ى / ي
    expect(hitsTrigger('رؤية', 'روية الآن')).toBe(false) // ؤ / و
    expect(hitsTrigger('مسؤول', 'مسوول الآن')).toBe(false) // ؤ / و
  })

  it('keeps the word boundaries the pattern was built on', () => {
    expect(hitsTrigger('الأرقام', 'بالأرقام الآن')).toBe(false) // no prefix
    expect(hitsTrigger('الأرقام', 'الأرقامية الآن')).toBe(false) // no suffix
    expect(hitsTrigger('الأرقام', 'تدقيق الأرقام الآن')).toBe(true)
  })

  it('leaves Latin matching exactly as it was', () => {
    expect(hitsTrigger('file pdf', 'SEND THE FILE PDF NOW')).toBe(true)
    expect(hitsTrigger('file pdf', 'files pdf now')).toBe(false)
  })
})
