import { describe, expect, it } from 'vitest'

import baselineData from './locale-key-coverage.baseline.json'
import { collectKeyPaths, missingKeyPaths, parseObjectLiteralForTest } from './locale-key-coverage.util'

/**
 * Locale key coverage guard (exact-baseline ratchet).
 *
 * `defineLocale()` deep-merges a locale's overrides onto `en`, so a key a
 * locale never defines silently falls back to English at runtime. That is
 * the intended behavior for partial translations - see #65425 - but it
 * also means catalog drift accumulates invisibly: nothing fails when
 * `en.ts` gains keys a locale never picks up.
 *
 * This proposes a narrower blocking policy than strict parity: ar, ru,
 * ja, zh, and zh-hant all have real pre-existing gaps, recorded exactly
 * (as key-path sets, not counts) in locale-key-coverage.baseline.json.
 * The baseline is intentionally generated data, not a hand-maintained
 * expectation: it exists because current locales already carry
 * substantial accepted debt, and only an exact key-wise record can tell
 * an existing gap apart from a new regression. Requiring zero missing
 * keys today would fail CI on unrelated PRs. Related to #65425; this
 * does not attempt to close that broader localization-policy discussion.
 *
 * The baseline must equal the current missing set exactly, checked in
 * both directions, because a one-directional (subset) check alone is
 * not monotonic: if a locale's gap is translated without regenerating
 * the baseline, the baseline goes stale, and the *same key regressing
 * again later* would pass a subset check (it is still a member of the
 * now-stale baseline) even though real coverage went backwards. Checking
 * both directions turns that stale baseline into a required
 * generator run instead of a silent gap:
 * - newlyMissing (current - baseline): a real regression - translate it.
 * - resolvedSinceBaseline (baseline - current): coverage improved but
 *   the baseline snapshot wasn't updated - run
 *   generate-locale-key-coverage-baseline.ts and commit the result.
 *
 * Baseline growth should be reviewed as a coverage regression in the
 * PR diff; normal translation work should only shrink it.
 *
 * Companion to locale-parity.test.ts (#71233), which guards the opposite
 * direction: a key a locale defines that English no longer has. That
 * guard intentionally does not flag missing keys, and this one
 * intentionally does not flag extra keys - together they cover both
 * directions of drift.
 *
 * Follows the Vitest/TypeScript-native direction requested during review
 * of #66759 (a Python/CLI checker for the same problem) rather than
 * replacing it - #66759 still covers the CLI/gateway YAML catalogs this
 * guard does not touch.
 */

const baseline: Record<string, string[]> = baselineData

describe('locale key coverage', () => {
  it.each(Object.entries(baseline))('%s has no newly missing keys beyond its recorded baseline', (locale, recorded) => {
    const recordedSet = new Set(recorded)
    const newlyMissing = missingKeyPaths(locale).filter(key => !recordedSet.has(key))

    expect(
      newlyMissing,
      newlyMissing.length
        ? `${locale} has ${newlyMissing.length} newly missing key(s) not in the recorded baseline - ` +
            `this is a real regression. New English keys need a ${locale} translation:\n  ${newlyMissing.join('\n  ')}`
        : ''
    ).toEqual([])
  })

  it.each(Object.entries(baseline))('%s baseline is not stale (no keys resolved since it was recorded)', (locale, recorded) => {
    const currentSet = new Set(missingKeyPaths(locale))
    const resolvedSinceBaseline = recorded.filter(key => !currentSet.has(key))

    expect(
      resolvedSinceBaseline,
      resolvedSinceBaseline.length
        ? `${locale}'s baseline records ${resolvedSinceBaseline.length} key(s) that are no longer missing - ` +
            `coverage improved but the baseline snapshot is stale. Run ` +
            `generate-locale-key-coverage-baseline.ts and commit the result:\n  ${resolvedSinceBaseline.join('\n  ')}`
        : ''
    ).toEqual([])
  })
})

describe('collectKeyPaths', () => {
  it('descends into a non-empty nested object literal', () => {
    const obj = parseObjectLiteralForTest("export const x = { a: { b: 'c' } }")
    expect(collectKeyPaths(obj)).toEqual(['a.b'])
  })

  it('contributes zero leaves for an empty object literal', () => {
    const obj = parseObjectLiteralForTest("export const x = { a: {}, b: 'c' }")
    expect(collectKeyPaths(obj)).toEqual(['b'])
  })

  it('treats a dotted string-literal key as one leaf, dot-joined like nesting', () => {
    const obj = parseObjectLiteralForTest("export const x = { a: { 'b.c': 'd' } }")
    expect(collectKeyPaths(obj)).toEqual(['a.b.c'])
  })

  it('unwraps a defineFieldCopy(...) call to its argument, at full granularity', () => {
    const obj = parseObjectLiteralForTest("export const x = { a: defineFieldCopy({ b: 'c', d: { e: 'f' } }) }")
    expect(collectKeyPaths(obj).sort()).toEqual(['a.b', 'a.d.e'])
  })

  it('peels an `as T` wrapper before classifying the value', () => {
    const obj = parseObjectLiteralForTest("export const x = { a: { b: 'c' } as Record<string, string> }")
    expect(collectKeyPaths(obj)).toEqual(['a.b'])
  })

  it('peels `as T` around a defineFieldCopy(...) call too', () => {
    const obj = parseObjectLiteralForTest("export const x = { a: defineFieldCopy({ b: 'c' }) as Record<string, string> }")
    expect(collectKeyPaths(obj)).toEqual(['a.b'])
  })

  it('treats arrow functions, arrays, numbers, and booleans as leaves', () => {
    const obj = parseObjectLiteralForTest("export const x = { a: (n) => `${n}`, b: [1, 2], c: 3, d: true }")
    expect(collectKeyPaths(obj).sort()).toEqual(['a', 'b', 'c', 'd'])
  })

  it('throws on a spread member instead of silently skipping it', () => {
    const obj = parseObjectLiteralForTest("export const x = { ...shared, a: 'b' }")
    expect(() => collectKeyPaths(obj)).toThrow(/unsupported object member/)
  })

  it('throws on a computed property key instead of silently skipping it', () => {
    const obj = parseObjectLiteralForTest("export const x = { [dynamicKey]: 'b' }")
    expect(() => collectKeyPaths(obj)).toThrow(/unsupported property name/)
  })

  it('throws on an unrecognized bare identifier value', () => {
    const obj = parseObjectLiteralForTest("export const x = { a: SOME_UNKNOWN_CONSTANT }")
    expect(() => collectKeyPaths(obj)).toThrow(/unsupported identifier reference/)
  })

  it('throws on an unrecognized call expression value', () => {
    const obj = parseObjectLiteralForTest("export const x = { a: buildSection({ b: 'c' }) }")
    expect(() => collectKeyPaths(obj)).toThrow(/unsupported value shape/)
  })
})
