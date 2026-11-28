import { describe, expect, it, vi } from 'vitest'

import { macSpellGuesses, parseGuesses, spellSuggestionsFor } from './spell-suggestions'

describe('spellSuggestionsFor', () => {
  it("keeps Chromium's own suggestions on every platform", async () => {
    const guess = vi.fn(async () => ['nope'])

    for (const platform of ['darwin', 'win32', 'linux'] as const) {
      expect(await spellSuggestionsFor('teh', ['the', 'ten'], platform, guess)).toEqual(['the', 'ten'])
    }

    expect(guess).not.toHaveBeenCalled()
  })

  it('asks the native spellchecker only on macOS when Chromium sent none', async () => {
    const guess = vi.fn(async () => ['the'])

    expect(await spellSuggestionsFor('teh', [], 'darwin', guess)).toEqual(['the'])
    expect(await spellSuggestionsFor('teh', [], 'win32', guess)).toEqual([])
    expect(await spellSuggestionsFor('teh', [], 'linux', guess)).toEqual([])
    expect(guess).toHaveBeenCalledTimes(1)
  })
})

describe('macSpellGuesses', () => {
  it('passes the word as an argument, never inside the script', async () => {
    const run = vi.fn(
      (
        _file: string,
        _args: string[],
        _options: { timeout: number },
        callback: (e: Error | null, out: string) => void
      ) => callback(null, '["the"]')
    )

    const word = '"); doEvil("'

    expect(await macSpellGuesses(word, run)).toEqual(['the'])
    const [file, args] = run.mock.calls[0]!

    expect(file).toBe('/usr/bin/osascript')
    expect(args.at(-1)).toBe(word)
    expect(args.slice(0, -1).join(' ')).not.toContain(word)
  })

  it('answers no suggestions when the runner fails or times out', async () => {
    const run = vi.fn(
      (_f: string, _a: string[], _o: { timeout: number }, callback: (e: Error | null, out: string) => void) =>
        callback(new Error('timed out'), '')
    )

    expect(await macSpellGuesses('teh', run)).toEqual([])
  })
})

describe('parseGuesses', () => {
  it('keeps at most five non-empty strings', () => {
    expect(parseGuesses('["a","b","c","d","e","f"]')).toEqual(['a', 'b', 'c', 'd', 'e'])
    expect(parseGuesses('["the", 3, null, ""]')).toEqual(['the'])
  })

  it('treats malformed output as no suggestions', () => {
    expect(parseGuesses('not json')).toEqual([])
    expect(parseGuesses('{"a":1}')).toEqual([])
  })
})
