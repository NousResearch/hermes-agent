// Spelling suggestions for the custom context menu's spell-check forward.
//
// Chromium's `context-menu` params carry `dictionarySuggestions`. On Windows and
// Linux they come from Hunspell and are filled in. On macOS Electron uses the
// native spellchecker: the misspelled word is reported, but the suggestion list
// arrives empty (observed on Electron 44), so the menu offered "Add to
// dictionary" with nothing to replace the word with. There we ask AppKit's
// NSSpellChecker — the same engine that underlined the word — through the
// system's JavaScript-for-Automation runner (~90 ms, no native module).

import { execFile } from 'node:child_process'

const MAX_SUGGESTIONS = 5
const GUESS_TIMEOUT_MS = 2000
const OSASCRIPT = '/usr/bin/osascript'

// The word arrives as argv, never interpolated into the script.
const MAC_SPELL_GUESSES_JXA = `ObjC.import("AppKit");
function run(argv) {
  const w = argv[0] || "";
  const g = $.NSSpellChecker.sharedSpellChecker.guessesForWordRangeInStringLanguageInSpellDocumentWithTag($.NSMakeRange(0, w.length), w, $(), 0);
  return JSON.stringify(ObjC.deepUnwrap(g) || []);
}`

type ExecFile = (
  file: string,
  args: string[],
  options: { timeout: number },
  callback: (error: Error | null, stdout: string | Buffer) => void
) => unknown

/** Parse the runner's JSON answer; anything unexpected is "no suggestions". */
export function parseGuesses(stdout: string): string[] {
  try {
    const parsed: unknown = JSON.parse(stdout.trim())

    return Array.isArray(parsed)
      ? parsed
          .filter((guess): guess is string => typeof guess === 'string' && guess.length > 0)
          .slice(0, MAX_SUGGESTIONS)
      : []
  } catch {
    return []
  }
}

/** NSSpellChecker's guesses for `word`; never rejects (errors and timeouts give []). */
export function macSpellGuesses(word: string, run: ExecFile = execFile as unknown as ExecFile): Promise<string[]> {
  return new Promise(resolve => {
    run(
      OSASCRIPT,
      ['-l', 'JavaScript', '-e', MAC_SPELL_GUESSES_JXA, word],
      { timeout: GUESS_TIMEOUT_MS },
      (error, stdout) => resolve(error ? [] : parseGuesses(String(stdout)))
    )
  })
}

/** The suggestions to forward: Chromium's own when it has any, otherwise (macOS only) the native guesses. */
export async function spellSuggestionsFor(
  word: string,
  reported: readonly string[],
  platform: NodeJS.Platform = process.platform,
  guess: (word: string) => Promise<string[]> = macSpellGuesses
): Promise<string[]> {
  if (reported.length > 0 || platform !== 'darwin' || !word) {
    return [...reported]
  }

  return guess(word)
}
