// `host.spellcheck`: Chromium's custom spell-check dictionary, one word at a
// time. The word gate and the session call live in main
// (`electron/spellcheck-dictionary-ipc.ts`); this side only forwards and turns
// every missing-capability case into `false`.

async function dictionary(op: 'add' | 'remove', word: string): Promise<boolean> {
  const bridge = typeof window === 'undefined' ? undefined : window.hermesDesktop

  if (typeof word !== 'string' || !bridge?.spellcheckDictionary) {
    return false
  }

  try {
    return (await bridge.spellcheckDictionary(op, word)) === true
  } catch {
    // A renderer bundle can outlive an older shell with no handler.
    return false
  }
}

export const spellcheckHost = {
  /** Teach the app's spell checker one word (letters, marks, digits, inner
   *  `'`/`-`; ≤ 64 chars) so Chromium stops underlining it everywhere.
   *  Resolves false when main refuses the word or the shell can't. */
  addWord: (word: string): Promise<boolean> => dictionary('add', word),
  /** Forget one word added earlier (same gate). */
  removeWord: (word: string): Promise<boolean> => dictionary('remove', word)
}
