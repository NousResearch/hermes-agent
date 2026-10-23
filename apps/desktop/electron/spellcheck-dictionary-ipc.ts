// Spell-check dictionary IPC: the custom context menu's native spell-check verbs
// and the plugin SDK's `host.spellcheck` door. Every handler acts on the
// SENDER's own session (or the named webview guest's), one word per call.

import type { IpcMain, IpcMainInvokeEvent, Session, WebContents } from 'electron'

const WORD_MAX = 64
// One word: letters, combining marks, digits; apostrophes and hyphens only inside.
const WORD_RE = /^[\p{L}\p{M}\p{N}](?:[\p{L}\p{M}\p{N}'\u2019-]*[\p{L}\p{M}\p{N}])?$/u
const LETTER_RE = /\p{L}/u

type DictionaryOp = (session: Session, word: string) => boolean

// A Map, not an object literal: `op` is renderer-supplied, and `'constructor'`
// or `'toString'` must not resolve to an inherited function.
const DICTIONARY_OPS = new Map<unknown, DictionaryOp>([
  ['add', (session, word) => session.addWordToSpellCheckerDictionary(word)],
  ['remove', (session, word) => session.removeWordFromSpellCheckerDictionary(word)]
])

/** The plugin door's word gate: exactly one bounded word with at least one
 *  letter, NFC-normalized, or null. No whitespace, controls, markup or paths
 *  ever reach Chromium's custom dictionary file through it. */
export function normalizeDictionaryWord(raw: unknown): null | string {
  if (typeof raw !== 'string') {
    return null
  }

  const word = raw.trim().normalize('NFC')

  return word.length <= WORD_MAX && WORD_RE.test(word) && LETTER_RE.test(word) ? word : null
}

export function registerSpellcheckDictionaryIpc(
  ipcMain: Pick<IpcMain, 'handle'>,
  contents: { fromId: (id: number) => undefined | WebContents }
): void {
  ipcMain.handle('hermes:context-menu:spellcheck', (event: IpcMainInvokeEvent, action) => {
    const kind = action?.kind
    const word = String(action?.word || '')

    if (!word) {
      return
    }

    if (kind === 'replace') {
      event.sender.replaceMisspelling(word)
    } else if (kind === 'add') {
      event.sender.session.addWordToSpellCheckerDictionary(word)
    }
  })

  // Guest dictionary add: the webview TAG exposes replaceMisspelling but no
  // session API, so the renderer names the guest by webContents id.
  ipcMain.handle('hermes:context-menu:guest-add-word', (_event: IpcMainInvokeEvent, payload) => {
    const word = String(payload?.word || '')
    const guest = contents.fromId(Number(payload?.webContentsId))

    if (word && guest && !guest.isDestroyed()) {
      guest.session.addWordToSpellCheckerDictionary(word)
    }
  })

  // `host.spellcheck.addWord/removeWord`. Resolves Chromium's own verdict
  // (false for an in-memory session), false for anything refused here.
  ipcMain.handle('hermes:spellcheck:dictionary', (event: IpcMainInvokeEvent, payload): boolean => {
    const op = DICTIONARY_OPS.get(payload?.op)
    const word = normalizeDictionaryWord(payload?.word)

    return Boolean(op && word && op(event.sender.session, word))
  })
}
