import {
  type ComposerInsertMode,
  type ComposerTarget,
  requestComposerFocus,
  requestComposerGetDraft,
  requestComposerInsertAcked,
  requestComposerSetDraft,
  requestComposerSpellcheck,
  requestComposerSubmit
} from '@/app/chat/composer/focus'
import { freshDraftScope, takeSessionDraft } from '@/store/composer'
import { $activeSessionId, $selectedStoredSessionId } from '@/store/session'
import { $sessionStates } from '@/store/session-states'

/**
 * A plugin's session address resolved for the two composer buses. `target` is
 * the focus/insert/submit routing key (`null` = no mounted surface can own the
 * address, so those verbs fail closed); `ids` are the session ids a mounted
 * surface answers draft read/write requests for, and `stored` the durable id
 * the persisted stash is keyed by.
 *
 * `null`/empty = the active composer (bus-resolved, like the internal helpers).
 * `'new'` = the draft with no session yet — the primary composer while it shows
 * no session (a tile always has one); it never falls through to `'active'`,
 * which could be a tile holding another session. A stored id routes to that
 * session's tile when one is open, else the primary composer (which renders
 * that session); a runtime id is mapped to its stored id first.
 */
const resolveComposerAddress = (
  sessionId: null | string | undefined
): { ids: string[]; stored: string; target: 'active' | ComposerTarget | null } => {
  const id = typeof sessionId === 'string' ? sessionId.trim() : ''

  if (!id) {
    return { ids: [], stored: '', target: 'active' }
  }

  if (id === 'new') {
    const primaryIsNewDraft = !$activeSessionId.get() && !$selectedStoredSessionId.get()

    return { ids: [freshDraftScope()], stored: freshDraftScope(), target: primaryIsNewDraft ? 'main' : null }
  }

  const stored = $sessionStates.get()[id]?.storedSessionId ?? id
  const ids = stored === id ? [id] : [id, stored]

  // The primary composer answers only for the session it is showing; a tile
  // answers for its own. Anything else stays `tile:<id>` — an absent tile
  // drops the request (fail-closed) rather than leaking it into whatever
  // session the primary is displaying.
  const shownInPrimary = ids.includes($selectedStoredSessionId.get() ?? '')

  return { ids, stored, target: shownInPrimary ? 'main' : (`tile:${stored}` as ComposerTarget) }
}

/** THE composer draft surface (#116305 item 1): read, write, insert, and
 *  submit a session's input WITHOUT touching app DOM — mounted surfaces
 *  answer for their own sessions over the app's focus bus, so a plugin
 *  addressing one session can never reach another's composer. Addressing:
 *  `null` = the active composer (what the user last clicked into); a
 *  session id (stored or runtime) = that session's composer, whether it is
 *  the primary surface or a tile; the literal `'new'` = the fresh draft
 *  with no session id yet. Every verb fails closed: an address that
 *  resolves to no live surface returns `null`/`false` (or is dropped, for
 *  `focus`) — it never broadcasts and never throws. */
export const composerHost = {
  /** The live draft text of one composer, or null when nothing holds it.
   *  Mounted surfaces answer with their in-DOM text (current, incl.
   *  unsaved keystrokes); an unmounted session falls back to its debounced
   *  persisted stash; `null` address = the active composer (no fallback —
   *  nobody on screen is answering, which is reported as null). */
  getDraft: async (sessionId: null | string = null): Promise<null | string> => {
    const { ids, stored } = resolveComposerAddress(sessionId)

    if (!ids.length) {
      const live = await requestComposerGetDraft([], { active: true })

      return live ? live.text : null
    }

    const live = await requestComposerGetDraft(ids)

    if (live) {
      return live.text
    }

    return takeSessionDraft(stored).text || null
  },

  /** Replace a composer's whole draft. `@`-ref / `/` tokens in the text
   *  hydrate into chips exactly like an official paste (the app owns the
   *  markup). Returns false when no mounted surface answers for the
   *  address — the draft of an unmounted session is never half-written. */
  setDraft: async (sessionId: null | string, text: string): Promise<boolean> => {
    if (typeof text !== 'string') {
      return false
    }

    const { ids } = resolveComposerAddress(sessionId)

    return requestComposerSetDraft(ids, text, ids.length ? undefined : { active: true })
  },

  /** Append text to a composer's draft through the app's own insert modes
   *  ('block' = paragraph at end, 'inline' = same line, 'prefix' = start —
   *  the slash-command seat). Acknowledged like `setDraft`: resolves true
   *  when a mounted surface claimed and applied the text, false when the
   *  text trims to nothing or no surface answers for the address within the
   *  bus settle window — never a silent no-op. */
  insertText: (sessionId: null | string, text: string, opts?: { mode?: ComposerInsertMode }): Promise<boolean> => {
    const { target } = resolveComposerAddress(sessionId)

    if (target === null) {
      return Promise.resolve(false)
    }

    return requestComposerInsertAcked(text, { mode: opts?.mode ?? 'block', target })
  },

  /** Send `text` as if the user typed it + pressed Enter, and return
   *  whether a visible surface claimed it. Same fail-closed contract as
   *  the internal bus: no exact visible composer for the address → false,
   *  never a broadcast into whichever pane happens to be mounted. */
  submit: (sessionId: null | string, text: string): boolean => {
    const { target } = resolveComposerAddress(sessionId)

    return target !== null && requestComposerSubmit(text, { target })
  },

  /** Put the caret in a composer — the app's own focus bus, same address
   *  resolution as the verbs above. `insertText`/`setDraft` already focus a
   *  VISIBLE surface they paint; this is the standalone verb for the other
   *  cases (return the caret after a plugin popover/dialog closes, a
   *  keybind that "goes to the input") that plugins used to reach with a
   *  hand-built `hermes:composer-focus` CustomEvent. Fail-closed like the
   *  rest: an absent tile drops the request instead of focusing whatever
   *  the primary happens to show. */
  focus: (sessionId: null | string = null): void => {
    const { target } = resolveComposerAddress(sessionId)

    if (target !== null) {
      requestComposerFocus(target)
    }
  },

  /** Turn Chromium's spell check (the red squiggles) on or off for one
   *  composer.
   *
   *  The inputs have spell check OFF by default (#44415) -- browser
   *  spellcheck flagged code, paths and slash commands. This does not change
   *  that for anyone: it is an opt-in for a plugin that ships its own
   *  dictionary, where the user installed the plugin precisely to get spell
   *  checking. Chromium still decides WHAT is flagged; a plugin cannot teach
   *  it about code or paths here, only add/remove single words through
   *  `host.spellcheck`.
   *
   *  Fail-closed, and it remembers nothing: this is a live setting for a
   *  mounted composer, not a preference stored against the address. `true`
   *  means a mounted surface applied it, right now. `false` means nothing
   *  happened -- no surface answered for the address (a closed tile, an
   *  unknown session, a non-boolean `enabled`) or the surface had no editable
   *  to write to -- and nothing will apply later. A plugin that wants the
   *  setting on a composer that is not mounted yet calls again once it is;
   *  the `false` is its cue to retry. Addressing is the same as every other
   *  verb -- `null` for the active composer, a session id for a specific one,
   *  `'new'` for the fresh draft.
   *
   *  `sessionId` is ROUTING, not authorization: it picks which composer
   *  answers, exactly like `setDraft`/`insertText`, and plugins already run
   *  in-process with the app's own privileges. It is never checked against
   *  the app's session list, so an id that names no session simply resolves to
   *  a surface nobody holds and fails closed.
   *
   *  On macOS this also re-enables Chromium's smart quotes / dashes for the
   *  composer: those substitutions ride the same spellcheck gate, and the app
   *  keeps `autocorrect` on there for the user's custom Text replacements.
   *
   *  Asynchronous, because the answer depends on whether a mounted composer
   *  actually claimed and applied the request -- a fire-and-forget call could
   *  not tell "on" from "never arrived". */
  setSpellcheck: (sessionId: null | string, enabled: boolean): Promise<boolean> => {
    if (typeof enabled !== 'boolean') {
      return Promise.resolve(false)
    }

    const { target } = resolveComposerAddress(sessionId)

    if (target === null) {
      return Promise.resolve(false)
    }

    return requestComposerSpellcheck(enabled, target)
  }
}
