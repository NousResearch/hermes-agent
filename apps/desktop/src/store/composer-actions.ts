import { atom } from 'nanostores'

/**
 * Composer micro actions — the floating pill strip at the top of the composer's
 * overlay lane.
 *
 * A badge is a label and a `run`; what it does is entirely the registrar's
 * business. Core ships none — the strip is a seam, filled by contributions to
 * the `composer.microActions` area (see `composer/contrib.ts`).
 *
 * Session-scoped like every other stack feed: keyed by RUNTIME session id, and
 * written immutably per key so `useSessionSlice` bails out on other sessions'
 * churn.
 */
export interface ComposerAction {
  /** Stable within a session; also the render key. */
  id: string
  label: string
  /** Codicon name rendered before the label. */
  icon?: string
  disabled?: boolean
  /** What the badge does. Anything: submit a prompt, open a pane, call the
   *  gateway. Receives the session it was registered for. */
  run: (sessionId: string) => Promise<void> | void
}

export const $composerActionsBySession = atom<Record<string, ComposerAction[]>>({})

/**
 * Two badge sets that would render identically.
 *
 * `run` is excluded on purpose: it's a fresh closure on every resolve, so
 * including it would make every comparison false and defeat the bail-out
 * below. The invariant that buys: a provider must not change what `run` DOES
 * without also changing a visible field.
 */
const same = (a: readonly ComposerAction[], b: readonly ComposerAction[]) =>
  a.length === b.length &&
  a.every((x, i) => {
    const y = b[i]!

    return x.id === y.id && x.label === y.label && x.icon === y.icon && x.disabled === y.disabled
  })

/**
 * Publish this session's badge set.
 *
 * A no-op resolve must not write: providers are re-resolved on every composer
 * render, and an equal-but-fresh array would re-render the stack on every
 * keystroke. An unchanged set keeps its previous reference, so the store stays
 * quiet.
 */
export function setComposerActions(sid: string, actions: ComposerAction[]) {
  const current = $composerActionsBySession.get()

  if (!sid || (current[sid] ? same(current[sid], actions) : actions.length === 0)) {
    return
  }

  const next = { ...current }

  if (actions.length > 0) {
    next[sid] = actions
  } else {
    delete next[sid]
  }

  $composerActionsBySession.set(next)
}
