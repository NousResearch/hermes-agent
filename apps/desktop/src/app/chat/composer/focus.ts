/**
 * Composer focus + external-insert bus.
 *
 * Mutations from outside the composer (sidebar attach, drag drop, terminal
 * Cmd+L, preview console, etc.) dispatch through here. Each composer subscribes
 * and routes the work back into its own ref/state.
 *
 * `dispatch` defers to a macrotask so synchronous click/keydown handlers
 * (react-arborist row focus, picker `node.select()`) finish first and don't
 * steal focus from the composer effect.
 */

import type { InlineRefInput } from './inline-refs'
import { RICH_INPUT_SLOT } from './rich-editor'

/** Composer routing key. The main chat is `'main'`, the edit composer
 *  `'edit'`; scoped composers (session tiles) use `'tile:<id>'`. */
export type ComposerTarget = 'edit' | 'main' | (string & {})
export type ComposerInsertMode = 'block' | 'inline'

export interface FocusDetail {
  target: ComposerTarget
  /** Append after focus (type-to-focus / soft `/`). */
  typeChar?: string
}

interface InsertDetail {
  mode: ComposerInsertMode
  target: ComposerTarget
  text: string
}

interface InsertRefsDetail {
  refs: InlineRefInput[]
  target: ComposerTarget
}

const FOCUS_EVENT = 'hermes:composer-focus'
const INSERT_EVENT = 'hermes:composer-insert'
const INSERT_REFS_EVENT = 'hermes:composer-insert-refs'
const SUBMIT_EVENT = 'hermes:composer-submit'
const VOICE_TOGGLE_EVENT = 'hermes:composer-voice-toggle'

interface SubmitDetail {
  target: ComposerTarget
  text: string
}

let activeTarget: ComposerTarget = 'main'

const resolve = (target: ComposerTarget | 'active') => (target === 'active' ? activeTarget : target)

const dispatch = <T>(name: string, detail: T) => {
  if (typeof window === 'undefined') {
    return
  }

  window.setTimeout(() => window.dispatchEvent(new CustomEvent<T>(name, { detail })), 0)
}

const subscribe = <T>(name: string, handler: (detail: T) => void) => {
  if (typeof window === 'undefined') {
    return () => undefined
  }

  const listener = (event: Event) => {
    const detail = (event as CustomEvent<T>).detail

    if (detail) {
      handler(detail)
    }
  }

  window.addEventListener(name, listener)

  return () => window.removeEventListener(name, listener)
}

export const markActiveComposer = (target: ComposerTarget) => {
  activeTarget = target
}

/** The composer that last held focus — the target `'active'` resolves to.
 *  Used by broadcast listeners (voice, Esc-to-stop) to act on exactly one. */
export const getActiveComposer = (): ComposerTarget => activeTarget

export const requestComposerFocus = (
  target: ComposerTarget | 'active' = 'active',
  { typeChar }: { typeChar?: string } = {}
) => dispatch<FocusDetail>(FOCUS_EVENT, { target: resolve(target), typeChar })

export const requestComposerInsert = (
  text: string,
  { mode = 'block', target = 'active' }: { mode?: ComposerInsertMode; target?: ComposerTarget | 'active' } = {}
) => {
  const trimmed = text.trim()

  if (!trimmed) {
    return
  }

  dispatch<InsertDetail>(INSERT_EVENT, { mode, target: resolve(target), text: trimmed })
}

export const onComposerFocusRequest = (handler: (detail: FocusDetail) => void) =>
  subscribe<FocusDetail>(FOCUS_EVENT, handler)

export const onComposerInsertRequest = (handler: (detail: InsertDetail) => void) =>
  subscribe<InsertDetail>(INSERT_EVENT, handler)

/** Insert typed ref chips (carrying a display label) into a composer — the
 * structured cousin of {@link requestComposerInsert}, used for session links. */
export const requestComposerInsertRefs = (
  refs: InlineRefInput[],
  { target = 'active' }: { target?: ComposerTarget | 'active' } = {}
) => {
  if (refs.length) {
    dispatch<InsertRefsDetail>(INSERT_REFS_EVENT, { refs, target: resolve(target) })
  }
}

export const onComposerInsertRefsRequest = (handler: (detail: InsertRefsDetail) => void) =>
  subscribe<InsertRefsDetail>(INSERT_REFS_EVENT, handler)

/** Submit a prompt through a composer as if the user typed + sent it. Lets
 * external panels (e.g. the review pane's "let the agent ship it" button) hand
 * the agent a task without the user round-tripping through the input. */
export const requestComposerSubmit = (
  text: string,
  { target = 'active' }: { target?: ComposerTarget | 'active' } = {}
) => {
  const trimmed = text.trim()

  if (trimmed) {
    dispatch<SubmitDetail>(SUBMIT_EVENT, { target: resolve(target), text: trimmed })
  }
}

export const onComposerSubmitRequest = (handler: (detail: SubmitDetail) => void) =>
  subscribe<SubmitDetail>(SUBMIT_EVENT, handler)

/** Toggle ONE composer's voice conversation — the `composer.voice` hotkey
 *  (Ctrl+B) reaches the composer that owns voice. Defaults to the active
 *  composer so N tiles don't all flip together. */
export const requestVoiceToggle = (target: ComposerTarget | 'active' = 'active') =>
  dispatch<{ target: ComposerTarget }>(VOICE_TOGGLE_EVENT, { target: resolve(target) })

export const onComposerVoiceToggleRequest = (handler: (target: ComposerTarget) => void) =>
  subscribe<{ target: ComposerTarget }>(VOICE_TOGGLE_EVENT, ({ target }) => handler(target))

/**
 * Focus a composer input across React commit + browser focus restore.
 *
 * The triple-call survives:
 *   - sync: contenteditable already mounted
 *   - rAF:  React just committed a `renderComposerContents` swap
 *   - 0ms:  browser focus reclaim from a click target inside an external panel
 */
export const focusComposerInput = (el: HTMLElement | null) => {
  if (!el) {
    return
  }

  // Skip when already focused: focus() runs the full focusing steps (forcing
  // layout) even on the active element, and during a session switch the DOM is
  // large and dirty — the redundant retries were measurably expensive there.
  const focus = () => {
    if (document.activeElement !== el) {
      el.focus({ preventScroll: true })
    }
  }

  focus()
  window.requestAnimationFrame(focus)
  window.setTimeout(focus, 0)
}

/** Drop focus from the main composer input (status-stack chrome, sidebar, etc.). */
export const blurComposerInput = () => {
  const el = document.querySelector(`[data-slot="${RICH_INPUT_SLOT}"]`) as HTMLElement | null

  if (el && document.activeElement === el) {
    el.blur()
  }
}
