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

import { queryVisible } from '@/components/pane-shell/pane-visibility'

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

/** Inline edit composer root — mounted only while a user bubble is being edited. */
const EDIT_COMPOSER_ROOT = '[data-slot="aui_edit-composer-root"]'

/** Attribute-safe selector fragment. jsdom (vitest) does not ship `CSS.escape`. */
const cssEscape = (value: string): string => {
  if (typeof CSS !== 'undefined' && typeof CSS.escape === 'function') {
    return CSS.escape(value)
  }

  // Our targets are `'main'` / `'edit'` / `'tile:<id>'` — alphanumerics plus `:`
  // and `-`. Escape anything outside that set so a weird id cannot break the
  // attribute selector.
  return value.replace(/[^a-zA-Z0-9_:-]/g, ch => `\\${ch}`)
}

interface SubmitDetail {
  target: ComposerTarget
  text: string
}

let activeTarget: ComposerTarget = 'main'

/**
 * The chat surface currently on screen (`data-composer-target` hung off each
 * ChatView). Inactive tabs stay mounted with `data-pane-hidden`, so this uses
 * the same visibility policy as every other document-wide surface lookup.
 */
const visibleChatTarget = (): ComposerTarget | null => {
  if (typeof document === 'undefined') {
    return null
  }

  const surface = queryVisible<HTMLElement>('[data-composer-target]')
  const target = surface?.dataset.composerTarget

  return target ? (target as ComposerTarget) : null
}

/** True when `target` still has a live, on-screen subscriber. */
const targetIsReachable = (target: ComposerTarget): boolean => {
  if (typeof document === 'undefined') {
    return true
  }

  // The edit composer is an in-thread overlay, not a chat surface — it never
  // stamps `data-composer-target`. While its root is mounted it still owns the
  // bus; once it tears down the claim is dead.
  if (target === 'edit') {
    return Boolean(document.querySelector(EDIT_COMPOSER_ROOT))
  }

  // Exact match on a VISIBLE surface. Background keep-alive tabs carry the same
  // `data-composer-target` but sit under `data-pane-hidden`, so queryVisible
  // filters them out.
  if (queryVisible(`[data-composer-target="${cssEscape(target)}"]`)) {
    return true
  }

  // A different chat surface is on screen → this claim is buried or gone.
  // (A claim with zero stamped surfaces yet — first paint, pure-unit tests —
  // keeps the marked key until the DOM contradicts it.)
  if (queryVisible('[data-composer-target]')) {
    return false
  }

  return true
}

/**
 * The composer `'active'` should route to right now.
 *
 * The cached claim (`activeTarget`) wins while its surface is still on screen.
 * Tab stacks keep inactive panes mounted, so focusing a tile then clicking the
 * main tab leaves `activeTarget` pointing at a buried composer — with no
 * subscriber on the visible surface, every type-to-focus keystroke is
 * preventDefault'd and dropped. Heal to the visible chat surface (or main)
 * whenever the claim is off-screen or gone, and keep the cache honest so Esc /
 * voice / soft `/` agree with the keyboard path.
 */
const resolveActive = (): ComposerTarget => {
  if (targetIsReachable(activeTarget)) {
    return activeTarget
  }

  const visible = visibleChatTarget() ?? 'main'

  activeTarget = visible

  return visible
}

const resolve = (target: ComposerTarget | 'active') => (target === 'active' ? resolveActive() : target)

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

/** Hand the routing key back when a composer unmounts, so `'active'` can never
 *  resolve to a composer that no longer has a subscriber — such a request is
 *  dispatched and then dropped by every mounted composer's target filter, and
 *  nothing re-marks the active composer on its own.
 *
 *  Guarded on identity: a composer unmounting AFTER another one claimed the key
 *  (closing a background tile, a deferred edit-close cleanup) must not steal it
 *  from the live claimant. Falls through to {@link resolveActive} when the
 *  caller's surface is buried rather than gone, so closing on a tab switch that
 *  already re-fronted another chat surfaces there immediately. */
export const releaseActiveComposer = (target: ComposerTarget) => {
  if (activeTarget !== target) {
    return
  }

  // Prefer the visible chat surface over a hard `'main'` default — releasing a
  // closed tile while another tile is fronted should land there, not the
  // (possibly buried) workspace tab.
  activeTarget = visibleChatTarget() ?? 'main'
}

/** The composer that last held focus — the target `'active'` resolves to.
 *  Used by broadcast listeners (voice, Esc-to-stop) to act on exactly one.
 *  Heals a stale claim the same way {@link requestComposerFocus} does, so Esc
 *  and type-to-focus never disagree after a tab switch left the bus pointing at
 *  a keep-alive-mounted background composer. */
export const getActiveComposer = (): ComposerTarget => resolveActive()

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

/** Drop focus from the main composer input (status-stack chrome, sidebar, etc.).
 *  Skips inactive tabs — they stay mounted, so an unscoped lookup can land on a
 *  background composer and leave the visible one focused. */
export const blurComposerInput = () => {
  const el = queryVisible(`[data-slot="${RICH_INPUT_SLOT}"]`)

  if (el && document.activeElement === el) {
    el.blur()
  }
}
