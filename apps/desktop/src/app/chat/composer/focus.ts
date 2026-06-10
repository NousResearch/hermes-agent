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

export type ComposerTarget = 'edit' | 'main'
export type ComposerInsertMode = 'block' | 'inline'

interface FocusDetail {
  target: ComposerTarget
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

export const requestComposerFocus = (target: ComposerTarget | 'active' = 'active') =>
  dispatch<FocusDetail>(FOCUS_EVENT, { target: resolve(target) })

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

export const onComposerFocusRequest = (handler: (target: ComposerTarget) => void) =>
  subscribe<FocusDetail>(FOCUS_EVENT, ({ target }) => handler(target))

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

  const focus = () => el.focus({ preventScroll: true })

  focus()
  window.requestAnimationFrame(focus)
  window.setTimeout(focus, 0)
}
