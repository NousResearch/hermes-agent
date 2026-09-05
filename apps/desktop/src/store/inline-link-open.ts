import { atom } from 'nanostores'

import { IS_MAC } from '@/lib/keybinds/combo'
import { persistBoolean, storedBoolean } from '@/lib/storage'

export const INLINE_LINK_REQUIRE_MODIFIER_STORAGE_KEY = 'hermes.desktop.inlineLinks.requireModifier'

/** Marks scoped chat URLs / paths / chips so CSS can swap cursor with the
 *  document flags below — one window listener, not one per link. */
export const INLINE_LINK_GATED_ATTR = 'data-inline-link-gated'
export const INLINE_LINK_REQUIRE_MODIFIER_ATTR = 'data-inline-link-require-modifier'
export const INLINE_LINK_MODIFIER_HELD_ATTR = 'data-inline-link-modifier'

/** Desktop-local: ⌘-click (Mac) / Ctrl-click (elsewhere) to open inline chat
 *  paths and URLs. Default off — a regular click still opens, matching upstream. */
export const $requireModifierToOpenInlineLinks = atom(
  storedBoolean(INLINE_LINK_REQUIRE_MODIFIER_STORAGE_KEY, false)
)

/** Live platform-modifier state while the preference is on. Document flags
 *  drive cursor so a stationary pointer still updates. */
export const $inlineLinkOpenModifierHeld = atom(false)

let modifierTrackingAttached = false

export function setRequireModifierToOpenInlineLinks(value: boolean) {
  $requireModifierToOpenInlineLinks.set(value)
}

export function isInlineLinkOpenModifier(event: { ctrlKey: boolean; metaKey: boolean }): boolean {
  return IS_MAC ? event.metaKey : event.ctrlKey
}

/** Whether this activation should open the inline link.
 *
 *  Keyboard activation of a focused `<a>` synthesizes `click` with `detail === 0`
 *  and must still open when the modifier setting is on. Mouse clicks need the
 *  platform modifier; a regular click is left for text selection. */
export function shouldOpenInlineLink(event: { ctrlKey: boolean; detail: number; metaKey: boolean }): boolean {
  if (!$requireModifierToOpenInlineLinks.get()) {
    return true
  }

  if (event.detail === 0) {
    return true
  }

  return isInlineLinkOpenModifier(event)
}

function isPlatformModifierRelease(event: KeyboardEvent): boolean {
  if (IS_MAC) {
    return event.key === 'Meta' || event.code === 'MetaLeft' || event.code === 'MetaRight'
  }

  return event.key === 'Control' || event.code === 'ControlLeft' || event.code === 'ControlRight'
}

function syncDocumentFlags() {
  if (typeof document === 'undefined') {
    return
  }

  const root = document.documentElement
  const requireModifier = $requireModifierToOpenInlineLinks.get()
  const held = $inlineLinkOpenModifierHeld.get()

  if (!requireModifier) {
    root.removeAttribute(INLINE_LINK_REQUIRE_MODIFIER_ATTR)
    root.removeAttribute(INLINE_LINK_MODIFIER_HELD_ATTR)

    return
  }

  root.setAttribute(INLINE_LINK_REQUIRE_MODIFIER_ATTR, 'true')

  if (held) {
    root.setAttribute(INLINE_LINK_MODIFIER_HELD_ATTR, 'held')
  } else {
    root.removeAttribute(INLINE_LINK_MODIFIER_HELD_ATTR)
  }
}

function setModifierHeld(held: boolean) {
  if ($inlineLinkOpenModifierHeld.get() === held) {
    syncDocumentFlags()

    return
  }

  $inlineLinkOpenModifierHeld.set(held)
  syncDocumentFlags()
}

function onModifierKey(event: KeyboardEvent) {
  if (!$requireModifierToOpenInlineLinks.get()) {
    return
  }

  if (event.type === 'keyup' && isPlatformModifierRelease(event)) {
    setModifierHeld(false)

    return
  }

  setModifierHeld(isInlineLinkOpenModifier(event))
}

function onModifierClear() {
  setModifierHeld(false)
}

function onVisibilityChange() {
  if (document.visibilityState !== 'visible') {
    onModifierClear()
  }
}

function attachModifierTracking() {
  if (modifierTrackingAttached || typeof window === 'undefined') {
    return
  }

  modifierTrackingAttached = true
  window.addEventListener('keydown', onModifierKey, true)
  window.addEventListener('keyup', onModifierKey, true)
  window.addEventListener('blur', onModifierClear)
  document.addEventListener('visibilitychange', onVisibilityChange)
}

function detachModifierTracking() {
  if (!modifierTrackingAttached || typeof window === 'undefined') {
    return
  }

  modifierTrackingAttached = false
  window.removeEventListener('keydown', onModifierKey, true)
  window.removeEventListener('keyup', onModifierKey, true)
  window.removeEventListener('blur', onModifierClear)
  document.removeEventListener('visibilitychange', onVisibilityChange)
}

function syncModifierTracking(requireModifier: boolean) {
  if (requireModifier) {
    attachModifierTracking()
    syncDocumentFlags()

    return
  }

  detachModifierTracking()
  setModifierHeld(false)
}

$requireModifierToOpenInlineLinks.subscribe(value => {
  persistBoolean(INLINE_LINK_REQUIRE_MODIFIER_STORAGE_KEY, value)
  syncModifierTracking(value)
})
