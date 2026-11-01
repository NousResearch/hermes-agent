import { flushSync } from 'react-dom'

import { onPaneFocusRequest } from '@/components/pane-shell/pane-focus'
import {
  $activeTreeGroup,
  $hoveredTreeGroup,
  isPaneVisible,
  noteActiveTreeGroup
} from '@/components/pane-shell/tree/store'
import { isEditableTarget, OVERLAY_SURFACE } from '@/lib/keybinds/combo'
import { isTouchInteraction } from '@/lib/touch-interaction'
import { $composerPopout } from '@/store/composer-popout'

import { $floatingComposerOwner, type FloatingComposerOwner } from './floating-state'
import { EDIT_COMPOSER_ROOT, markActiveComposer } from './focus'
import { placeCaretEnd } from './rich-editor'

const surfaces = new Map<string, Omit<FloatingComposerOwner, 'id'>>()
const captureOwners = new Set<string>()
const carets = new WeakMap<HTMLElement, Range>()
let pointerDownTarget: Element | null = null
let keyboardNavigation = false
let reconcileQueued = false
let pendingPaneFocus: string | null = null
let stopPaneFocus: (() => void) | undefined

/** The inline message edit lives in the transcript, outside every composer
 * host, and takes focus programmatically when a bubble is clicked. That focus
 * is the user's choice, not a delayed callback to redirect: stealing it hands
 * the caret back to the pane composer, and the edit's blur guard then cancels
 * the edit ~80ms after it opened (#112935). */
const inInlineEdit = (el: Element | null) => Boolean(el?.closest(EDIT_COMPOSER_ROOT))

/** The native overflow menu (⋮ → Download, Playback speed) of an inline
 * <audio controls>/<video controls> lives in the element's user-agent shadow
 * root, so its focus is retargeted to the AUDIO/VIDEO host. Redirecting that
 * focus to the pane composer — on focusin or on the next pointermove — made
 * Chromium close the menu the instant it opened (#135225). */
const inNativeMediaControls = (el: Element | null) => Boolean(el?.closest('audio[controls], video[controls]'))

/** A caret the user placed in a live text-entry surface outside every composer
 * — the clarify answer box, the transcript find bar, a dialog field — is their
 * own focus, exactly like the inline edit: moving the pointer must not hand the
 * caret back to the pane composer, and a refused redirect must leave that
 * element's focusin visible to React. Composer editors are excluded so stale
 * focus callbacks cannot override an explicitly selected pane. */
const keepsOwnFocus = (el: Element | null) =>
  inNativeMediaControls(el) ||
  inInlineEdit(el) ||
  (isEditableTarget(el) && !el?.closest('[data-slot="composer-rich-input"]'))

/** Focus inside an open floating layer — a popover, menu, listbox or dialog
 * portaled over the transcript — is the user's own as well. Radix moves focus
 * into such a layer when it opens and dismisses it the moment focus leaves, so
 * pulling the caret back to the pane composer on the very next pointermove
 * closed the message reaction picker before the pointer could reach it. */
const inFloatingLayer = (el: Element | null) => Boolean(el?.closest(OVERLAY_SURFACE))

function rememberCaret(editor: EventTarget | null) {
  const selection = window.getSelection()

  if (editor instanceof HTMLElement && editor.dataset.slot === 'composer-rich-input' && selection?.rangeCount) {
    const range = selection.getRangeAt(0)

    if (editor.contains(range.commonAncestorContainer)) {
      carets.set(editor, range.cloneRange())
    }
  }
}

const trackBlur = (event: FocusEvent) => rememberCaret(event.target)

const releasePointer = () => {
  pointerDownTarget = null
}

/** A non-collapsed selection anchored outside the composer editor is the user
 * selecting transcript text: focusing the composer must not clear it. */
function selectionOutsideComposer(): boolean {
  const selection = window.getSelection()

  if (!selection || selection.rangeCount === 0 || selection.isCollapsed) {
    return false
  }

  const { anchorNode } = selection
  const anchorEl = anchorNode instanceof Element ? anchorNode : (anchorNode?.parentElement ?? null)

  return !anchorEl?.closest('[data-slot="composer-rich-input"]')
}

/** Focus recovery preserves selections and controls that own their focus. */
function focusSelectedComposer() {
  const owner = $floatingComposerOwner.get()

  if (!owner || isTouchInteraction() || selectionOutsideComposer() || inFloatingLayer(document.activeElement)) {
    return
  }

  const host = [...document.querySelectorAll<HTMLElement>('[data-composer-owner]')].find(
    el => el.dataset.composerOwner === owner.id
  )

  const editor = host?.querySelector<HTMLElement>('[data-slot="composer-rich-input"]')

  if (editor && document.activeElement !== editor) {
    // Selection is already committed. Do not queue retries that could steal
    // the native focus a clicked control or inline editor is about to take.
    editor.focus({ preventScroll: true })
    const caret = carets.get(editor)
    const selection = window.getSelection()

    if (caret && editor.contains(caret.startContainer) && editor.contains(caret.endContainer) && selection) {
      selection.removeAllRanges()
      selection.addRange(caret)
    } else {
      placeCaretEnd(editor)
    }
  }
}

function selectSurface(id: string) {
  const captureOwner = [...captureOwners].find(owner => surfaces.has(owner))
  const surface = surfaces.get(id)

  if (!surface || ($composerPopout.get().poppedOut && captureOwner && captureOwner !== id)) {
    return
  }

  if ($floatingComposerOwner.get()?.id !== id) {
    const active = document.activeElement
    rememberCaret(active)

    // A preserve-focus control can cancel native blur. Do not leave another
    // visible composer accepting characters for the newly selected recipient.
    if (
      active instanceof HTMLElement &&
      active.dataset.slot === 'composer-rich-input' &&
      active.closest<HTMLElement>('[data-composer-owner]')?.dataset.composerOwner !== id
    ) {
      active.blur()
    }

    $floatingComposerOwner.set({ id, ...surface })
  }

  markActiveComposer(surface.target)
}

function surfaceAt(target: Element): string | undefined {
  const host = target.closest<HTMLElement>('[data-composer-owner]')

  if (host) {
    return host.hasAttribute('data-floating-composer-target') ? undefined : host.dataset.composerOwner
  }

  const id = target.closest<HTMLElement>('[data-chat-surface]')?.dataset.composerSurfaceId

  if (id && surfaces.has(id)) {
    return id
  }

  const groupId = target.closest<HTMLElement>('[data-tree-group]')?.dataset.treeGroup

  return [...surfaces].find(([, surface]) => surface.groupId === groupId)?.[0]
}

function trackPointerDown(event: PointerEvent) {
  const target = event.target instanceof Element ? event.target : null
  pointerDownTarget = target
  keyboardNavigation = false
  pendingPaneFocus = null

  if (!target) {
    return
  }

  const id = surfaceAt(target)

  if (id && $floatingComposerOwner.get()?.id !== id) {
    // Select before native focus and the next keystroke. Passing through a
    // different chat on the way to Files must not change the recipient.
    flushSync(() => selectSurface(id))
    // Let the clicked control take its native focus first. A control that
    // deliberately prevented that default instead leaves typing with this chat.
    queueMicrotask(() => {
      if ($floatingComposerOwner.get()?.id === id && document.activeElement === document.body) {
        focusSelectedComposer()
      }
    })
  }
}

function fulfillPaneFocus() {
  const paneId = pendingPaneFocus

  if (!paneId || !isPaneVisible(paneId)) {
    pendingPaneFocus = null

    return
  }

  const id = [...surfaces].find(([, surface]) => surface.paneId === paneId)?.[0]

  if (id) {
    pendingPaneFocus = null
    flushSync(() => selectSurface(id))

    if ($floatingComposerOwner.get()?.id === id) {
      noteActiveTreeGroup(surfaces.get(id)!.groupId)
      focusSelectedComposer()
    }
  }
}

function trackKey(event: KeyboardEvent) {
  keyboardNavigation = event.type === 'keydown' && event.key === 'Tab' && !event.ctrlKey && !event.metaKey

  if (keyboardNavigation) {
    // Tab is a newer explicit choice, including a control in the same owner
    // or outside chat surfaces. A resumed pane must not reclaim that focus.
    pendingPaneFocus = null
  }
}

function trackFocus(event: FocusEvent) {
  const target = event.target instanceof Element ? event.target : null

  if (!target) {
    return
  }

  const hostId = target.closest<HTMLElement>('[data-composer-owner]')?.dataset.composerOwner

  if (hostId === $floatingComposerOwner.get()?.id) {
    return
  }

  const id = hostId ?? surfaceAt(target)

  if (!id) {
    return
  }

  // A delayed focus/restore isn't a navigation gesture. A clicked control,
  // keyboard Tab, or the inline edit opened by a bubble click still keeps its
  // normal focus, without redirecting to the input.
  if (
    isTouchInteraction() ||
    keyboardNavigation ||
    keepsOwnFocus(target) ||
    (pointerDownTarget && target.contains(pointerDownTarget))
  ) {
    flushSync(() => selectSurface(id))
  } else {
    // A refused redirect leaves focus where it landed: that element's focusin
    // must still reach React and other root listeners.
    if (selectionOutsideComposer()) {
      return
    }

    event.stopImmediatePropagation()
    const owner = $floatingComposerOwner.get()

    if (owner) {
      noteActiveTreeGroup(owner.groupId)
    }

    focusSelectedComposer()
  }
}

function reconcile() {
  const owner = $floatingComposerOwner.get()
  const groupId = owner?.groupId ?? $hoveredTreeGroup.get() ?? $activeTreeGroup.get()
  const replacement = [...surfaces].find(([, surface]) => surface.groupId === groupId)?.[0]
  const next = replacement ?? (owner && surfaces.has(owner.id) ? owner.id : surfaces.keys().next().value)

  if (!next) {
    $floatingComposerOwner.set(null)
  } else {
    selectSurface(next)
  }
}

/** Registration can span an old tab's cleanup and its replacement's mount.
 * Reconcile after that commit, never temporarily select an unrelated split. */
export function registerFloatingComposer(id: string, surface: Omit<FloatingComposerOwner, 'id'>): () => void {
  surfaces.set(id, surface)

  if (surfaces.size === 1) {
    stopPaneFocus = onPaneFocusRequest(paneId => {
      pendingPaneFocus = paneId
      queueMicrotask(fulfillPaneFocus)
    })
    window.addEventListener('pointerdown', trackPointerDown, true)
    window.addEventListener('focusin', trackFocus, true)
    window.addEventListener('focusout', trackBlur, true)
    window.addEventListener('pointerup', releasePointer, true)
    window.addEventListener('keydown', trackKey, true)
    window.addEventListener('keyup', trackKey, true)
  }

  if (!$floatingComposerOwner.get() || $floatingComposerOwner.get()?.groupId === surface.groupId) {
    selectSurface(id)
  }

  if (pendingPaneFocus) {
    queueMicrotask(fulfillPaneFocus)
  }

  return () => {
    surfaces.delete(id)

    if (!reconcileQueued) {
      reconcileQueued = true
      queueMicrotask(() => {
        reconcileQueued = false
        flushSync(reconcile)
      })
    }

    if (surfaces.size === 0) {
      stopPaneFocus?.()
      stopPaneFocus = undefined
      window.removeEventListener('pointerdown', trackPointerDown, true)
      window.removeEventListener('focusin', trackFocus, true)
      window.removeEventListener('focusout', trackBlur, true)
      window.removeEventListener('pointerup', releasePointer, true)
      window.removeEventListener('keydown', trackKey, true)
      window.removeEventListener('keyup', trackKey, true)
      pointerDownTarget = null
      keyboardNavigation = false
    }
  }
}

export function claimFloatingComposer(id: string) {
  selectSurface(id)
  const owner = $floatingComposerOwner.get()

  if (owner) {
    markActiveComposer(owner.target)
  }
}

/** Keep the microphone's stop controls and hotkey on its real owner. */
export function pinFloatingComposerCapture(id: string): () => void {
  captureOwners.add(id)
  claimFloatingComposer(id)

  return () => {
    captureOwners.delete(id)
  }
}
