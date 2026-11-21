/**
 * ⌘C / Ctrl+C copies an UNFOCUSED text selection.
 *
 * Chromium's copy command acts on the focused frame's editable or on its
 * selection, and transcript text is neither: the selection lives in the DOM
 * while focus sits on `body` (or stays wherever it was). A real keypress hides
 * this because the transcript surface answers the chord itself — a shortcut
 * that only synthesizes VK_C/VK_V reaches Chromium directly (a mouse button
 * mapped to Copy in Logi Options+, say) and lands on the silent path, so the
 * clipboard never changes.
 *
 * Only text selections qualify. Inputs and textareas carry their selection on
 * the element (window.getSelection() stays empty there), so they keep the
 * native command, which is also what a focused contenteditable needs.
 */

/** True when the clipboard chord should be answered from the DOM selection. */
export function shouldCopyUnfocusedSelection(event: Pick<KeyboardEvent, 'target'>): boolean {
  const selection = typeof document.getSelection === 'function' ? document.getSelection() : null

  if (!selection || selection.isCollapsed) {
    return false
  }

  if (!selection.toString().trim()) {
    return false
  }

  const active = document.activeElement

  // A focused form field owns its own copy: its selection is not in
  // `document.getSelection()`, and the native command handles it correctly.
  if (active instanceof HTMLInputElement || active instanceof HTMLTextAreaElement) {
    return false
  }

  const target = event.target instanceof Element ? event.target : null
  const anchor = selection.anchorNode
  const anchorElement = anchor instanceof Element ? anchor : anchor?.parentElement

  // The selection must live where the chord landed — a stale selection in
  // another pane must not answer a keystroke aimed at this one.
  if (target && anchorElement && !target.contains(anchorElement) && !anchorElement.contains(target)) {
    return false
  }

  return true
}
