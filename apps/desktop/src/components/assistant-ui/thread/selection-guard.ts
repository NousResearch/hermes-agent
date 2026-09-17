/**
 * Keeping a reader's text highlight alive while the transcript virtualizes.
 *
 * The timeline renders a bounded DOM budget: as a turn streams, the cut
 * (`hiddenCount`) advances and the head group UNMOUNTS. A DOM Selection is
 * anchored to live nodes, so unmounting the node the user highlighted destroys
 * the highlight — mid-drag, mid-copy, with no warning. Reported as: select
 * three bullets of a reply while a subagent runs, and the highlight vanishes.
 *
 * The fix is to treat a live highlight as a reason not to hide MORE content.
 * It is deliberately narrow: holding refuses to advance the cut, it never drags
 * the cut backwards, never pins the scroll position, and never survives the
 * release of the mouse. The budget resumes advancing the moment the user
 * collapses the selection.
 */

/** The transcript scroller; selections elsewhere (composer, sidebar) don't count. */
export function selectionWithin(root: HTMLElement | null): boolean {
  if (!root) {
    return false
  }

  const selection = window.getSelection()

  if (!selection || selection.isCollapsed || selection.rangeCount === 0) {
    return false
  }

  // toString() is the honest test for "is anything actually highlighted" —
  // a non-collapsed range across zero-width nodes still reads as empty.
  if (selection.toString().length === 0) {
    return false
  }

  const anchor = selection.anchorNode
  const focus = selection.focusNode

  // Either end inside the transcript is enough: a drag that started on a
  // message and ran off its end still means "I am copying this message".
  return Boolean((anchor && root.contains(anchor)) || (focus && root.contains(focus)))
}

export interface HoldWindowInput {
  /** The cut the budget wants to apply now. */
  next: number
  /** The cut currently rendered. */
  previous: number
  root: HTMLElement | null
}

/**
 * The cut to actually render. Returns `next` unless advancing it would hide
 * content while the reader holds a highlight in the transcript, in which case
 * the previous cut stands until the selection is released.
 */
export function holdWindowForSelection({ next, previous, root }: HoldWindowInput): number {
  // Not advancing: nothing can be unmounted, so there is nothing to protect.
  // Checked before touching the Selection API so the common path stays free.
  if (next <= previous) {
    return next
  }

  return selectionWithin(root) ? previous : next
}
