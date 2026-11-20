/**
 * A flat, pointer-following drag chip — the shared "what am I holding"
 * affordance for in-app pointer drags. Plain DOM (no React) so it survives a
 * pointer-capture drag without re-renders and tears down synchronously on Esc.
 *
 * Flat by design: a solid app surface with the dragged item's label, no
 * border / radius / shadow, dimmed — it copies the real row/tab it represents
 * rather than reading as a separate pill. Any pointer drag whose source does
 * not stay visibly "held" can reuse this (the drag primitive in
 * `pane-shell/tree/renderer/drag-session.ts`, and anything built on it).
 */

/** How far (px) the chip trails the pointer so it never sits under the cursor. */
const OFFSET_X = 14
const OFFSET_Y = 12

export interface DragGhost {
  /** Reposition the chip near the current pointer point. */
  moveTo(x: number, y: number): void
  /** Remove the chip from the DOM. Idempotent. */
  destroy(): void
}

export function createDragGhost(label: string): DragGhost {
  const el = document.createElement('div')

  el.textContent = label
  el.style.cssText =
    'position:fixed;left:0;top:0;z-index:9999;pointer-events:none;max-width:16rem;overflow:hidden;' +
    'text-overflow:ellipsis;white-space:nowrap;padding:0.25rem 0.625rem;opacity:0.6;' +
    'background:var(--ui-sidebar-surface-background,var(--dt-card));color:var(--ui-text-primary);' +
    'font-size:0.75rem;font-weight:500;will-change:transform'
  document.body.appendChild(el)

  return {
    moveTo(x, y) {
      el.style.transform = `translate3d(${x + OFFSET_X}px, ${y + OFFSET_Y}px, 0)`
    },
    destroy() {
      el.remove()
    }
  }
}
