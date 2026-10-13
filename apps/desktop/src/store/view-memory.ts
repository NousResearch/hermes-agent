import { atom } from 'nanostores'

/**
 * Per-view memory of the last sub-state a view was on — the Capabilities pane,
 * the Settings tab — so reopening a view lands where it was left instead of
 * snapping back to its default.
 *
 * IN-MEMORY for the window only: never persisted (localStorage would leak one
 * window's context into the next, and this is presentation, not preference),
 * and keyed per view so no view ever reads another's value. A view writes its
 * key whenever its sub-state changes and reads it only when the route names
 * none — an explicit `?tab=` (or a deep link carrying one) always wins.
 */
export const $viewMemory = atom<Record<string, string>>({})

/** Remembered sub-state for `view`, or null when it has none yet. */
export function rememberedViewSubState(view: string): string | null {
  return $viewMemory.get()[view] ?? null
}

/** Remember `value` as `view`'s sub-state. No-op when unchanged, so a view's
 *  render-driven write never churns the atom. */
export function rememberViewSubState(view: string, value: string): void {
  const current = $viewMemory.get()

  if (current[view] === value) {
    return
  }

  $viewMemory.set({ ...current, [view]: value })
}

/** Drop every remembered sub-state. Test-only. */
export function resetViewMemory(): void {
  $viewMemory.set({})
}
