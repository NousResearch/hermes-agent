import { atom } from 'nanostores'

/**
 * Manual collapse choices for the composer status stack's `StatusSection`
 * groups, keyed `<sessionId>:<groupKey>`. Store-driven renders (tool.start /
 * tool.complete ticks) remount the stack's lane, which resets anything kept in
 * component state — so a todo group the user collapsed re-opened on the next
 * tick. Reading the choice through this store keeps it outside the remount
 * boundary; visibility stays a user choice, matching the files-panel precedent
 * (reactive unhide must not expand a user-collapsed rail). In-memory only: a
 * reload falls back to each section's default, and sessions never share keys.
 */
export const $statusSectionCollapsedByKey = atom<Record<string, boolean>>({})

export function setStatusSectionCollapsed(key: string, collapsed: boolean): void {
  const current = $statusSectionCollapsedByKey.get()

  if (current[key] === collapsed) {
    return
  }

  $statusSectionCollapsedByKey.set({ ...current, [key]: collapsed })
}
