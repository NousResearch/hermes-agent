/** The `engine` field to spread into a room record rebuilt from `sources`
 *  (storage hydration, the ui_meta projection and its merges). The mark is set
 *  once, when a room is created or edited across machines, and never cleared,
 *  so any source carrying it keeps it. */
export function stickyGroupEngine(...sources: Array<{ engine?: unknown } | null | undefined>): { engine?: 'desktop' } {
  return sources.some(source => source?.engine === 'desktop') ? { engine: 'desktop' } : {}
}
