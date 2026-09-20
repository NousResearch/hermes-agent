export function matchesSessionTags(session: { tags?: string[] }, selected: readonly string[]): boolean {
  return !selected.length || selected.some(tag => session.tags?.includes(tag))
}
