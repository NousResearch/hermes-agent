// Dependency-free canonical profile key. Keep this module free of stores and
// app imports so identity helpers can depend on it without creating cycles.
export function normalizeProfileKey(name: string | null | undefined): string {
  const value = (name ?? '').trim()

  return value || 'default'
}
