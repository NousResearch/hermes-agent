import { Codecs, persistentAtom } from '@/lib/persisted'

const STORAGE_KEY = 'hermes.desktop.collapsed-providers'

/** Set of provider slugs whose model groups are currently collapsed in the
 *  model picker dropdown. Persisted to localStorage globally — this is a
 *  presentation-layer preference, not a per-profile setting. */
export const $collapsedProviders = persistentAtom<string[]>(
  STORAGE_KEY,
  [],
  Codecs.stringArray
)

/** Toggle a provider slug in/out of the collapsed set. */
export function toggleCollapsedProvider(slug: string): void {
  const current = $collapsedProviders.get()
  $collapsedProviders.set(
    current.includes(slug)
      ? current.filter(s => s !== slug)
      : [...current, slug]
  )
}

/** Remove slugs from the collapsed set that no longer match any current
 *  provider. Call this after provider list changes (e.g. Refresh Models,
 *  plugin install/uninstall) to prevent unbounded growth. */
export function pruneStaleCollapsed(slugs: string[]): void {
  const valid = new Set(slugs)
  const next = $collapsedProviders.get().filter(s => valid.has(s))

  if (next.length !== $collapsedProviders.get().length) {
    $collapsedProviders.set(next)
  }
}
