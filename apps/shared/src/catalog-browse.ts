/** Shared with the docs catalog: taxonomy order is independent of popularity. Keep keys in sync
 *  with CATALOG_CATEGORIES in hermes_cli/plugin_catalog.py and website/scripts/extract-plugins.py. */
export const PLUGIN_CATEGORIES: Record<string, { label: string; icon: string; blurb: string }> = {
  desktop: { label: 'Desktop', icon: '🖥️', blurb: 'Panes, tabs and views for Hermes Desktop' },
  memory: { label: 'Memory', icon: '🧠', blurb: 'Memory providers and context engines' },
  platform: { label: 'Platforms', icon: '💬', blurb: 'Messaging and channel adapters' },
  web: { label: 'Web & Browser', icon: '🌐', blurb: 'Search backends, extraction and browser control' },
  tools: { label: 'Tools', icon: '🛠️', blurb: 'New tools the agent can call' },
  voice: { label: 'Voice', icon: '🎙️', blurb: 'Speech, TTS and realtime audio' },
  automation: { label: 'Automation', icon: '⏱️', blurb: 'Hooks, wake triggers and session automation' },
  models: { label: 'Models', icon: '✨', blurb: 'Model and inference providers' },
  general: { label: 'General', icon: '📦', blurb: 'Plugins that span several areas' }
}

export const PLUGIN_CATEGORY_ORDER = Object.keys(PLUGIN_CATEGORIES)
export type PluginCatalogSort = 'stars' | 'newest' | 'updated'

export function sortCatalogPlugins<T extends { name: string; addedAt?: string | null; updatedAt?: string | null }>(entries: T[], sort: PluginCatalogSort): T[] {
  // The published snapshot already sorts stars descending, then name. Preserve
  // its order, including ties, rather than reinterpreting missing star counts.
  if (sort === 'stars') return entries
  const field = sort === 'newest' ? 'addedAt' : 'updatedAt'
  const date = (value?: string | null) => {
    const ms = value ? Date.parse(value) : NaN
    return Number.isFinite(ms) ? ms : -Infinity
  }
  return [...entries].sort((a, b) => date(b[field]) - date(a[field]) || a.name.localeCompare(b.name))
}

export function groupCatalogPlugins<T extends { category: string }>(entries: T[]): [string, T[]][] {
  const buckets = new Map<string, T[]>()
  for (const entry of entries) {
    const key = Object.hasOwn(PLUGIN_CATEGORIES, entry.category) ? entry.category : 'general'
    const bucket = buckets.get(key) ?? []
    bucket.push(entry)
    buckets.set(key, bucket)
  }
  return PLUGIN_CATEGORY_ORDER.filter(key => buckets.has(key)).map(key => [key, buckets.get(key)!])
}
