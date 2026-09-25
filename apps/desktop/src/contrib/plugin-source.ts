import type { ContributionSource } from './types'

/** Canonical provenance vocabulary for host-scoped plugin contributions. */
export const PLUGIN_SOURCE_PREFIX = 'plugin:' as const
export type PluginSource = `${typeof PLUGIN_SOURCE_PREFIX}${string}`

export const pluginSource = (pluginId: string): PluginSource => `${PLUGIN_SOURCE_PREFIX}${pluginId}`

export function isPluginSource(source: ContributionSource | undefined): source is PluginSource {
  return source?.startsWith(PLUGIN_SOURCE_PREFIX) ?? false
}

export const pluginIdFromSource = (source: PluginSource): string => source.slice(PLUGIN_SOURCE_PREFIX.length)
