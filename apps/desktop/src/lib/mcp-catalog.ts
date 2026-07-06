import type { McpCatalogEntry } from '@/types/hermes'

export type ConnectorPrimaryActionKind = 'connect' | 'install' | 'installed'

export function connectorDisplayName(entry: Pick<McpCatalogEntry, 'display_name' | 'name'>): string {
  const displayName = entry.display_name?.trim()

  if (displayName) {
    return displayName
  }

  return entry.name
    .split(/[-_\s]+/)
    .filter(Boolean)
    .map(part => part.charAt(0).toUpperCase() + part.slice(1))
    .join(' ')
}

export function connectorPrimaryActionKind(
  entry: Pick<McpCatalogEntry, 'auth_type' | 'installed' | 'transport'>
): ConnectorPrimaryActionKind {
  if (entry.installed) {
    return 'installed'
  }

  return entry.auth_type === 'oauth' && entry.transport === 'http' ? 'connect' : 'install'
}
