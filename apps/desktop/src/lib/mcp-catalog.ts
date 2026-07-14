import type { McpCatalogEntry } from '@/types/hermes'

export type ConnectorPrimaryActionKind = 'connect' | 'install' | 'installed'

export function connectorIdentityKey(entry: Pick<McpCatalogEntry, 'icon' | 'name'>): string {
  return entry.icon?.trim() || entry.name
}

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

  return entry.auth_type === 'oauth' && ['http', 'sse'].includes(entry.transport) ? 'connect' : 'install'
}

export function connectorSetupSummary(
  entry: Pick<McpCatalogEntry, 'auth_type' | 'needs_install' | 'required_env' | 'setup_steps' | 'transport'>
): string {
  const parts: string[] = []
  const stepCount = entry.setup_steps.length || entry.required_env.filter(env => env.required).length

  if (stepCount > 0) {
    parts.push(`${stepCount} setup step${stepCount === 1 ? '' : 's'}`)
  }

  if (entry.auth_type === 'oauth' && entry.transport === 'http') {
    parts.push('Browser OAuth')
  } else if (entry.auth_type === 'api_key' || entry.required_env.length > 0) {
    parts.push('Requires credentials')
  } else if (entry.needs_install) {
    parts.push('Local build')
  }

  return parts.join(' · ')
}
