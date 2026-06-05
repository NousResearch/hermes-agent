import type { ModelOptionProvider } from '@/types/hermes'

export interface RouterHostSummary {
  label: string
  title: string
}

function normalizeHosts(value: unknown): string[] {
  if (!Array.isArray(value)) {
    return []
  }

  return value.map(host => String(host).trim()).filter(Boolean)
}

export function routerHostSummary(provider: ModelOptionProvider, model: string): RouterHostSummary | null {
  const metadata = provider.model_metadata?.[model]
  if (!metadata) {
    return null
  }

  const primary = String(metadata.router_host ?? '').trim()
  const hosts = normalizeHosts(metadata.router_hosts)
  const orderedHosts = [primary, ...hosts].filter((host, index, all) => host && all.indexOf(host) === index)

  if (orderedHosts.length === 0) {
    return null
  }

  const label = orderedHosts.length > 1 ? `${orderedHosts[0]} +${orderedHosts.length - 1}` : orderedHosts[0]
  const backend = String(metadata.router_backend ?? '').trim()
  const title = [
    `AI-Router host${orderedHosts.length > 1 ? 's' : ''}: ${orderedHosts.join(', ')}`,
    backend ? `backend: ${backend}` : ''
  ]
    .filter(Boolean)
    .join(' | ')

  return { label, title }
}
