export type MessagingSortMode = 'default' | 'name-asc' | 'name-desc' | 'status'
export type MessagingStatusFilter = 'all' | 'attention' | 'connected' | 'error' | 'inactive'

export interface MessagingListPlatform {
  description?: null | string
  enabled?: boolean
  id: string
  name: string
  state?: null | string
}

const STATUS_ALIASES: Record<string, MessagingStatusFilter> = {
  all: 'all',
  attention: 'attention',
  warn: 'attention',
  warning: 'attention',
  connected: 'connected',
  good: 'connected',
  error: 'error',
  bad: 'error',
  inactive: 'inactive',
  disabled: 'inactive',
  muted: 'inactive',
  off: 'inactive'
}

const SORT_ALIASES: Record<string, MessagingSortMode> = {
  default: 'default',
  name: 'name-asc',
  'name-asc': 'name-asc',
  'name-desc': 'name-desc',
  status: 'status'
}

const STATUS_ORDER: Record<Exclude<MessagingStatusFilter, 'all'>, number> = {
  error: 0,
  attention: 1,
  connected: 2,
  inactive: 3
}

export function messagingStatusFilter(platform: MessagingListPlatform): Exclude<MessagingStatusFilter, 'all'> {
  if (platform.enabled === false) {
    return 'inactive'
  }

  if (platform.state === 'connected') {
    return 'connected'
  }

  if (platform.state === 'fatal' || platform.state === 'startup_failed') {
    return 'error'
  }

  return 'attention'
}

export function parseMessagingQuery(query: string): {
  sort: MessagingSortMode
  status: MessagingStatusFilter
  text: string
} {
  let sort: MessagingSortMode = 'default'
  let status: MessagingStatusFilter = 'all'
  const textParts: string[] = []

  for (const part of query.trim().split(/\s+/).filter(Boolean)) {
    const separator = part.indexOf(':')

    if (separator > 0) {
      const key = part.slice(0, separator).toLowerCase()
      const value = part.slice(separator + 1).toLowerCase()

      if (key === 'status' && STATUS_ALIASES[value]) {
        status = STATUS_ALIASES[value]
        continue
      }

      if (key === 'sort' && SORT_ALIASES[value]) {
        sort = SORT_ALIASES[value]
        continue
      }
    }

    textParts.push(part)
  }

  return { sort, status, text: textParts.join(' ').trim().toLowerCase() }
}

export function setMessagingQueryToken(
  query: string,
  key: 'sort' | 'status',
  value: MessagingSortMode | MessagingStatusFilter
): string {
  const parts = query.trim().split(/\s+/).filter(Boolean)
  const next = parts.filter(part => !part.toLowerCase().startsWith(`${key}:`))
  const isDefault = (key === 'status' && value === 'all') || (key === 'sort' && value === 'default')

  if (!isDefault) {
    next.push(`${key}:${value}`)
  }

  return next.join(' ')
}

export function filterAndSortMessagingPlatforms<T extends MessagingListPlatform>(
  platforms: readonly T[],
  query: string,
  stateLabel: (state: null | string | undefined) => string
): T[] {
  const parsed = parseMessagingQuery(query)
  const filtered = platforms.filter(platform => {
    if (parsed.status !== 'all' && messagingStatusFilter(platform) !== parsed.status) {
      return false
    }

    if (!parsed.text) {
      return true
    }

    return [platform.id, platform.name, platform.description, platform.state, stateLabel(platform.state)]
      .filter(Boolean)
      .some(value => String(value).toLowerCase().includes(parsed.text))
  })

  if (parsed.sort === 'default') {
    return filtered
  }

  return [...filtered].sort((a, b) => {
    if (parsed.sort === 'status') {
      const byStatus = STATUS_ORDER[messagingStatusFilter(a)] - STATUS_ORDER[messagingStatusFilter(b)]

      if (byStatus !== 0) {
        return byStatus
      }
    }

    const byName = a.name.localeCompare(b.name, undefined, { sensitivity: 'base' })

    return parsed.sort === 'name-desc' ? -byName : byName
  })
}
