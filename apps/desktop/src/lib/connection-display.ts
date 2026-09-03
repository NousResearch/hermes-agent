import type { DesktopRegistryConnection } from '@/global'

export const CONNECTION_SEARCH_THRESHOLD = 8

const connectionLabelCollator = new Intl.Collator(undefined, {
  numeric: true,
  sensitivity: 'base'
})

/** Local stays anchored; every other gateway has one stable, human-readable order. */
export function sortConnectionsForDisplay<T extends Pick<DesktopRegistryConnection, 'id' | 'kind' | 'label'>>(
  connections: readonly T[]
): T[] {
  return [...connections].sort((left, right) => {
    const localOrder = Number(right.kind === 'local') - Number(left.kind === 'local')

    return (
      localOrder ||
      connectionLabelCollator.compare(left.label, right.label) ||
      connectionLabelCollator.compare(left.id, right.id)
    )
  })
}

function normalizeSearchText(value: string): string {
  return value.normalize('NFKD').replace(/\p{M}/gu, '').toLocaleLowerCase()
}

/** Search non-secret details users can see or reasonably remember about a gateway. */
export function connectionMatchesQuery(
  connection: DesktopRegistryConnection,
  query: string,
  aliases: readonly string[] = []
): boolean {
  const needles = normalizeSearchText(query).trim().split(/\s+/).filter(Boolean)

  if (needles.length === 0) {
    return true
  }

  const haystack = [
    connection.label,
    connection.kind,
    connection.url,
    connection.host,
    connection.user,
    connection.port == null ? null : String(connection.port),
    connection.org,
    connection.remoteProfile,
    ...aliases
  ]
    .filter((value): value is string => Boolean(value))
    .join(' ')

  const normalizedHaystack = normalizeSearchText(haystack)

  return needles.every(needle => normalizedHaystack.includes(needle))
}

/** Human-readable, non-secret endpoint for on-demand gateway details. */
export function connectionEndpoint(connection: DesktopRegistryConnection): null | string {
  if (connection.kind === 'ssh') {
    const host = connection.host?.trim()

    if (!host) {
      return null
    }

    const authority = `${connection.user?.trim() ? `${connection.user.trim()}@` : ''}${host}`

    return connection.port == null ? authority : `${authority}:${connection.port}`
  }

  return connection.url?.trim() || null
}

/** Full gateway identity for a hover tip without keeping technical routing in chrome. */
export function connectionTooltip(connection: DesktopRegistryConnection): string {
  const endpoint = connectionEndpoint(connection)

  return endpoint ? `${connection.label}\n${endpoint}` : connection.label
}
