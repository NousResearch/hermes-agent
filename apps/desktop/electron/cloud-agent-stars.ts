interface ConnectionDocument {
  starredCloudAgentIds?: unknown
  [key: string]: unknown
}

export function starredCloudAgentIds(raw: unknown): string[] {
  if (!Array.isArray(raw)) {
    return []
  }

  const seen = new Set<string>()
  const ids: string[] = []

  for (const value of raw) {
    const id = String(value ?? '').trim()

    if (id && !seen.has(id)) {
      seen.add(id)
      ids.push(id)
    }
  }

  return ids
}

export function setCloudAgentStarred(config: ConnectionDocument, rawId: unknown, starred: boolean): ConnectionDocument {
  const id = String(rawId ?? '').trim()

  if (!id) {
    throw new Error('Cloud agent id is required.')
  }

  const ids = starredCloudAgentIds(config.starredCloudAgentIds)
  const nextIds = starred ? (ids.includes(id) ? ids : [...ids, id]) : ids.filter(candidate => candidate !== id)

  return { ...config, starredCloudAgentIds: nextIds }
}
