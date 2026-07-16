/** Compatibility helpers for Desktop session lists served by older remote backends. */

function missingSessionListEndpoint(error: unknown): boolean {
  const message = error instanceof Error ? error.message : String(error || '')

  return /(?:^|\s)404(?::|\s)/.test(message) || message.includes('endpoint is likely missing')
}

function normalizeRemoteSessionList(data: any, profile: string): any {
  const rawRows = Array.isArray(data?.sessions) ? data.sessions : Array.isArray(data?.data) ? data.data : []
  const rows = rawRows.map(row => {
    if (!row || typeof row !== 'object' || row.profile || !profile || profile === 'all') {
      return row
    }

    return {
      ...row,
      profile,
      is_default_profile: profile === 'default'
    }
  })
  const total = Number.isFinite(data?.total) ? Number(data.total) : rows.length + (data?.has_more ? 1 : 0)
  const profileTotals =
    data?.profile_totals && typeof data.profile_totals === 'object'
      ? data.profile_totals
      : profile && profile !== 'all'
        ? { [profile]: total }
        : {}

  return {
    ...data,
    sessions: rows,
    total,
    profile_totals: profileTotals
  }
}

async function fetchRemoteSessionListWithFallback(
  fetchPath: (path: string) => Promise<any>,
  aggregatePath: string,
  legacyPath: string,
  profile: string
): Promise<any> {
  try {
    return await fetchPath(aggregatePath)
  } catch (error) {
    if (!missingSessionListEndpoint(error)) {
      throw error
    }
  }

  return normalizeRemoteSessionList(await fetchPath(legacyPath), profile)
}

export { fetchRemoteSessionListWithFallback, missingSessionListEndpoint, normalizeRemoteSessionList }
