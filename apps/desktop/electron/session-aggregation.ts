export interface ProfileSessionListError {
  error: string
  profile: string
}

interface MergeProfileSessionPagesOptions {
  fetchPrimary: (searchParams: URLSearchParams) => Promise<unknown>
  fetchRemote: (profile: string, searchParams: URLSearchParams) => Promise<unknown>
  remoteProfiles: string[]
  searchParams: URLSearchParams
}

const rowsOf = (data: unknown): Array<Record<string, unknown>> => {
  if (!data || typeof data !== 'object') {
    return []
  }

  const sessions = (data as { sessions?: unknown }).sessions

  return Array.isArray(sessions) ? sessions : []
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error)
}

function listErrors(data: unknown): ProfileSessionListError[] {
  if (!data || typeof data !== 'object' || !Array.isArray((data as { errors?: unknown }).errors)) {
    return []
  }

  return (data as { errors: unknown[] }).errors.flatMap(value => {
    if (!value || typeof value !== 'object') {
      return []
    }

    const entry = value as { error?: unknown; profile?: unknown }

    if (typeof entry.profile !== 'string' || typeof entry.error !== 'string') {
      return []
    }

    return [{ error: entry.error, profile: entry.profile }]
  })
}

export async function mergeProfileSessionPages({
  fetchPrimary,
  fetchRemote,
  remoteProfiles,
  searchParams
}: MergeProfileSessionPagesOptions): Promise<Record<string, unknown>> {
  const limit = Math.max(1, Number(searchParams.get('limit')) || 20)
  const offset = Math.max(0, Number(searchParams.get('offset')) || 0)
  const order = searchParams.get('order') === 'created' ? 'started_at' : 'last_active'
  let base: Record<string, unknown>

  try {
    const primary = await fetchPrimary(searchParams)

    if (!primary || typeof primary !== 'object' || Array.isArray(primary)) {
      throw new Error('Invalid primary session response')
    }

    base = primary as Record<string, unknown>
  } catch (error) {
    base = {
      errors: [{ error: errorMessage(error), profile: 'default' }],
      profile_totals: {},
      sessions: [],
      total: 0
    }
  }

  const remoteSet = new Set(remoteProfiles)
  const errors = listErrors(base).filter(error => !remoteSet.has(error.profile))
  const remoteParams = new URLSearchParams(searchParams)

  remoteParams.set('limit', String(limit + offset))
  remoteParams.set('offset', '0')

  const merged = rowsOf(base).filter(session => !remoteSet.has(session.profile as string))
  const profileTotals = { ...((base.profile_totals as Record<string, number> | undefined) || {}) }

  let total =
    (Number(base.total) || 0) - remoteProfiles.reduce((count, profile) => count + (profileTotals[profile] || 0), 0)

  await Promise.all(
    remoteProfiles.map(async profile => {
      let list: unknown

      try {
        list = await fetchRemote(profile, remoteParams)

        if (!list || typeof list !== 'object' || Array.isArray(list)) {
          throw new Error('Invalid remote session response')
        }
      } catch (error) {
        delete profileTotals[profile]
        errors.push({ error: errorMessage(error), profile })

        return
      }

      const rows = rowsOf(list)

      merged.push(...rows)
      profileTotals[profile] = Number((list as { total?: unknown })?.total) || rows.length
      total += profileTotals[profile]
      errors.push(...listErrors(list))
    })
  )

  const recency = (session: Record<string, unknown>) => Number(session[order] ?? session.started_at ?? 0)

  merged.sort((left, right) => recency(right) - recency(left))
  errors.sort((left, right) => left.profile.localeCompare(right.profile))

  return {
    ...base,
    errors,
    profile_totals: profileTotals,
    sessions: merged.slice(offset, offset + limit),
    total
  }
}
