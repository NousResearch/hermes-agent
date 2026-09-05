import { atom } from 'nanostores'

import type { AgentOverview, OverviewSource } from '@/global'

interface OverviewCacheState {
  data?: OverviewState
  loading: boolean
  error: string | null
}

export function createOverviewCache(fetch: (options?: { force?: boolean }) => Promise<AgentOverview>) {
  const state = atom<OverviewCacheState>({ loading: false, error: null })
  let users = 0
  let generation = 0
  let failures = 0
  let pending = false
  let timer: ReturnType<typeof setTimeout> | undefined
  let cancelRead: (() => void) | undefined

  const refresh = async (options?: { force?: boolean }): Promise<void> => {
    if (pending) {
      return
    }

    clearTimeout(timer)
    const token = ++generation
    pending = true
    const old = state.get()

    if (!old.data) {
      state.set({ ...old, loading: true })
    }

    let deadline: ReturnType<typeof setTimeout> | undefined

    try {
      const snapshot = await Promise.race([
        fetch(options),
        new Promise<never>((_, reject) => {
          cancelRead = () => reject(new Error('overview-hidden'))
          deadline = setTimeout(() => reject(new Error('Agent overview read timed out. Retry to reconnect.')), 20_000)
        })
      ])

      if (token !== generation) {
        return
      }

      failures = 0
      const data = mergeOverview(state.get().data, snapshot)
      const previous = state.get()

      if (
        previous.error ||
        previous.loading ||
        data.rows !== previous.data?.rows ||
        data.sources !== previous.data?.sources
      ) {
        state.set({ data, error: null, loading: false })
      }
    } catch (error) {
      if (token !== generation) {
        return
      }

      failures++
      const previous = state.get()
      state.set({
        data: previous.data && {
          ...previous.data,
          rows: previous.data.rows.map(staleRow),
          sources: previous.data.sources.map(unreachableSource)
        },
        loading: false,
        error: String(error instanceof Error ? error.message : error).slice(0, 240)
      })
    } finally {
      clearTimeout(deadline)

      if (token === generation) {
        pending = false
        cancelRead = undefined

        if (users > 0) {
          // Exhaust the fast retries, not recovery. Keep a quiet visible-only
          // probe so transient bridge outages heal without requiring Retry.
          timer = setTimeout(() => void refresh(), failures < 3 ? 5_000 : 60_000)
        }
      }
    }
  }

  return {
    state,
    refresh,
    watch() {
      if (++users === 1) {
        failures = 0
        void refresh()
      }

      return () => {
        if (--users === 0) {
          clearTimeout(timer)
          generation++
          cancelRead?.()
          cancelRead = undefined
          pending = false

          if (state.get().loading) {
            state.set({ ...state.get(), loading: false })
          }
        }
      }
    }
  }
}

export const overviewCache = createOverviewCache(async options => {
  const fetch = window.hermesDesktop?.getAgentOverview

  if (!fetch) {
    throw new Error('Agent overview is unavailable in this desktop version.')
  }

  return fetch(options)
})

export type Attention = 'needs-you' | 'working' | 'unread' | 'idle'
export interface AgentRow {
  key: string
  connectionId: string
  sourceLabel: string
  profile: string
  owner: string
  description: string
  id: string
  resolvedId: string
  runtimeId?: string
  history: 'known-empty' | 'known-history' | 'unknown'
  title: string
  preview: string
  model: string
  provider: string
  platform: string
  attention: Attention
  // Presentation only: retained on stale rows, never runtime/action authority.
  lastKnownAttention?: 'working' | 'needs-you'
  canonical: boolean
  stale: boolean
  lastActive: number
}

// Stale retained work belongs outside live attention counts. The visible stale
// label distinguishes unknown status from confirmed idle (never 'completed').
const staleRow = (row: AgentRow): AgentRow => ({
  ...row,
  stale: true,
  runtimeId: undefined,
  lastKnownAttention:
    row.attention === 'working' || row.attention === 'needs-you' ? row.attention : row.lastKnownAttention,
  attention: row.attention === 'unread' ? 'unread' : 'idle'
})

export interface OverviewState {
  rows: AgentRow[]
  sources: OverviewSource[]
  fetchedAt: number
}

// A bridge failure says nothing about the sources themselves, but it does end
// their freshness claim: keep last-known counts, drop "ready"/"complete" until
// a successful read replaces the source from a fresh snapshot.
const unreachableSource = (source: OverviewSource): OverviewSource =>
  source.state === 'ready' || source.state === 'partial' || source.complete
    ? { ...source, state: 'offline', complete: false }
    : source

export const agentOwnerKey = (connectionId: string, profile: string, id: string): string =>
  JSON.stringify([connectionId, profile, id])

function sourceRows(source: OverviewSource): AgentRow[] {
  const aliases = new Map<string, string>()

  // A canonical registry root owns its compression tip, never vice versa.
  for (const item of [...source.canonical, ...source.sessions, ...source.live]) {
    const key = agentOwnerKey(source.connectionId, item.profile, item.id)

    if (!aliases.has(key)) {
      aliases.set(key, key)
    }

    if (item.resolved_id) {
      aliases.set(agentOwnerKey(source.connectionId, item.profile, item.resolved_id), aliases.get(key)!)
    }
  }

  const rows = new Map<string, AgentRow>()
  const profiles = new Map(source.profiles.map(profile => [profile.name, profile]))

  for (const [items, canonical] of [
    [source.sessions, false],
    [source.canonical, true],
    [source.live, false]
  ] as const) {
    for (const item of items) {
      const key = aliases.get(agentOwnerKey(source.connectionId, item.profile, item.id))!
      const previous = rows.get(key)
      const profile = profiles.get(item.profile)
      const status = item.status
      const count = item.message_count

      // Counts are authority; blank previews and lazy runtime metadata are not.
      // Live snapshots may report zero while persisted history is already known.
      const history: AgentRow['history'] =
        previous?.history === 'known-history' || (Number.isFinite(count) && count! > 0)
          ? 'known-history'
          : count === 0
            ? 'known-empty'
            : (previous?.history ?? 'unknown')

      const attention: Attention =
        status === 'waiting'
          ? 'needs-you'
          : status === 'working' || status === 'starting'
            ? 'working'
            : status === 'idle'
              ? item.unread || previous?.attention === 'unread'
                ? 'unread'
                : 'idle'
              : (previous?.attention ?? (item.unread ? 'unread' : 'idle'))

      rows.set(key, {
        key,
        connectionId: source.connectionId,
        sourceLabel: source.label,
        profile: item.profile,
        owner: profile?.display_name || item.profile,
        description: profile?.description || '',
        id: JSON.parse(key)[2] as string,
        resolvedId: item.resolved_id || previous?.resolvedId || item.id,
        runtimeId: item.runtime_id || previous?.runtimeId,
        history,
        title: item.title || previous?.title || '',
        preview: item.preview || previous?.preview || '',
        model: item.model || previous?.model || '',
        provider: item.provider || previous?.provider || '',
        platform: item.source || previous?.platform || '',
        attention,
        canonical: canonical || previous?.canonical || false,
        stale: !source.complete && source.state !== 'ready',
        // Backend Unix seconds: message/heartbeat activity or actual session start.
        // A lazy live summary must not replace newer persisted-message activity.
        lastActive: Math.max(item.last_active || item.started_at || 0, previous?.lastActive || 0)
      })
    }
  }

  return [...rows.values()]
}

export function mergeOverview(previous: OverviewState | undefined, snapshot: AgentOverview): OverviewState {
  const oldSources = new Map(previous?.sources.map(source => [source.connectionId, source]))
  const oldRows = new Map(previous?.rows.map(row => [row.key, row]))

  const sources = snapshot.sources.map(source => {
    const old = oldSources.get(source.connectionId)

    return old && !source.complete
      ? { ...source, total: Math.max(old.total, source.total), offset: Math.max(old.offset, source.offset) }
      : source
  })

  const rows = sources.flatMap(source => {
    const fresh = sourceRows(source)
    const keys = new Set(fresh.map(row => row.key))

    const retained = source.complete
      ? []
      : (previous?.rows ?? [])
          .filter(row => row.connectionId === source.connectionId && !keys.has(row.key))
          .map(staleRow)

    return [...fresh.map(row => (row.stale ? staleRow(row) : row)), ...retained].map(row => {
      const old = oldRows.get(row.key)

      if (old?.history === 'known-history' && row.history !== 'known-history') {
        row = { ...row, history: 'known-history' }
      }

      return old && JSON.stringify(old) === JSON.stringify(row) ? old : row
    })
  })

  return {
    rows:
      previous && rows.length === previous.rows.length && rows.every((row, i) => row === previous.rows[i])
        ? previous.rows
        : rows,
    sources: previous && JSON.stringify(sources) === JSON.stringify(previous.sources) ? previous.sources : sources,
    fetchedAt: snapshot.fetchedAt
  }
}
