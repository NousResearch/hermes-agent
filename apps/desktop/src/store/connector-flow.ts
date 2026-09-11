import { atom } from 'nanostores'

import { connectorAuthorizationUrl, type ConnectorRow, recordOf } from '@/lib/connector-tools'

export type ConnectorPhase = 'idle' | 'opening' | 'waiting' | 'connected' | 'timeout' | 'error' | 'skipped'
export interface ConnectorFlowRow extends ConnectorRow {
  phase: ConnectorPhase
  error?: string
}
export interface ConnectorFlowState {
  loading: boolean
  available: boolean
  rows: ConnectorFlowRow[]
  error?: string
}
export interface ConnectorFlowDeps {
  request: <T>(method: string, params: { session_id: string; connectors?: string[]; reconnect?: boolean }) => Promise<T>
  open: (url: string) => Promise<void>
  delay?: () => Promise<void>
  now?: () => number
}

/** One mounted tool offer. Only explicit user actions can mint or open links. */
export function createConnectorFlow(sessionId: string, seeds: ConnectorRow[], deps: ConnectorFlowDeps) {
  const state = atom<ConnectorFlowState>({
    loading: true,
    available: false,
    rows: seeds.map(row => ({ ...row, phase: 'idle', connected: false }))
  })

  let disposed = false
  let refreshGeneration = 0
  const attempts = new Map<string, number>()
  const delay = deps.delay ?? (() => new Promise(resolve => setTimeout(resolve, 2000)))
  const now = deps.now ?? Date.now
  const valid = (slug: string, token: number) => !disposed && attempts.get(slug) === token

  const update = (slug: string, changes: Partial<ConnectorFlowRow>) => {
    if (disposed) {
      return
    }

    refreshGeneration += 1
    const current = state.get()
    state.set({ ...current, rows: current.rows.map(row => (row.connector === slug ? { ...row, ...changes } : row)) })
  }

  const list = async () => {
    const response = await deps.request<{ available: boolean; connectors: ConnectorRow[] }>('connectors.list', {
      session_id: sessionId
    })

    if ((response.available !== true && response.available !== false) || !Array.isArray(response.connectors)) {
      throw new Error('Invalid connector status response')
    }

    return response
  }

  const refresh = async () => {
    disposed = false
    const generation = ++refreshGeneration

    try {
      const response = await list()

      if (disposed || generation !== refreshGeneration) {
        return
      }

      const current = state.get()
      // The tool may return the whole catalog or a few requested apps. Do not
      // replace a targeted offer with every app the gateway happens to know.
      const wanted = seeds.length ? seeds : response.connectors

      const rows = wanted.map((seed): ConnectorFlowRow => {
        const live = response.connectors.find(row => row.connector === seed.connector)
        const previous = current.rows.find(row => row.connector === seed.connector)
        const phase = previous?.phase ?? 'idle'

        return {
          ...seed,
          ...live,
          connector: seed.connector,
          enabled: response.available && live?.enabled !== false && !!live,
          connected: live?.connected === true,
          phase:
            phase === 'skipped'
              ? phase
              : live?.connected
                ? 'connected'
                : ['opening', 'waiting'].includes(phase)
                  ? phase
                  : 'idle'
        }
      })

      state.set({ loading: false, available: response.available, rows })
    } catch {
      if (!disposed && generation === refreshGeneration) {
        state.set({ ...state.get(), loading: false, error: 'status' })
      }
    }
  }

  const wait = async (slug: string, token: number) => {
    const deadline = now() + 120000
    let failures = 0

    while (valid(slug, token) && now() < deadline) {
      await delay()

      if (!valid(slug, token)) {
        return
      }

      try {
        const response = await list()

        if (!valid(slug, token)) {
          return
        }

        if (!response.available) {
          update(slug, { phase: 'error', error: 'unavailable' })

          return
        }

        const row = response.connectors.find(row => row.connector === slug)

        if (row?.connected) {
          update(slug, { ...row, phase: 'connected', error: undefined })

          return
        }

        if (row?.enabled === false || !row) {
          update(slug, { phase: 'error', error: 'unavailable' })

          return
        }

        failures = 0
      } catch {
        if (++failures >= 3) {
          update(slug, { phase: 'error', error: 'status' })

          return
        }
      }
    }

    if (valid(slug, token)) {
      update(slug, { phase: 'timeout' })
    }
  }

  const connect = async (slug: string) => {
    const row = state.get().rows.find(row => row.connector === slug)

    if (
      disposed ||
      !state.get().available ||
      state.get().error ||
      !row ||
      row.enabled === false ||
      ['opening', 'waiting'].includes(row.phase)
    ) {
      return
    }

    const token = (attempts.get(slug) ?? 0) + 1
    attempts.set(slug, token)
    update(slug, { phase: 'opening', error: undefined })

    try {
      const response = await deps.request<{ results: unknown[] }>('connectors.connect', {
        session_id: sessionId,
        connectors: [slug],
        reconnect: ['expired', 'revoked'].includes(row.connectionStatus ?? '')
      })

      if (!valid(slug, token)) {
        return
      }

      const entry = (response.results ?? []).map(recordOf).find(result => result.connector === slug)

      if (entry?.status === 'active') {
        update(slug, { phase: 'idle' })
        await refresh()

        return
      }

      const url = connectorAuthorizationUrl(entry?.connect_url)

      if (entry?.status !== 'initiated' || !url) {
        throw new Error('Authorization unavailable')
      }

      await deps.open(url)

      if (!valid(slug, token)) {
        return
      }

      update(slug, { phase: 'waiting' })
      await wait(slug, token)
    } catch {
      if (valid(slug, token)) {
        update(slug, { phase: 'error', error: 'connect' })
      }
    }
  }

  return {
    state,
    refresh,
    connect,
    keepWaiting: async (slug: string) => {
      const row = state.get().rows.find(row => row.connector === slug)

      if (
        disposed ||
        !row ||
        row.enabled === false ||
        !state.get().available ||
        !['timeout', 'error'].includes(row.phase)
      ) {
        return
      }

      const token = (attempts.get(slug) ?? 0) + 1
      attempts.set(slug, token)
      update(slug, { phase: 'waiting', error: undefined })
      await wait(slug, token)
    },
    skip: (slug: string) => {
      attempts.set(slug, (attempts.get(slug) ?? 0) + 1)
      update(slug, { phase: 'skipped', error: undefined })
    },
    dispose: () => {
      disposed = true
      refreshGeneration += 1

      for (const [slug, token] of attempts) {
        attempts.set(slug, token + 1)
      }
    }
  }
}
