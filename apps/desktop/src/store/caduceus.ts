import { atom } from 'nanostores'

import { $gateway } from './gateway'

export interface CaduceusTier {
  provider: string
  model: string
}

export interface CaduceusLocalModel {
  id: string
  card: string
  maxContext: number
  maxSlots: number
  default: boolean
}

export interface CaduceusLocalLoaded {
  model: string
  profile: string
  slots: number
}

export interface CaduceusLocal {
  enabled: boolean
  models: CaduceusLocalModel[]
  loaded: CaduceusLocalLoaded | null
  defaultWorker: string
}

export interface CaduceusSummary {
  enabled: boolean
  orchestrator: CaduceusTier
  worker: CaduceusTier
  budget: number | null
  effort?: string
  split?: boolean
  routerEnabled: boolean
  local: CaduceusLocal
}

const EMPTY_TIER: CaduceusTier = { model: '', provider: '' }
const EMPTY_LOCAL: CaduceusLocal = { defaultWorker: '', enabled: false, loaded: null, models: [] }

const DEFAULT_STATE: CaduceusSummary = {
  budget: null,
  effort: 'high',
  enabled: false,
  local: { ...EMPTY_LOCAL },
  orchestrator: { ...EMPTY_TIER },
  routerEnabled: false,
  split: false,
  worker: { ...EMPTY_TIER }
}

/** Live Caduceus mode state for the active session (mirrors the backend). */
export const $caduceus = atom<CaduceusSummary>(DEFAULT_STATE)

/** Whether the full Orchestration Theater overlay is open. */
export const $caduceusTheaterOpen = atom(false)

function normalizeTier(t: unknown): CaduceusTier {
  const o = (t ?? {}) as Record<string, unknown>
  return { model: String(o.model ?? ''), provider: String(o.provider ?? '') }
}

function normalizeLocal(raw: unknown, fallback: CaduceusLocal): CaduceusLocal {
  if (!raw || typeof raw !== 'object') {
    return fallback
  }
  const o = raw as Record<string, unknown>
  const models = Array.isArray(o.models)
    ? o.models.map((m): CaduceusLocalModel => {
        const x = (m ?? {}) as Record<string, unknown>
        return {
          card: String(x.card ?? ''),
          default: Boolean(x.default),
          id: String(x.id ?? ''),
          maxContext: Number(x.max_context ?? 0),
          maxSlots: Number(x.max_slots ?? 1)
        }
      })
    : fallback.models
  let loaded: CaduceusLocalLoaded | null = null
  if (o.loaded && typeof o.loaded === 'object') {
    const l = o.loaded as Record<string, unknown>
    loaded = { model: String(l.model ?? ''), profile: String(l.profile ?? ''), slots: Number(l.slots ?? 0) }
  }
  return {
    defaultWorker: String(o.default_worker ?? fallback.defaultWorker),
    enabled: typeof o.enabled === 'boolean' ? o.enabled : fallback.enabled,
    loaded,
    models
  }
}

export function setCaduceusState(next: Partial<CaduceusSummary> | Record<string, unknown>): void {
  const cur = $caduceus.get()
  const raw = next as Record<string, unknown>
  $caduceus.set({
    budget: 'budget' in raw ? ((raw.budget as number | null) ?? null) : cur.budget,
    effort: typeof raw.effort === 'string' ? raw.effort : cur.effort,
    enabled: typeof raw.enabled === 'boolean' ? raw.enabled : cur.enabled,
    local: 'local' in raw ? normalizeLocal(raw.local, cur.local) : cur.local,
    orchestrator: raw.orchestrator ? normalizeTier(raw.orchestrator) : cur.orchestrator,
    routerEnabled: typeof raw.router_enabled === 'boolean' ? raw.router_enabled : cur.routerEnabled,
    split: typeof raw.split === 'boolean' ? raw.split : cur.split,
    worker: raw.worker ? normalizeTier(raw.worker) : cur.worker
  })
}

async function call(method: string, params: Record<string, unknown>): Promise<void> {
  const gw = $gateway.get()
  if (!gw) {
    return
  }
  try {
    const res = await gw.request<Record<string, unknown>>(method, params)
    if (res && typeof res === 'object') {
      setCaduceusState(res)
    }
  } catch {
    // Non-fatal — the statusbar reflects last-known state.
  }
}

export async function toggleCaduceus(sessionId: null | string): Promise<void> {
  await call('caduceus.set', { enabled: !$caduceus.get().enabled, session_id: sessionId ?? '' })
}

export async function setCaduceusEnabled(sessionId: null | string, enabled: boolean): Promise<void> {
  await call('caduceus.set', { enabled, session_id: sessionId ?? '' })
}

/** Toggle the Auto Router (per-task worker model selection). */
export async function setCaduceusRouter(sessionId: null | string, enabled: boolean): Promise<void> {
  await call('caduceus.set', { router: enabled, session_id: sessionId ?? '' })
}

/** Toggle Local mode (run workflow workers on local GPU models). */
export async function setCaduceusLocal(sessionId: null | string, enabled: boolean): Promise<void> {
  await call('caduceus.set', { local: enabled, session_id: sessionId ?? '' })
}

export async function refreshCaduceus(sessionId: null | string): Promise<void> {
  const gw = $gateway.get()
  if (!gw) {
    return
  }
  try {
    const res = await gw.request<CaduceusSummary>('caduceus.status', { session_id: sessionId ?? '' })
    setCaduceusState(res)
  } catch {
    // ignore
  }
}

export const openTheater = () => $caduceusTheaterOpen.set(true)
export const closeTheater = () => $caduceusTheaterOpen.set(false)
export const toggleTheater = () => $caduceusTheaterOpen.set(!$caduceusTheaterOpen.get())
