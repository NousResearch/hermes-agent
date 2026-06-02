import { atom } from 'nanostores'

import { $gateway } from './gateway'

export interface CaduceusTier {
  provider: string
  model: string
}

export interface CaduceusSummary {
  enabled: boolean
  orchestrator: CaduceusTier
  worker: CaduceusTier
  budget: number | null
  effort?: string
  split?: boolean
}

const EMPTY_TIER: CaduceusTier = { model: '', provider: '' }

const DEFAULT_STATE: CaduceusSummary = {
  budget: null,
  effort: 'high',
  enabled: false,
  orchestrator: { ...EMPTY_TIER },
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

export function setCaduceusState(next: Partial<CaduceusSummary> | Record<string, unknown>): void {
  const cur = $caduceus.get()
  const raw = next as Record<string, unknown>
  $caduceus.set({
    budget: 'budget' in raw ? ((raw.budget as number | null) ?? null) : cur.budget,
    effort: typeof raw.effort === 'string' ? raw.effort : cur.effort,
    enabled: typeof raw.enabled === 'boolean' ? raw.enabled : cur.enabled,
    orchestrator: raw.orchestrator ? normalizeTier(raw.orchestrator) : cur.orchestrator,
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
