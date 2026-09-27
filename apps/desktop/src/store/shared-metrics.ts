import type { SharedMetricsConsentResult } from '@hermes/shared'

/** Public page describing exactly what shared metrics contain and how consent windows work. */
export const SHARED_METRICS_DOCS_URL = 'https://hermes-agent.nousresearch.com/docs/developer-guide/relay-shared-metrics'

export type SharedMetricsConsent = SharedMetricsConsentResult

/** The first-run answers, as the two config opt-ins `hermes setup` writes. */
export type SharedMetricsChoice = 'local' | 'off' | 'share'

export const SHARED_METRICS_CHOICES: Record<SharedMetricsChoice, { enabled: boolean; send: boolean }> = {
  share: { enabled: true, send: true },
  local: { enabled: true, send: false },
  off: { enabled: false, send: false }
}

export type SharedMetricsRequester = <T = unknown>(method: string, params?: Record<string, unknown>) => Promise<T>

function isConsent(value: unknown): value is SharedMetricsConsent {
  return typeof value === 'object' && value !== null && typeof (value as SharedMetricsConsent).decided === 'boolean'
}

/**
 * The profile's opt-ins, straight from its config.yaml (the backend is the
 * only authority; nothing is latched in the renderer, so an answer given in
 * `hermes setup` and one given here are the same answer). `null` when the
 * backend could not say — an older backend without the method, or a flap —
 * which every caller treats as "don't ask".
 */
export async function readSharedMetricsConsent(request: SharedMetricsRequester): Promise<SharedMetricsConsent | null> {
  try {
    const consent = await request<SharedMetricsConsent>('shared_metrics.status')

    return isConsent(consent) ? consent : null
  } catch {
    return null
  }
}

/** Write both opt-ins; the backend forces `send` off without `enabled` and reconciles consent. */
export async function saveSharedMetricsConsent(
  request: SharedMetricsRequester,
  flags: { enabled: boolean; send: boolean },
  { firstRun = false }: { firstRun?: boolean } = {}
): Promise<SharedMetricsConsent> {
  const consent = await request<SharedMetricsConsent>('shared_metrics.set', { ...flags, first_run: firstRun })

  if (!isConsent(consent)) {
    throw new Error('Unexpected shared_metrics.set response')
  }

  return consent
}
