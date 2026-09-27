import type { SharedMetricsConsentResult } from '@hermes/shared'
import { atom } from 'nanostores'

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

/**
 * The focused profile's answer, as last read from the backend by the consent
 * host. `null` = not asked yet or the backend could not say (never offer then).
 * A cache of backend truth: the host re-reads it on every profile switch.
 */
export const $sharedMetricsConsent = atom<SharedMetricsConsent | null>(null)

/** The "What is collected" details dialog, opened only by the user from the strip. */
export const $sharedMetricsDetailsOpen = atom(false)

/** The first-run offer is still unanswered for the focused profile. */
export function sharedMetricsOfferPending(consent: SharedMetricsConsent | null): boolean {
  return consent?.decided === false
}

/** Record a first-run answer; the strip and the dialog retire as soon as the backend reports it decided. */
export async function answerSharedMetricsOffer(
  request: SharedMetricsRequester,
  choice: SharedMetricsChoice
): Promise<void> {
  $sharedMetricsConsent.set(await saveSharedMetricsConsent(request, SHARED_METRICS_CHOICES[choice], { firstRun: true }))
  $sharedMetricsDetailsOpen.set(false)
}
