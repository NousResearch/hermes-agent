import { useI18n } from '@/i18n'
import { getManagedRolloutMessages } from '@/i18n/managed-rollouts'
import type { RolloutCapabilities, TargetAttempt } from '@/lib/managed-rollout-contract'
import { estimateCapabilityBoundOperationalMs } from '@/lib/managed-rollout-waves'

import type { RolloutDraft } from './rollout-config'

export type WavePlanner = (draft: RolloutDraft) => string[][]

export function canonicalWaves(draft: RolloutDraft): string[][] {
  const canary = draft.canaryInstallId ? [draft.canaryInstallId] : []

  return [canary, draft.selectedInstallIds.filter(id => id !== draft.canaryInstallId)]
}

/**
 * Observed machine durations, keyed by installation, from settled receipts that
 * recorded both ends of the work. An installation with no measured window stays
 * absent rather than being credited with a guessed duration.
 */
export function observedDurations(attempts: readonly TargetAttempt[]): Record<string, number> {
  const durations: Record<string, number> = {}

  for (const attempt of attempts) {
    const started = attempt.receipt?.startedAt
    const finished = attempt.receipt?.finishedAt

    if (!started || !finished) {continue}

    const start = Date.parse(started)
    const end = Date.parse(finished)

    if (!Number.isFinite(start) || !Number.isFinite(end) || end < start) {continue}

    durations[attempt.identity.installId] = end - start
  }

  return durations
}

/** Machine-time label for an estimate. Approval waiting is never included. */
export function formatOperationalDuration(ms: number): string {
  const totalSeconds = Math.max(0, Math.round(ms / 1000))
  const hours = Math.floor(totalSeconds / 3600)
  const minutes = Math.floor((totalSeconds % 3600) / 60)
  const seconds = totalSeconds % 60

  if (hours > 0) {return `${hours}h ${String(minutes).padStart(2, '0')}m`}

  if (minutes > 0) {return `${minutes}m ${String(seconds).padStart(2, '0')}s`}

  return `${seconds}s`
}

/**
 * The estimate is always derived from the exact waves the preview renders, so
 * a submitted draft can never be presented with a different partitioning's
 * number. Unknown history or an unadvertised capability returns null.
 */
export function previewEstimate(
  waves: readonly (readonly string[])[],
  durations: Record<string, number>,
  concurrency: number,
  capabilities: RolloutCapabilities
): number | null {
  return estimateCapabilityBoundOperationalMs(waves, durations, concurrency, capabilities)
}

export function WavePreview({ draft, planner, durations, capabilities }: {
  draft: RolloutDraft
  planner: WavePlanner
  durations?: Record<string, number> | null
  capabilities?: RolloutCapabilities | null
}) {
  const { t } = useI18n()
  const messages = getManagedRolloutMessages(t)
  const waves = planner(draft)
  const estimate = durations && capabilities ? previewEstimate(waves, durations, draft.concurrency, capabilities) : null

  return <section aria-label={messages.sections.wavePreview} className="grid gap-2">
    <p className="text-sm">{messages.sections.wavePreview}</p>
    {waves.map((wave, index) => <div className="text-xs" key={index}>{messages.labels.wave(index + 1, wave.join(', '))}</div>)}
    <p className="text-xs">{estimate === null ? messages.warnings.estimateUnavailable : messages.labels.operationalEstimate(formatOperationalDuration(estimate))}</p>
    {estimate === null ? null : <p className="text-xs text-(--ui-text-tertiary)">{messages.descriptions.historicalEstimate}</p>}
  </section>
}
