import { useI18n } from '@/i18n'
import { getManagedRolloutMessages } from '@/i18n/managed-rollouts'

import type { RolloutDraft } from './rollout-config'

export type WavePlanner = (draft: RolloutDraft) => string[][]

export function canonicalWaves(draft: RolloutDraft): string[][] {
  const canary = draft.canaryInstallId ? [draft.canaryInstallId] : []

  return [canary, draft.selectedInstallIds.filter(id => id !== draft.canaryInstallId)]
}

export function WavePreview({ draft, planner }: { draft: RolloutDraft; planner: WavePlanner }) {
  const { t } = useI18n()
  const messages = getManagedRolloutMessages(t)

  return <section aria-label={messages.sections.wavePreview} className="grid gap-2"><p className="text-sm">{messages.sections.wavePreview}</p>{planner(draft).map((wave, index) => <div className="text-xs" key={index}>{messages.labels.wave(index + 1, wave.join(', '))}</div>)}</section>
}
