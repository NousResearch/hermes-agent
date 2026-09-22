import { useEffect } from 'react'
import { useStore } from '@nanostores/react'
import { useI18n } from '@/i18n'
import {
  $managedRollouts,
  startManagedRolloutPolling,
  type ManagedRolloutsState
} from '@/store/managed-rollouts'
import { RolloutHistory, type RolloutHistoryEntry } from './rollout-history'

function bridgeAvailable(): boolean {
  return (
    typeof window !== 'undefined' &&
    Boolean(window.hermesDesktop?.connections?.managedRollouts)
  )
}

function snapshotLabel(state: ManagedRolloutsState, noActive: string): string {
  if (!state.snapshot) return noActive
  return `Active phase: ${state.snapshot.phase}`
}

export function ManagedRolloutsSection({
  history = [],
  onSelect = () => undefined
}: {
  history?: readonly RolloutHistoryEntry[]
  onSelect?: (id: string) => void
}) {
  const { t } = useI18n()
  const state = useStore($managedRollouts)
  const available = bridgeAvailable()

  useEffect(() => {
    if (!available) return
    return startManagedRolloutPolling()
  }, [available])

  if (!available) return null

  return (
    <section aria-label={t.settings.managedRollouts.title} className="grid gap-3">
      <div>
        <h2 className="text-sm font-medium">{t.settings.managedRollouts.title}</h2>
        <p className="text-xs text-(--ui-text-tertiary)">
          {state.status === 'unsupported' ? state.error || t.settings.managedRollouts.noActive : snapshotLabel(state, t.settings.managedRollouts.noActive)}
        </p>
      </div>
      <RolloutHistory entries={history} onSelect={onSelect} />
    </section>
  )
}
