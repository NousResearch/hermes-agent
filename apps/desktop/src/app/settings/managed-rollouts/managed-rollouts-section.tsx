import { useStore } from '@nanostores/react'
import { useEffect } from 'react'

import { useI18n } from '@/i18n'
import { getManagedRolloutMessages } from '@/i18n/managed-rollouts'
import {
  $managedRollouts,
  type ManagedRolloutsState,
  startManagedRolloutPolling
} from '@/store/managed-rollouts'

import { RolloutHistory, type RolloutHistoryEntry } from './rollout-history'

function bridgeAvailable(): boolean {
  return (
    typeof window !== 'undefined' &&
    Boolean(window.hermesDesktop?.connections?.managedRollouts)
  )
}

function snapshotLabel(
  state: ManagedRolloutsState,
  noActive: string,
  activePhase: (phase: string) => string
): string {
  if (!state.snapshot) {return noActive}

  return activePhase(state.snapshot.phase)
}

export function ManagedRolloutsSection({
  history = [],
  onSelect = () => undefined
}: {
  history?: readonly RolloutHistoryEntry[]
  onSelect?: (id: string) => void
}) {
  const { t, locale } = useI18n()
  const messages = getManagedRolloutMessages(t, locale)
  const state = useStore($managedRollouts)
  const available = bridgeAvailable()

  useEffect(() => {
    if (!available) {return}

    return startManagedRolloutPolling()
  }, [available])

  if (!available) {return null}

  const unsupported = state.status === 'unsupported'

  return (
    <section aria-label={messages.title} className="grid min-w-0 gap-3">
      <div>
        <h2 className="text-sm font-medium">{messages.title}</h2>
        <p aria-live="polite" className="text-xs text-(--ui-text-tertiary)" role="status">
          {unsupported
            ? messages.warnings.unavailable
            : snapshotLabel(state, messages.noActive, messages.status.activePhase)}
        </p>
        {unsupported && state.error ? (
          <p aria-label={state.error} className="text-xs text-amber-600" role="alert">
            {state.error}
          </p>
        ) : null}
      </div>
      <RolloutHistory entries={history} onSelect={onSelect} />
    </section>
  )
}
