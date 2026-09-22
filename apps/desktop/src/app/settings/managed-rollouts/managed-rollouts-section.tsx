import { useStore } from '@nanostores/react'
import { $managedRollouts } from '@/store/managed-rollouts'
import { RolloutHistory, type RolloutHistoryEntry } from './rollout-history'

function bridgeAvailable(): boolean {
  return typeof window !== 'undefined' && Boolean((window as Window & { hermesDesktop?: { managedRollouts?: unknown } }).hermesDesktop?.managedRollouts)
}

export function ManagedRolloutsSection({ history, onSelect }: { history: readonly RolloutHistoryEntry[]; onSelect: (id: string) => void }) {
  const state = useStore($managedRollouts)
  if (!bridgeAvailable()) return null
  return <section aria-label="Managed rollouts" className="grid gap-3"><div><h2 className="text-sm font-medium">Managed rollouts</h2><p className="text-xs text-(--ui-text-tertiary)">{state.snapshot ? `Active phase: ${state.snapshot.phase}` : 'No active rollout snapshot.'}</p></div><RolloutHistory entries={history} onSelect={onSelect} /></section>
}
