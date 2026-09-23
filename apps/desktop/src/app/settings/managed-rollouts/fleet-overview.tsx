import { type ManagedRolloutTarget, targetIdentity, TargetRow } from './target-row'

export function FleetOverview({ targets, selected, onToggle }: { targets: readonly ManagedRolloutTarget[]; selected: ReadonlySet<string>; onToggle: (identity: string) => void }) {
  return <section aria-label="Managed rollout fleet" className="grid min-w-0 gap-1">{targets.map(target => <TargetRow key={targetIdentity(target)} onToggle={onToggle} selected={selected.has(targetIdentity(target))} target={target} />)}</section>
}
