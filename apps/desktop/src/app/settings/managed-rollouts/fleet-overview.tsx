import { TargetRow, targetIdentity, type ManagedRolloutTarget } from './target-row'

export function FleetOverview({ targets, selected, onToggle }: { targets: readonly ManagedRolloutTarget[]; selected: ReadonlySet<string>; onToggle: (identity: string) => void }) {
  return <section aria-label="Managed rollout fleet" className="grid gap-1">{targets.map(target => <TargetRow key={targetIdentity(target)} target={target} selected={selected.has(targetIdentity(target))} onToggle={onToggle} />)}</section>
}
