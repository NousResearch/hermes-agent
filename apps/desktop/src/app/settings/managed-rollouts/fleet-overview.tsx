import type { ManagedRolloutInventory } from '@/store/managed-rollouts'

import { type ManagedRolloutTarget, targetIdentity, TargetRow } from './target-row'

export function targetsFromInventory(inventory: ManagedRolloutInventory): ManagedRolloutTarget[] {
  const machineCounts = new Map<string, number>()

  for (const row of inventory.observations) {
    const machine = row.source.verifiedHostKeyFingerprint
    machineCounts.set(machine, (machineCounts.get(machine) ?? 0) + 1)
  }

  return inventory.observations.map(row => ({
    installId: row.installId,
    connectionId: row.connectionId,
    label: row.connectionId,
    alias: row.aliasConnectionIds.length ? row.aliasConnectionIds.join(', ') : null,
    machineId: row.source.verifiedHostKeyFingerprint,
    supported: true,
    sharedMachine: (machineCounts.get(row.source.verifiedHostKeyFingerprint) ?? 0) > 1
  }))
}

export function FleetOverview({ targets, selected, onToggle }: { targets: readonly ManagedRolloutTarget[]; selected: ReadonlySet<string>; onToggle: (identity: string) => void }) {
  return <section aria-label="Managed rollout fleet" className="grid min-w-0 gap-1">{targets.length ? targets.map(target => <TargetRow key={targetIdentity(target)} onToggle={onToggle} selected={selected.has(targetIdentity(target))} target={target} />) : <p className="text-xs text-(--ui-text-tertiary)">No managed SSH installations were observed.</p>}</section>
}
