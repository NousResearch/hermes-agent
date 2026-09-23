import { useI18n } from '@/i18n'
import { getManagedRolloutMessages } from '@/i18n/managed-rollouts'
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
    aliases: row.aliasConnectionIds,
    machineId: row.source.verifiedHostKeyFingerprint,
    headSha: row.headSha,
    eligibility: 'unknown',
    observedAt: null,
    sharedMachine: (machineCounts.get(row.source.verifiedHostKeyFingerprint) ?? 0) > 1
  }))
}

export function FleetOverview({ targets, selected, onToggle, emptyMessage }: { targets: readonly ManagedRolloutTarget[]; selected: ReadonlySet<string>; onToggle: (identity: string) => void; emptyMessage?: string }) {
  const { t } = useI18n()
  const messages = getManagedRolloutMessages(t)

  return <section aria-label={messages.sections.fleet} className="grid max-h-80 min-w-0 gap-1 overflow-y-auto overscroll-contain" tabIndex={0}>{targets.length ? targets.map(target => <TargetRow key={targetIdentity(target)} onToggle={onToggle} selected={selected.has(targetIdentity(target))} target={target} />) : <p className="text-xs text-(--ui-text-tertiary)">{emptyMessage ?? messages.descriptions.noObservedInstallations}</p>}</section>
}
