import { Button } from '@/components/ui/button'
import { useI18n } from '@/i18n'
import { getManagedRolloutMessages } from '@/i18n/managed-rollouts'

export interface ManagedRolloutTarget {
  installId: string
  connectionId?: string
  label: string
  aliases?: readonly string[]
  machineId: string
  headSha: string
  eligibility: 'unknown'
  observedAt: string | null
  sharedMachine: boolean
}

export function targetIdentity(target: ManagedRolloutTarget): string {
  return JSON.stringify([target.machineId, target.installId])
}

export function TargetRow({ target, selected, onToggle }: { target: ManagedRolloutTarget; selected: boolean; onToggle: (identity: string) => void }) {
  const identity = targetIdentity(target)
  const { t } = useI18n()
  const messages = getManagedRolloutMessages(t)

  return <div className="flex min-w-0 items-start justify-between gap-3 border-b border-(--ui-stroke-secondary) py-2">
    <div className="min-w-0 flex-1">
      <p className="break-all text-sm font-medium">{target.label}</p>
      <p className="break-all text-xs text-(--ui-text-tertiary)">{target.machineId} · {target.installId}</p>
      {target.aliases?.length ? <details className="text-xs">
        <summary className="w-fit cursor-pointer text-(--ui-text-secondary)">{messages.labels.aliases(target.aliases.length)}</summary>
        <ul className="ms-4 list-disc break-all text-(--ui-text-secondary)">{target.aliases.map(alias => <li key={alias}>{alias}</li>)}</ul>
      </details> : null}
      <p className="break-all text-xs">{messages.labels.observedHead}: <span>{target.headSha ?? messages.labels.unknownFact}</span></p>
      <p className="text-xs">{messages.labels.eligibility}: {messages.labels.unknownFact}</p>
      <p className="text-xs">{messages.labels.observationTime}: {target.observedAt ?? messages.labels.unknownFact}</p>
      {target.sharedMachine ? <p className="border-s-2 border-amber-700 ps-2 text-xs text-(--ui-text-primary) dark:border-amber-400">{messages.warnings.sharedMachine}</p> : null}
    </div>
    <Button aria-pressed={selected} className="motion-reduce:transition-none" onClick={() => onToggle(identity)} size="sm" type="button" variant={selected ? 'default' : 'outline'}>{selected ? messages.actions.selected : messages.actions.select}</Button>
  </div>
}
