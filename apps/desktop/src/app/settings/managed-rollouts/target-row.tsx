import { Button } from '@/components/ui/button'
import { useI18n } from '@/i18n'
import { getManagedRolloutMessages } from '@/i18n/managed-rollouts'

export interface ManagedRolloutTarget {
  installId: string
  connectionId?: string
  label: string
  alias?: string | null
  machineId: string
  supported: boolean
  sharedMachine: boolean
}

export function targetIdentity(target: ManagedRolloutTarget): string {
  return JSON.stringify([target.machineId, target.installId])
}

export function TargetRow({ target, selected, onToggle }: { target: ManagedRolloutTarget; selected: boolean; onToggle: (identity: string) => void }) {
  const identity = targetIdentity(target)
  const { t } = useI18n()
  const messages = getManagedRolloutMessages(t)

  return <div className="flex min-w-0 items-center justify-between gap-3 border-b border-(--ui-stroke-secondary) py-2">
    <div className="min-w-0"><p className="truncate text-sm">{target.label}</p><p className="truncate text-xs text-(--ui-text-tertiary)">{target.alias ? `${target.alias} · ` : ''}{target.machineId} · {target.installId}</p>{target.sharedMachine ? <p className="text-xs text-amber-600">{messages.warnings.sharedMachine}</p> : null}{!target.supported ? <p className="text-xs text-amber-600">{messages.warnings.unsupportedTarget}</p> : null}</div>
    <Button aria-pressed={selected} className="motion-reduce:transition-none" disabled={!target.supported} onClick={() => onToggle(identity)} size="sm" type="button" variant={selected ? 'default' : 'outline'}>{selected ? messages.actions.selected : messages.actions.select}</Button>
  </div>
}
