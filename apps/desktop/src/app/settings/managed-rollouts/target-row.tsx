import { Button } from '@/components/ui/button'

export interface ManagedRolloutTarget {
  installId: string
  label: string
  alias?: string | null
  machineId: string
  supported: boolean
  sharedMachine: boolean
}

export function targetIdentity(target: ManagedRolloutTarget): string {
  return `${target.machineId}:${target.installId}`
}

export function TargetRow({ target, selected, onToggle }: { target: ManagedRolloutTarget; selected: boolean; onToggle: (identity: string) => void }) {
  const identity = targetIdentity(target)
  return <div className="flex items-center justify-between gap-3 border-b border-(--ui-stroke-secondary) py-2">
    <div className="min-w-0"><p className="truncate text-sm">{target.label}</p><p className="truncate text-xs text-(--ui-text-tertiary)">{target.alias ? `${target.alias} · ` : ''}{target.machineId} · {target.installId}</p>{target.sharedMachine ? <p className="text-xs text-amber-600">Shared machine; review ownership before preparation.</p> : null}{!target.supported ? <p className="text-xs text-amber-600">Unsupported target.</p> : null}</div>
    <Button aria-pressed={selected} disabled={!target.supported} onClick={() => onToggle(identity)} size="sm" variant={selected ? 'default' : 'outline'}>{selected ? 'Selected' : 'Select'}</Button>
  </div>
}
