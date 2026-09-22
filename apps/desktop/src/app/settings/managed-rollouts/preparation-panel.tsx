import { useMemo, useState } from 'react'
import { Button } from '@/components/ui/button'
import { FleetOverview } from './fleet-overview'
import { targetIdentity, type ManagedRolloutTarget } from './target-row'

export function PreparationPanel({ targets, onPrepare }: { targets: readonly ManagedRolloutTarget[]; onPrepare: (targets: ManagedRolloutTarget[]) => void }) {
  const [selected, setSelected] = useState<ReadonlySet<string>>(new Set())
  const [confirmed, setConfirmed] = useState(false)
  const chosen = useMemo(() => targets.filter(target => selected.has(targetIdentity(target))), [selected, targets])
  const toggle = (identity: string) => { setConfirmed(false); setSelected(current => { const next = new Set(current); if (next.has(identity)) next.delete(identity); else next.add(identity); return next }) }
  const confirmPreparation = () => { if (!confirmed || chosen.length === 0) return; onPrepare(chosen); setConfirmed(false) }
  return <section className="grid gap-3" aria-label="Managed rollout preparation">
    <FleetOverview onToggle={toggle} selected={selected} targets={targets} />
    <p className="text-xs text-(--ui-text-tertiary)">Preparation invalidates prior review and requires requalification after target, alias, source, or scope changes.</p>
    <label className="flex items-center gap-2 text-sm"><input checked={confirmed} onChange={event => setConfirmed(event.target.checked)} type="checkbox" />I confirm these targets are the intended separate preparation set.</label>
    <Button disabled={!confirmed || chosen.length === 0} onClick={confirmPreparation}>Prepare selected targets</Button>
  </section>
}
