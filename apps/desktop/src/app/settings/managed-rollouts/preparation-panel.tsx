import { useEffect, useMemo, useState } from 'react'
import { Button } from '@/components/ui/button'
import { FleetOverview } from './fleet-overview'
import { targetIdentity, type ManagedRolloutTarget } from './target-row'

export function PreparationPanel({ targets, reviewGeneration, onPrepare }: { targets: readonly ManagedRolloutTarget[]; reviewGeneration: string; onPrepare: (targets: ManagedRolloutTarget[]) => void }) {
  const [selected, setSelected] = useState<ReadonlySet<string>>(new Set())
  const [confirmed, setConfirmed] = useState(false)
  const [pending, setPending] = useState(false)
  useEffect(() => { setConfirmed(false); setPending(false); setSelected(new Set()) }, [reviewGeneration])
  const chosen = useMemo(() => targets.filter(target => target.supported && selected.has(targetIdentity(target))), [selected, targets])
  const toggle = (identity: string) => { setConfirmed(false); setSelected(current => { const next = new Set(current); if (next.has(identity)) next.delete(identity); else next.add(identity); return next }) }
  const confirmPreparation = () => { if (!confirmed || chosen.length === 0 || pending) return; setPending(true); onPrepare(chosen); setConfirmed(false) }
  return <section className="grid gap-3" aria-label="Managed rollout preparation">
    <FleetOverview key={reviewGeneration} onToggle={toggle} selected={selected} targets={targets} />
    <p className="text-xs text-(--ui-text-tertiary)">Preparation invalidates prior review and requires requalification after target, alias, source, or scope changes.</p>
    <label className="flex items-center gap-2 text-sm"><input checked={confirmed} onChange={event => setConfirmed(event.target.checked)} type="checkbox" />I confirm these targets are the intended separate preparation set.</label>
    <Button disabled={!confirmed || chosen.length === 0 || pending} onClick={confirmPreparation}>{pending ? 'Preparing…' : 'Prepare selected targets'}</Button>
  </section>
}
