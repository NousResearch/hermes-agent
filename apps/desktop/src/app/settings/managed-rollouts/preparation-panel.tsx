import { useEffect, useMemo, useState } from 'react'

import { Button } from '@/components/ui/button'

import { FleetOverview } from './fleet-overview'
import { type ManagedRolloutTarget, targetIdentity } from './target-row'

export function PreparationPanel({ targets, reviewGeneration, disabled = false, reviewDisabled = false, onPrepare, onReview }: { targets: readonly ManagedRolloutTarget[]; reviewGeneration: string; disabled?: boolean; reviewDisabled?: boolean; onPrepare: (targets: ManagedRolloutTarget[]) => Promise<void> | void; onReview?: (targets: ManagedRolloutTarget[]) => void }) {
  const [selected, setSelected] = useState<ReadonlySet<string>>(new Set())
  const [confirmed, setConfirmed] = useState(false)
  const [pending, setPending] = useState(false)
  useEffect(() => { setConfirmed(false); setPending(false); setSelected(new Set()) }, [reviewGeneration])
  const chosen = useMemo(() => targets.filter(target => target.supported && selected.has(targetIdentity(target))), [selected, targets])

  const toggle = (identity: string) => { setConfirmed(false); setSelected(current => { const next = new Set(current);

 if (next.has(identity)) {next.delete(identity);} else {next.add(identity);}

 return next }) }

  const confirmPreparation = async () => {
    if (!confirmed || chosen.length === 0 || pending || disabled) {return}
    setPending(true)
    setConfirmed(false)

    try { await onPrepare(chosen) } finally { setPending(false) }
  }

  return <section aria-label="Managed rollout preparation" className="grid gap-3">
    <FleetOverview key={reviewGeneration} onToggle={toggle} selected={selected} targets={targets} />
    <p className="text-xs text-(--ui-text-tertiary)">Preparation runs the existing individual updater at the configured branch tip. It is unpinned, separate from the rollout, and invalidates every earlier review. Reinspect the inventory afterward.</p>
    {onReview ? <Button className="motion-reduce:transition-none" disabled={disabled || reviewDisabled || pending || chosen.length === 0} onClick={() => onReview(chosen)} type="button" variant="outline">Review selected target</Button> : null}
    <label className="flex items-center gap-2 text-sm"><input checked={confirmed} onChange={event => setConfirmed(event.target.checked)} type="checkbox" />I confirm these targets are the intended separate preparation set.</label>
    <Button className="motion-reduce:transition-none" disabled={disabled || !confirmed || chosen.length === 0 || pending} onClick={() => { void confirmPreparation() }} type="button">{pending ? 'Preparing…' : 'Prepare selected targets'}</Button>
  </section>
}
