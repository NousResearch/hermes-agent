import { useState } from 'react'

import { Button } from '@/components/ui/button'
import type { PlanChange, RolloutPlan } from '@/lib/managed-rollout-contract'

import type { RolloutDraft } from './rollout-config'
import type { WavePlanner } from './wave-preview'
import { WavePreview } from './wave-preview'

export function PreflightReview({ draft, compatible, reviewedToken, planner, plan, changes = [], blockers = [], expiresAt = null, pending = false, onStart, onRenew }: { draft: RolloutDraft; compatible: boolean; reviewedToken: string | null; planner?: WavePlanner; plan?: RolloutPlan; changes?: PlanChange[]; blockers?: string[]; expiresAt?: number | null; pending?: boolean; onStart: (draft: RolloutDraft, reviewedToken: string) => Promise<void> | void; onRenew?: () => void }) {
  const [started, setStarted] = useState(false)
  const [confirmed, setConfirmed] = useState(false)

  const start = async () => {
    if (!compatible || !confirmed || started || pending || !reviewedToken || (expiresAt !== null && Date.now() >= expiresAt)) {return}
    setStarted(true)

    try { await onStart(draft, reviewedToken) } catch { setStarted(false) }
  }

  const expired = expiresAt !== null && Date.now() >= expiresAt

  return <section aria-label="Managed rollout preflight review" className="grid gap-3">
    {plan ? <p className="text-sm">Pinned target: {plan.target.repositoryId} · {plan.target.branch} · {plan.target.sha}</p> : null}
    {planner ? <WavePreview draft={draft} planner={planner} /> : <p className="text-xs text-amber-600">Wave projection is unavailable; preflight cannot advance.</p>}
    <p className="text-xs text-(--ui-text-tertiary)">This is Main's canonical plan. Changed rows require a renewed review and confirmation.</p>
    {changes.map((change, index) => <p className="text-xs text-amber-600" key={`${change.installId}:${change.field}:${index}`}>Changed {change.installId}: {change.field} ({change.before ?? 'none'} → {change.after ?? 'none'}).</p>)}
    {blockers.map(blocker => <p className="text-xs text-amber-600" key={blocker}>Blocked: {blocker}</p>)}
    {!compatible || expired ? <p className="text-xs text-amber-600">Current review cannot authorize Start. Requalify the plan.</p> : null}
    {onRenew ? <Button disabled={pending} onClick={onRenew} type="button" variant="outline">Renew review</Button> : null}
    <label className="flex items-center gap-2 text-sm"><input checked={confirmed} onChange={event => setConfirmed(event.target.checked)} type="checkbox" />I confirm this exact preflight review.</label>
    <Button className="motion-reduce:transition-none" disabled={!planner || !compatible || expired || !confirmed || started || pending || !reviewedToken || changes.length > 0 || blockers.length > 0} onClick={() => { void start() }} type="button">{started || pending ? 'Starting…' : 'Start rollout'}</Button>
  </section>
}
