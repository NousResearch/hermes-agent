import { useState } from 'react'

import { Button } from '@/components/ui/button'

import type { WavePlanner } from './wave-preview'
import type { RolloutDraft } from './rollout-config'
import { WavePreview } from './wave-preview'

export function PreflightReview({ draft, compatible, reviewedToken, planner, onStart }: { draft: RolloutDraft; compatible: boolean; reviewedToken: string | null; planner?: WavePlanner; onStart: (draft: RolloutDraft, reviewedToken: string) => void }) {
  const [started, setStarted] = useState(false)
  const [confirmed, setConfirmed] = useState(false)
  const start = () => { if (!compatible || !confirmed || started || !reviewedToken) return; setStarted(true); onStart(draft, reviewedToken) }
  return <section aria-label="Managed rollout preflight review" className="grid gap-3">{planner ? <WavePreview draft={draft} planner={planner} /> : <p className="text-xs text-amber-600">Wave projection is unavailable; preflight cannot advance.</p>}<p className="text-xs text-(--ui-text-tertiary)">The submitted draft is exactly the previewed canonical wave plan. Renewed or changed rows require reconfirmation.</p>{!compatible ? <p className="text-xs text-amber-600">Current capability is incompatible with this plan; Start is unavailable.</p> : null}<label className="flex items-center gap-2 text-sm"><input checked={confirmed} onChange={event => setConfirmed(event.target.checked)} type="checkbox" />I confirm this preflight review.</label><Button className="motion-reduce:transition-none" disabled={!planner || !compatible || !confirmed || started || !reviewedToken} onClick={start} type="button">{started ? 'Starting…' : 'Start rollout'}</Button></section>
}
