import { useState } from 'react'

import { Button } from '@/components/ui/button'

import { canonicalWaves } from './wave-preview'
import type { RolloutDraft } from './rollout-config'
import { WavePreview } from './wave-preview'

export function PreflightReview({ draft, compatible, onStart }: { draft: RolloutDraft; compatible: boolean; onStart: () => void }) {
  const [started, setStarted] = useState(false)
  const [confirmed, setConfirmed] = useState(false)
  const start = () => { if (!compatible || !confirmed || started) return; setStarted(true); onStart() }
  return <section aria-label="Managed rollout preflight review" className="grid gap-3"><WavePreview draft={draft} /><p className="text-xs text-(--ui-text-tertiary)">The submitted draft is exactly the previewed canonical wave plan. Renewed or changed rows require reconfirmation.</p>{!compatible ? <p className="text-xs text-amber-600">Current capability is incompatible with this plan; Start is unavailable.</p> : null}<label className="flex items-center gap-2 text-sm"><input checked={confirmed} onChange={event => setConfirmed(event.target.checked)} type="checkbox" />I confirm this preflight review.</label><Button disabled={!compatible || !confirmed || started} onClick={start}>{started ? 'Starting…' : 'Start rollout'}</Button>{canonicalWaves(draft).length === 0 ? null : null}</section>
}
