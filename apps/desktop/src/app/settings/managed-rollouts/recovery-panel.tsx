import { useState } from 'react'

import { Button } from '@/components/ui/button'

export interface RecoveryTarget { installId: string; phase: string; unknown: boolean; fenced: boolean; reason: string | null }

export function RecoveryPanel({ target, onRecheck, onRecover, onRetry, onExclude, onStop }: { target: RecoveryTarget; onRecheck: () => void; onRecover: () => void; onRetry: () => void; onExclude: (reason: string) => void; onStop: () => void }) {
  const [reason, setReason] = useState('')

  return <section aria-label="Managed rollout recovery" className="grid min-w-0 gap-2">
    <p className="text-sm">{target.installId} · {target.phase}</p>
    {target.unknown || target.fenced ? <p className="text-xs text-amber-600">Unknown outcome or recovery fence retained: {target.reason ?? 'manual action required'}.</p> : null}
    <p className="text-xs text-(--ui-text-tertiary)">Recheck is read-only. Recover requires correlated clearance. Retry starts a new review; exclusion leaves the original attempt and fence visible.</p>
    <label className="grid gap-1 text-xs">Exclusion reason<input maxLength={512} onChange={event => setReason(event.target.value)} type="text" value={reason} /></label>
    <div className="flex min-w-0 flex-wrap gap-2">
      <Button className="motion-reduce:transition-none" onClick={onRecheck} type="button" variant="outline">Recheck</Button>
      <Button className="motion-reduce:transition-none" disabled={!target.fenced} onClick={onRecover} type="button">Recover</Button>
      <Button className="motion-reduce:transition-none" onClick={onRetry} type="button" variant="secondary">Retry with new review</Button>
      <Button className="motion-reduce:transition-none" disabled={!reason.trim()} onClick={() => onExclude(reason.trim())} type="button" variant="secondary">Exclude</Button>
      <Button className="motion-reduce:transition-none" onClick={onStop} type="button" variant="destructive">Stop</Button>
    </div>
  </section>
}
