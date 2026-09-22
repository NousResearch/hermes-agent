import { useState } from 'react'

import { Button } from '@/components/ui/button'

export interface RolloutDraft {
  mode: 'manual' | 'auto-if-healthy'
  concurrency: number
  canaryInstallId: string | null
  selectedInstallIds: string[]
}

export function RolloutConfig({ draft, reviewedToken, draftToken, onChange, onContinue }: { draft: RolloutDraft; reviewedToken: string | null; draftToken: string; onChange: (draft: RolloutDraft) => void; onContinue: (reviewedToken: string, draft: RolloutDraft) => void }) {
  const changed = reviewedToken !== draftToken
  return <section className="grid gap-3" aria-label="Managed rollout configuration">
    <label className="grid gap-1 text-sm">Progression mode<select value={draft.mode} onChange={event => onChange({ ...draft, mode: event.target.value as RolloutDraft['mode'] })}><option value="manual">Manual approval</option><option value="auto-if-healthy">Automatic when healthy</option></select></label>
    <label className="grid gap-1 text-sm">Concurrency<input min={1} readOnly type="number" value={draft.concurrency} /></label>
    <p className="text-xs text-(--ui-text-tertiary)">Serial capability is required by the active target contract. The selected canary remains stable until the draft changes.</p>
    {changed ? <p className="text-xs text-amber-600">Configuration changed; renew the review token before continuing.</p> : null}
    <Button className="motion-reduce:transition-none" disabled={changed || !reviewedToken || draft.selectedInstallIds.length === 0 || !draft.canaryInstallId} onClick={() => reviewedToken && onContinue(reviewedToken, draft)} type="button">Continue to preflight</Button>
  </section>
}
