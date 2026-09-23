
import { Button } from '@/components/ui/button'

export interface RolloutDraft {
  mode: 'manual' | 'auto-if-healthy'
  concurrency: number
  canaryInstallId: string | null
  selectedInstallIds: string[]
}

export function RolloutConfig({ draft, onChange, onContinue, pending = false }: { draft: RolloutDraft; onChange: (draft: RolloutDraft) => void; onContinue: (draft: RolloutDraft) => void; pending?: boolean }) {
  return <section aria-label="Managed rollout configuration" className="grid gap-3">
    <label className="grid gap-1 text-sm">Progression mode<select onChange={event => onChange({ ...draft, mode: event.target.value as RolloutDraft['mode'] })} value={draft.mode}><option value="manual">Manual approval</option><option value="auto-if-healthy">Automatic when healthy</option></select></label>
    <label className="grid gap-1 text-sm">Concurrency<input min={1} readOnly type="number" value={draft.concurrency} /></label>
    <p className="text-xs text-(--ui-text-tertiary)">Updates remain serial. The first selected installation is the canary; Main validates the complete submitted wave plan before issuing a review token.</p>
    <Button className="motion-reduce:transition-none" disabled={pending || draft.selectedInstallIds.length === 0 || !draft.canaryInstallId} onClick={() => onContinue(draft)} type="button">{pending ? 'Checking…' : 'Continue to preflight'}</Button>
  </section>
}
