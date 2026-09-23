
import { Button } from '@/components/ui/button'
import { useI18n } from '@/i18n'
import { getManagedRolloutMessages } from '@/i18n/managed-rollouts'

export interface RolloutDraft {
  mode: 'manual' | 'auto-if-healthy'
  concurrency: number
  canaryInstallId: string | null
  selectedInstallIds: string[]
}

export function RolloutConfig({ draft, onChange, onContinue, pending = false }: { draft: RolloutDraft; onChange: (draft: RolloutDraft) => void; onContinue: (draft: RolloutDraft) => void; pending?: boolean }) {
  const { t } = useI18n()
  const messages = getManagedRolloutMessages(t)

  return <section aria-label={messages.sections.configuration} className="grid gap-3">
    <label className="grid gap-1 text-sm">{messages.labels.progressionMode}<select onChange={event => onChange({ ...draft, mode: event.target.value as RolloutDraft['mode'] })} value={draft.mode}><option value="manual">{messages.policy.manual}</option><option value="auto-if-healthy">{messages.policy.automatic}</option></select></label>
    <label className="grid gap-1 text-sm">{messages.labels.concurrency}<input min={1} readOnly type="number" value={draft.concurrency} /></label>
    <p className="text-xs text-(--ui-text-tertiary)">{messages.descriptions.configuration}</p>
    <Button className="motion-reduce:transition-none" disabled={pending || draft.selectedInstallIds.length === 0 || !draft.canaryInstallId} onClick={() => onContinue(draft)} type="button">{pending ? messages.actions.checking : messages.actions.continueToPreflight}</Button>
  </section>
}
