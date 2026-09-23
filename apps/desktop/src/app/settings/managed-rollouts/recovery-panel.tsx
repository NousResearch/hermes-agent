import { useState } from 'react'

import { Button } from '@/components/ui/button'
import { useI18n } from '@/i18n'
import { getManagedRolloutMessages } from '@/i18n/managed-rollouts'

export interface RecoveryTarget { installId: string; phase: string; unknown: boolean; fenced: boolean; reason: string | null }

export function RecoveryPanel({ target, onRecheck, onRecover, onRetry, onExclude, onStop }: { target: RecoveryTarget; onRecheck: () => void; onRecover: () => void; onRetry: () => void; onExclude: (reason: string) => void; onStop: () => void }) {
  const { t } = useI18n()
  const messages = getManagedRolloutMessages(t)
  const [reason, setReason] = useState('')

  return <section aria-label={messages.sections.recovery} className="grid min-w-0 gap-2">
    <p className="text-sm">{target.installId} · {target.phase}</p>
    {target.unknown ? <p className="text-xs text-amber-600">{messages.warnings.unknownOutcome}</p> : null}
    {target.fenced ? <p className="text-xs text-amber-600">{messages.warnings.recoveryFence}</p> : null}
    {target.reason ? <p className="text-xs">{messages.labels.reason(target.reason)}</p> : null}
    <p className="text-xs text-(--ui-text-tertiary)">{messages.descriptions.recoveryActions}</p>
    <label className="grid gap-1 text-xs">{messages.labels.exclusionReason}<input maxLength={512} onChange={event => setReason(event.target.value)} type="text" value={reason} /></label>
    <div className="flex min-w-0 flex-wrap gap-2">
      <Button className="motion-reduce:transition-none" onClick={onRecheck} type="button" variant="outline">{messages.actions.recheckOutcome}</Button>
      <Button className="motion-reduce:transition-none" disabled={!target.fenced} onClick={onRecover} type="button">{messages.actions.recoverConnections}</Button>
      <Button className="motion-reduce:transition-none" onClick={onRetry} type="button" variant="secondary">{messages.actions.retry}</Button>
      <Button className="motion-reduce:transition-none" disabled={!reason.trim()} onClick={() => onExclude(reason.trim())} type="button" variant="secondary">{messages.actions.exclude}</Button>
      <Button className="motion-reduce:transition-none" onClick={onStop} type="button" variant="destructive">{messages.actions.stop}</Button>
    </div>
  </section>
}
