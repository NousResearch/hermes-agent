import { useState } from 'react'

import { Button } from '@/components/ui/button'
import { Checkbox } from '@/components/ui/checkbox'
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from '@/components/ui/dialog'
import { useI18n } from '@/i18n'
import type { CatalogReview } from '@/types/skill-catalog'

import type { useProviderAction } from './provider-action'
import { reviewConsent } from './provider-consent'

interface ReviewPlanProps {
  review: CatalogReview
  busy: boolean
  acknowledged: string[]
  onAcknowledge: (ids: string[]) => void
  onConfirm: () => void
}

function ReviewFindings({ review, busy, acknowledged, onAcknowledge }: ReviewPlanProps) {
  const { t } = useI18n()

  return <>
    <p>{review.scan.summary}</p>
    <ul className="space-y-3">{review.scan.findings.map(finding => <li key={finding.id}>
      <p>{finding.severity} · {finding.category} · {finding.file}{finding.line ? `:${finding.line}` : ''}</p>
      {finding.acknowledgement_required ? <label className="flex items-start gap-2">
        <Checkbox checked={acknowledged.includes(finding.id)} disabled={busy} onCheckedChange={checked => onAcknowledge(checked ? [...acknowledged, finding.id] : acknowledged.filter(id => id !== finding.id))} />
        <span>{finding.description}</span>
      </label> : <p>{finding.description}</p>}
    </li>)}</ul>
    {review.setup_guidance.length > 0 && <section><p>{t.catalog.provider.setup}</p>{review.setup_guidance.map(step => <div key={step.id}><h4>{step.summary}</h4><p className="whitespace-pre-wrap">{step.detail}</p></div>)}</section>}
  </>
}

function ReviewPlan(props: ReviewPlanProps) {
  const { t } = useI18n()
  const c = t.catalog.provider
  const { review, busy, acknowledged, onConfirm } = props
  const { required, confirmable } = reviewConsent(review)
  const expired = Boolean(review.expires_at_ms && Date.now() >= review.expires_at_ms)
  const disabled = busy || expired || acknowledged.length !== required.length || !required.every(id => acknowledged.includes(id))

  return <>
    <p>{t.catalog.version}: {review.version}</p>
    <p>{t.catalog.installTo}: {review.target_relative_path}</p>
    <ul>{review.files.map(file => <li key={file.path}><code>{file.path}</code> · {file.change} · {file.size}</li>)}</ul>
    <ReviewFindings {...props} />
    {expired && <p role="alert">{c.changed}</p>}
    {confirmable ? <Button disabled={disabled} onClick={onConfirm} size="sm">{c.confirm(c.actions[review.action])}</Button> : <p role="status">{c.incomplete}</p>}
  </>
}

export function ProviderReviewDialog({ controller }: { controller: ReturnType<typeof useProviderAction> }) {
  const { t } = useI18n()
  const c = t.catalog.provider
  const state = controller.state!
  const [acknowledged, setAcknowledged] = useState<string[]>([])
  const operation = state.operation
  const confirm = () => void controller.confirm(acknowledged)

  return <Dialog onOpenChange={open => { if (!open) { controller.close() } }} open>
    <DialogContent className="max-h-[85vh] overflow-y-auto">
      <DialogHeader>
        <DialogTitle>{c.review(c.actions[state.action])}</DialogTitle>
        <DialogDescription>{state.review?.summary ?? state.identifier}</DialogDescription>
      </DialogHeader>
      {state.busy && <p role="status">{t.skills.loading}</p>}
      {state.error && <p role="alert">{state.error}</p>}
      {operation ? <>
        <p role="status">{operation.outcome ?? c.status[operation.state]}</p>
        <p>{operation.message}</p>
        {operation.settlement === 'reconciling' && <p role="status">{c.finalizing}</p>}
        {controller.operationError && <Button onClick={controller.retryStatus} size="sm" variant="secondary">{t.catalog.retry}</Button>}
      </> : state.uncertain ? <>
        <p role="status">{c.unknown}</p>
        <Button disabled={state.busy} onClick={confirm} size="sm">{c.recover}</Button>
      </> : state.review ? <ReviewPlan acknowledged={acknowledged} busy={state.busy} onAcknowledge={setAcknowledged} onConfirm={confirm} review={state.review} /> : !state.busy && <p>{c.changed}</p>}
      <Button disabled={state.busy} onClick={controller.close} size="sm" variant="secondary">{t.common.close}</Button>
    </DialogContent>
  </Dialog>
}
