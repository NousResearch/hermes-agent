import { AlertTriangle, CheckCircle2, CircleHelp, Loader2, XCircle } from 'lucide-react'

import type { WisdomReviewCheck, WisdomReviewCheckRow, WisdomReviewStatus } from '@/hermes'

const PROFESSIONALISM_LABELS: Record<string, string> = {
  profanity_or_abuse: 'Profanity or abusive language',
  hate_or_harassment: 'Hate or harassment',
  sexual_or_graphic_language: 'Sexual or graphic language',
  manipulative_or_spam: 'Manipulative, deceptive, or spam-like wording'
}

function statusLabel(status: string): string {
  return status.replaceAll('_', ' ').replace(/^./, value => value.toUpperCase())
}

function StatusIcon({ status }: { status: WisdomReviewStatus }) {
  if (status === 'pass') {
    return <CheckCircle2 aria-hidden className="size-3" />
  }

  if (status === 'blocked') {
    return <XCircle aria-hidden className="size-3" />
  }

  if (status === 'pending' || status === 'retry' || status === 'running') {
    return <Loader2 aria-hidden className="size-3 animate-spin" />
  }

  if (status === 'advisory') {
    return <AlertTriangle aria-hidden className="size-3" />
  }

  return <CircleHelp aria-hidden className="size-3" />
}

function tone(status: WisdomReviewStatus): string {
  if (status === 'pass') {
    return 'border-emerald-500/50 text-emerald-600'
  }

  if (status === 'blocked') {
    return 'border-destructive/60 text-destructive'
  }

  if (status === 'advisory') {
    return 'border-amber-500/60 text-amber-500'
  }

  return 'border-(--ui-stroke-tertiary) text-muted-foreground'
}

export function WisdomCheckBadge({ label, value }: { label: string; value?: null | WisdomReviewCheck }) {
  const status = value?.status ?? 'unavailable'

  return (
    <span className={cnStatus(status)}>
      <StatusIcon status={status} />
      {label}: {statusLabel(status)}
    </span>
  )
}

function cnStatus(status: WisdomReviewStatus): string {
  return `inline-flex items-center gap-1 rounded-sm border px-1.5 py-0.5 text-[0.6rem] ${tone(status)}`
}

function rowLabel(row: WisdomReviewCheckRow): string {
  return row.label || PROFESSIONALISM_LABELS[row.key] || statusLabel(row.key)
}

function CheckTable({ label, note, value }: { label: string; note: string; value?: null | WisdomReviewCheck }) {
  const status = value?.status ?? 'unavailable'
  const rows = value?.checks ?? []

  return (
    <section aria-label={label} className="border-t border-(--ui-stroke-tertiary) py-3 first:border-0">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h3 className="text-xs font-medium">{label}</h3>
        <WisdomCheckBadge label="Result" value={{ status }} />
      </div>
      {value?.summary && <p className="mt-1 text-[0.65rem] text-muted-foreground">{value.summary}</p>}
      <p className="mt-1 text-[0.6rem] text-muted-foreground">{note}</p>
      {rows.length > 0 && (
        <div className="mt-2 overflow-x-auto">
          <table className="w-full border-collapse text-left text-[0.65rem]">
            <thead className="text-muted-foreground">
              <tr>
                <th className="border-b border-(--ui-stroke-tertiary) py-2 pr-3 font-medium">Check</th>
                <th className="border-b border-(--ui-stroke-tertiary) py-2 pr-3 font-medium">Status</th>
                <th className="border-b border-(--ui-stroke-tertiary) py-2 font-medium">Details</th>
              </tr>
            </thead>
            <tbody>
              {rows.map(row => (
                <tr key={row.key}>
                  <th className="border-b border-(--ui-stroke-tertiary) py-2 pr-3 font-normal">{rowLabel(row)}</th>
                  <td className="border-b border-(--ui-stroke-tertiary) py-2 pr-3">
                    <span className="inline-flex items-center gap-1">
                      <StatusIcon status={row.status} /> {statusLabel(row.status)}
                    </span>
                  </td>
                  <td className="border-b border-(--ui-stroke-tertiary) py-2 text-muted-foreground">
                    {row.details.length > 0
                      ? row.details.join(' ')
                      : row.finding_count > 0
                        ? `${row.finding_count} finding${row.finding_count === 1 ? '' : 's'}`
                        : 'No known matches detected'}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  )
}

export function WisdomReviewTables({
  professionalism,
  security
}: {
  professionalism?: null | WisdomReviewCheck
  security?: null | WisdomReviewCheck
}) {
  return (
    <div className="mt-3 border-y border-(--ui-stroke-tertiary)">
      <CheckTable
        label="Security check"
        note="Deterministic Gateway scan. A pass means no known matches were detected, not that the package is certified secure."
        value={security}
      />
      <CheckTable
        label="Professionalism check"
        note="Agent-assessed and advisory. It does not block publication or installation."
        value={professionalism}
      />
    </div>
  )
}
