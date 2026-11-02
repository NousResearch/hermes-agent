// Updates overlay states for an in-app update that would interrupt agent work:
// the choice ("update now" interrupts, "update when idle" waits) and the
// cancellable wait. Copy lives in t.updates.whenIdle.

import { Button } from '@/components/ui/button'
import { DialogDescription, DialogTitle } from '@/components/ui/dialog'
import { Loader } from '@/components/ui/loader'
import type { DesktopUpdateBusyReason } from '@/global'
import { useI18n } from '@/i18n'
import type { UpdateWhenIdleCopy } from '@/i18n/types_update_when_idle'
import { AlertCircle } from '@/lib/icons'

const MAX_TITLES = 3

const REASON_LABEL: {
  [K in DesktopUpdateBusyReason['kind']]: (
    reason: Extract<DesktopUpdateBusyReason, { kind: K }>,
    copy: UpdateWhenIdleCopy
  ) => string
} = {
  chats: (reason, copy) => copy.chats(reason.count),
  cron: (reason, copy) => copy.cron(reason.jobs.join(', ')),
  delegation: (_reason, copy) => copy.delegation,
  input: (_reason, copy) => copy.input,
  unknown: (_reason, copy) => copy.unknown
}

function reasonLabel(reason: DesktopUpdateBusyReason, copy: UpdateWhenIdleCopy): string {
  return (REASON_LABEL[reason.kind] as (reason: DesktopUpdateBusyReason, copy: UpdateWhenIdleCopy) => string)(
    reason,
    copy
  )
}

/** How many things "update now" interrupts: each chat and cron job, one per other reason. */
export function interruptedCount(reasons: readonly DesktopUpdateBusyReason[]): number {
  return reasons.reduce(
    (total, reason) =>
      total + (reason.kind === 'chats' ? reason.count : reason.kind === 'cron' ? Math.max(1, reason.jobs.length) : 1),
    0
  )
}

function ReasonList({ reasons }: { reasons: readonly DesktopUpdateBusyReason[] }) {
  const { t } = useI18n()
  const copy = t.updates.whenIdle

  return (
    <ul className="grid gap-1.5 rounded-md border border-border/70 bg-muted/35 px-3 py-2 text-left text-sm">
      {reasons.map(reason => (
        <li key={reason.kind === 'cron' ? `cron:${reason.jobs.join(',')}` : reason.kind}>
          <span className="font-medium">{reasonLabel(reason, copy)}</span>
          {reason.kind === 'chats' && reason.titles.length > 0 ? (
            <span className="block truncate text-xs text-muted-foreground">
              {reason.titles.slice(0, MAX_TITLES).join(' · ')}
            </span>
          ) : null}
        </li>
      ))}
    </ul>
  )
}

export function UpdateBusyView({
  onLater,
  onUpdateNow,
  onUpdateWhenIdle,
  reasons
}: {
  onLater: () => void
  onUpdateNow: () => void
  onUpdateWhenIdle: () => void
  reasons: readonly DesktopUpdateBusyReason[]
}) {
  const { t } = useI18n()
  const copy = t.updates.whenIdle

  return (
    <div className="grid gap-5 px-6 pb-6 pt-7 pr-8">
      <div className="flex flex-col items-center gap-3 text-center">
        <AlertCircle className="size-8 text-amber-500" />
        <DialogTitle className="text-center text-xl">{copy.busyTitle}</DialogTitle>
        <DialogDescription className="text-center text-sm text-muted-foreground">{copy.busyBody}</DialogDescription>
      </div>

      <ReasonList reasons={reasons} />

      <div className="grid gap-2">
        <Button className="font-semibold" onClick={onUpdateWhenIdle} size="lg">
          {copy.updateWhenIdle}
        </Button>
        <Button onClick={onUpdateNow} variant="secondary">
          {copy.updateNow(interruptedCount(reasons))}
        </Button>
        <Button onClick={onLater} variant="text">
          {t.updates.notNow}
        </Button>
      </div>
    </div>
  )
}

export function UpdateWaitingView({
  deadlineAt,
  onCancel,
  reasons
}: {
  deadlineAt: number | null
  onCancel: () => void
  reasons: readonly DesktopUpdateBusyReason[]
}) {
  const { t } = useI18n()
  const copy = t.updates.whenIdle

  const until =
    deadlineAt === null ? null : new Date(deadlineAt).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })

  return (
    <div className="grid gap-5 px-6 pb-6 pt-7 pr-8">
      <div className="flex flex-col items-center gap-3 text-center">
        <Loader className="size-12" label={copy.waitingTitle} type="lemniscate-bloom" />
        <DialogTitle className="text-center text-xl">{copy.waitingTitle}</DialogTitle>
        <DialogDescription className="text-center text-sm text-muted-foreground">{copy.waitingBody}</DialogDescription>
      </div>

      {reasons.length > 0 ? (
        <div className="grid gap-1.5">
          <p className="text-xs font-medium text-muted-foreground">{copy.waitingFor}</p>
          <ReasonList reasons={reasons} />
        </div>
      ) : (
        <p className="text-center text-sm">{copy.waitingIdle}</p>
      )}

      {until ? <p className="text-center text-xs text-muted-foreground">{copy.waitingUntil(until)}</p> : null}

      <Button onClick={onCancel} variant="secondary">
        {copy.cancel}
      </Button>
    </div>
  )
}
