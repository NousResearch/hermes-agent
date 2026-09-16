import { memo, useCallback, useEffect, useState } from 'react'

import { StatusControlRow } from '@/components/chat/status-control-row'
import { StatusSection } from '@/components/chat/status-section'
import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { ConfirmDialog } from '@/components/ui/confirm-dialog'
import {
  ContextMenu,
  ContextMenuContent,
  ContextMenuItem,
  ContextMenuSeparator,
  ContextMenuTrigger
} from '@/components/ui/context-menu'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle
} from '@/components/ui/dialog'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger
} from '@/components/ui/dropdown-menu'
import { Field } from '@/components/ui/field'
import { Input } from '@/components/ui/input'
import { Textarea } from '@/components/ui/textarea'
import { useI18n } from '@/i18n'
import {
  runSessionControlAction,
  type SessionControlAction,
  type SessionControlActionArgs,
  type SessionControlHeartbeat
} from '@/store/session-control'

import {
  type ConfirmState,
  formatHeartbeatCountdown,
  formatHeartbeatInterval,
  formatHeartbeatIntervalInput
} from './session-control-utils'

interface HeartbeatSectionProps {
  heartbeat: SessionControlHeartbeat
  sessionId: string
  pendingAction: SessionControlAction | null
  onFeedback: (error: string | null, success: string | null) => void
}

function useHeartbeatClock(active: boolean): number {
  const [now, setNow] = useState(() => Date.now())

  useEffect(() => {
    if (!active) {
      return
    }

    const tick = () => setNow(Date.now())
    const interval = window.setInterval(tick, 1_000)

    return () => window.clearInterval(interval)
  }, [active])

  return now
}

export const SessionControlHeartbeatSection = memo(function SessionControlHeartbeatSection({
  heartbeat,
  sessionId,
  pendingAction,
  onFeedback
}: HeartbeatSectionProps) {
  const { t } = useI18n()
  const s = t.statusStack
  const ctrl = s.control

  const [confirmState, setConfirmState] = useState<ConfirmState | null>(null)
  const [menuOpen, setMenuOpen] = useState(false)
  const [editOpen, setEditOpen] = useState(false)
  const [editError, setEditError] = useState<string | null>(null)
  const [editPrompt, setEditPrompt] = useState('')
  const [editInterval, setEditInterval] = useState('')
  // What the dialog staged when it opened: parsing changes against it keeps an untouched field out of
  // the request, so a newer edit from another window is not overwritten by this dialog's stale copy.
  const [editBaseline, setEditBaseline] = useState({ interval: '', prompt: '' })
  const isBusy = Boolean(pendingAction)

  const handleAction = useCallback(
    async (
      action: SessionControlAction,
      args?: SessionControlActionArgs,
      onFailure?: (message: string) => void
    ): Promise<boolean> => {
      onFeedback(null, null)

      try {
        await runSessionControlAction(sessionId, action, args)
        onFeedback(null, ctrl.actionSucceeded)

        return true
      } catch (err) {
        const msg = err instanceof Error ? err.message : String(err)
        const failure = ctrl.actionFailed(msg)
        onFeedback(failure, null)
        onFailure?.(failure)

        return false
      }
    },
    [sessionId, onFeedback, ctrl]
  )

  const openEditHeartbeat = useCallback(() => {
    const prompt = heartbeat.prompt
    const interval = formatHeartbeatIntervalInput(heartbeat.interval_seconds)

    setEditPrompt(prompt)
    setEditInterval(interval)
    setEditBaseline({ interval, prompt })
    setEditError(null)
    setEditOpen(true)
  }, [heartbeat.interval_seconds, heartbeat.prompt])

  const stateLabel = heartbeat.status === 'paused' ? ctrl.heartbeatPaused : ctrl.heartbeatActive
  const now = useHeartbeatClock(heartbeat.status === 'active')
  const intervalLabel = formatHeartbeatInterval(heartbeat.interval_seconds, t)
  const nextDueTimestamp = (heartbeat.last_fired_at || heartbeat.created_at) + heartbeat.interval_seconds
  const nextDueMs = nextDueTimestamp > 1e11 ? nextDueTimestamp : nextDueTimestamp * 1_000
  const isDue = heartbeat.status === 'active' && nextDueMs <= now

  const nextRunLabel =
    heartbeat.status === 'active'
      ? ` · ${isDue ? ctrl.heartbeatDueWaitingForIdle : ctrl.heartbeatNext(formatHeartbeatCountdown(nextDueTimestamp, now))}`
      : ''

  const iconClass = heartbeat.status === 'paused' ? 'text-red-500' : isDue ? 'text-amber-500' : 'text-emerald-500'

  const headerLabel = `${stateLabel} · ${intervalLabel}${nextRunLabel}`

  const confirmClearHeartbeat = () => {
    setConfirmState({
      title: ctrl.clearHeartbeatConfirmTitle,
      description: ctrl.clearHeartbeatConfirmBody,
      destructive: true,
      confirmLabel: ctrl.clearHeartbeat,
      onConfirm: async () => {
        await handleAction('heartbeat.clear')
      }
    })
  }

  const renderMenuItems = (isContext = false) => {
    const Item = isContext ? ContextMenuItem : DropdownMenuItem
    const Sep = isContext ? ContextMenuSeparator : DropdownMenuSeparator

    return (
      <>
        <Item disabled={isBusy} onSelect={openEditHeartbeat}>
          <Codicon name="edit" size="0.8rem" />
          <span>{ctrl.editHeartbeat}</span>
        </Item>
        <Sep />
        {heartbeat.status === 'active' && (
          <Item disabled={isBusy} onSelect={() => void handleAction('heartbeat.pause')}>
            <Codicon name="debug-pause" size="0.8rem" />
            <span>{ctrl.pauseHeartbeat}</span>
          </Item>
        )}
        {heartbeat.status === 'paused' && (
          <Item disabled={isBusy} onSelect={() => void handleAction('heartbeat.resume')}>
            <Codicon name="play" size="0.8rem" />
            <span>{ctrl.resumeHeartbeat}</span>
          </Item>
        )}
        <Sep />
        <Item disabled={isBusy} onSelect={confirmClearHeartbeat} variant="destructive">
          <Codicon name="trash" size="0.8rem" />
          <span>{ctrl.clearHeartbeat}</span>
        </Item>
      </>
    )
  }

  return (
    <>
      <ContextMenu>
        <ContextMenuTrigger asChild>
          <div data-slot="session-control-heartbeat">
            <StatusSection
              accessory={
                <DropdownMenu onOpenChange={setMenuOpen} open={menuOpen}>
                  <span className="inline-flex">
                    <DropdownMenuTrigger asChild>
                      <Button
                        aria-haspopup="menu"
                        aria-label={ctrl.heartbeatActions}
                        className="size-6 rounded-md text-muted-foreground/70 hover:text-foreground/90"
                        disabled={isBusy}
                        onClick={event => {
                          // Radix opens pointer interactions from pointerdown. Keyboard,
                          // assistive-tech, and programmatic clicks have no pointer sequence.
                          if (event.detail === 0) {
                            setMenuOpen(true)
                          }
                        }}
                        onKeyDown={e => {
                          if (e.key === 'F10' && e.shiftKey) {
                            e.preventDefault()
                            setMenuOpen(true)
                          }
                        }}
                        size="icon-xs"
                        type="button"
                        variant="ghost"
                      >
                        <Codicon name="ellipsis" size="0.8rem" />
                      </Button>
                    </DropdownMenuTrigger>
                  </span>
                  <DropdownMenuContent align="end" className="w-44">
                    {renderMenuItems(false)}
                  </DropdownMenuContent>
                </DropdownMenu>
              }
              defaultCollapsed={true}
              icon={<Codicon className={iconClass} name="pulse" size="0.8rem" />}
              label={headerLabel}
            >
              <div>
                <StatusControlRow className="text-[0.73rem] leading-4 text-foreground/92 break-words" icon="bell">
                  {heartbeat.prompt}
                </StatusControlRow>
                <StatusControlRow icon="history">{ctrl.heartbeatFiredCount(heartbeat.fire_count)}</StatusControlRow>
              </div>
            </StatusSection>
          </div>
        </ContextMenuTrigger>
        <ContextMenuContent className="w-44">{renderMenuItems(true)}</ContextMenuContent>
      </ContextMenu>

      <Dialog onOpenChange={setEditOpen} open={editOpen}>
        <DialogContent className="max-w-md">
          <form
            onSubmit={async e => {
              e.preventDefault()
              const prompt = editPrompt.trim()
              const interval = editInterval.trim()

              if (!prompt || !interval || isBusy) {
                return
              }

              // Send only the fields the user actually changed: the staged snapshot can be older than the
              // live heartbeat (another window edited it meanwhile), and echoing an untouched field would
              // silently restore its superseded value.
              const changes =
                prompt !== editBaseline.prompt && interval !== editBaseline.interval
                  ? { interval, prompt }
                  : prompt !== editBaseline.prompt
                    ? { prompt }
                    : interval !== editBaseline.interval
                      ? { interval }
                      : null

              if (!changes) {
                setEditOpen(false)

                return
              }

              setEditError(null)
              const ok = await handleAction('heartbeat.update', changes, setEditError)

              if (ok) {
                setEditOpen(false)
              }
            }}
          >
            <DialogHeader>
              <DialogTitle>{ctrl.editHeartbeat}</DialogTitle>
              <DialogDescription>{ctrl.editHeartbeatDescription}</DialogDescription>
            </DialogHeader>
            <div className="grid gap-4 py-2">
              <Field htmlFor="heartbeat-edit-message" label={ctrl.heartbeatMessageLabel}>
                <Textarea
                  autoFocus
                  disabled={isBusy}
                  id="heartbeat-edit-message"
                  onChange={event => setEditPrompt(event.target.value)}
                  rows={3}
                  value={editPrompt}
                />
              </Field>
              <Field htmlFor="heartbeat-edit-interval" label={ctrl.heartbeatFrequencyLabel}>
                <Input
                  disabled={isBusy}
                  id="heartbeat-edit-interval"
                  onChange={event => setEditInterval(event.target.value)}
                  placeholder={ctrl.heartbeatFrequencyPlaceholder}
                  value={editInterval}
                />
              </Field>
            </div>
            {editError && (
              <div className="text-xs text-destructive" role="alert">
                {editError}
              </div>
            )}
            <DialogFooter>
              <Button disabled={isBusy} onClick={() => setEditOpen(false)} type="button" variant="ghost">
                {t.common.cancel}
              </Button>
              <Button disabled={isBusy || !editPrompt.trim() || !editInterval.trim()} type="submit">
                {t.common.save}
              </Button>
            </DialogFooter>
          </form>
        </DialogContent>
      </Dialog>

      {confirmState && (
        <ConfirmDialog
          cancelLabel={t.common.cancel}
          confirmLabel={confirmState.confirmLabel}
          description={confirmState.description}
          destructive={confirmState.destructive}
          onClose={() => setConfirmState(null)}
          onConfirm={confirmState.onConfirm}
          open={Boolean(confirmState)}
          title={confirmState.title}
        />
      )}
    </>
  )
})
