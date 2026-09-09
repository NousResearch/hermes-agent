import { memo, useCallback, useEffect, useState } from 'react'

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
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger
} from '@/components/ui/dropdown-menu'
import { Tip } from '@/components/ui/tooltip'
import { useI18n } from '@/i18n'
import {
  runSessionControlAction,
  type SessionControlAction,
  type SessionControlActionArgs,
  type SessionControlHeartbeat
} from '@/store/session-control'

import { type ConfirmState, formatHeartbeatCountdown, formatHeartbeatInterval } from './session-control-utils'

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
  const isBusy = Boolean(pendingAction)

  const handleAction = useCallback(
    async (action: SessionControlAction, args?: SessionControlActionArgs): Promise<boolean> => {
      onFeedback(null, null)

      try {
        await runSessionControlAction(sessionId, action, args)
        onFeedback(null, ctrl.actionSucceeded)

        return true
      } catch (err) {
        const msg = err instanceof Error ? err.message : String(err)
        onFeedback(ctrl.actionFailed(msg), null)

        return false
      }
    },
    [sessionId, onFeedback, ctrl]
  )

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
                  <Tip label={ctrl.heartbeatActions}>
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
                  </Tip>
                  <DropdownMenuContent align="end" className="w-44">
                    {renderMenuItems(false)}
                  </DropdownMenuContent>
                </DropdownMenu>
              }
              defaultCollapsed={true}
              icon={<Codicon className={iconClass} name="pulse" size="0.8rem" />}
              label={headerLabel}
            >
              <div className="space-y-1 px-1 py-1 text-xs">
                <div className="text-foreground/92 leading-relaxed break-words">{heartbeat.prompt}</div>
                <div className="text-[0.7rem] text-muted-foreground/80">
                  <span>{ctrl.heartbeatFiredCount(heartbeat.fire_count)}</span>
                </div>
              </div>
            </StatusSection>
          </div>
        </ContextMenuTrigger>
        <ContextMenuContent className="w-44">{renderMenuItems(true)}</ContextMenuContent>
      </ContextMenu>

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
