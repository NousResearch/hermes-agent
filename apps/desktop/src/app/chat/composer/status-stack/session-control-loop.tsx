import { memo, useCallback, useState } from 'react'

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
  type SessionControlLoop
} from '@/store/session-control'

import { type ConfirmState, formatApproximateTime, formatInterval } from './session-control-utils'

interface LoopSectionProps {
  loop: SessionControlLoop
  sessionId: string
  pendingAction: SessionControlAction | null
  onFeedback: (error: string | null, success: string | null) => void
}

export const SessionControlLoopSection = memo(function SessionControlLoopSection({
  loop,
  sessionId,
  pendingAction,
  onFeedback
}: LoopSectionProps) {
  const { t } = useI18n()
  const s = t.statusStack
  const ctrl = s.control

  const [confirmState, setConfirmState] = useState<ConfirmState | null>(null)
  const [menuOpen, setMenuOpen] = useState(false)
  const isBusy = Boolean(pendingAction)

  const iconClass =
    loop.status === 'done'
      ? 'text-muted-foreground/70'
      : loop.status === 'paused' || loop.deferred_by_goal
        ? 'text-amber-500'
        : 'text-emerald-500'

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

  const stateLabel = loop.deferred_by_goal
    ? ctrl.loopDeferred
    : loop.status === 'paused'
      ? ctrl.loopPaused
      : loop.status === 'done'
        ? ctrl.loopFinished
        : loop.awaiting_response
          ? `${ctrl.loopActive} · ${ctrl.loopAwaitingResponse}`
          : ctrl.loopActive

  const runCountLabel =
    loop.times > 0 ? ctrl.loopRunCount(loop.ticks_fired, loop.times) : ctrl.loopRuns(loop.ticks_fired)

  const nextRunLabel =
    loop.status === 'active' && !loop.deferred_by_goal && loop.next_due_at
      ? ` · ${ctrl.loopNext(formatApproximateTime(loop.next_due_at))}`
      : ''

  const headerLabel = `${stateLabel} · ${runCountLabel}${nextRunLabel}`

  const confirmStopLoop = () => {
    if (loop.status === 'done') {
      void handleAction('loop.stop')

      return
    }

    setConfirmState({
      title: ctrl.stopLoopConfirmTitle,
      description: ctrl.stopLoopConfirmBody,
      destructive: true,
      confirmLabel: ctrl.stopLoop,
      onConfirm: async () => {
        await handleAction('loop.stop')
      }
    })
  }

  const renderMenuItems = (isContext = false) => {
    const Item = isContext ? ContextMenuItem : DropdownMenuItem
    const Sep = isContext ? ContextMenuSeparator : DropdownMenuSeparator

    return (
      <>
        {loop.status === 'active' && (
          <Item disabled={isBusy} onSelect={() => void handleAction('loop.pause')}>
            <Codicon name="debug-pause" size="0.8rem" />
            <span>{ctrl.pauseLoop}</span>
          </Item>
        )}
        {loop.status === 'paused' && (
          <Item disabled={isBusy} onSelect={() => void handleAction('loop.resume')}>
            <Codicon name="play" size="0.8rem" />
            <span>{ctrl.resumeLoop}</span>
          </Item>
        )}
        {loop.status !== 'done' && <Sep />}
        {loop.status === 'done' ? (
          <Item disabled={isBusy} onSelect={() => void handleAction('loop.stop')}>
            <Codicon name="close" size="0.8rem" />
            <span>{ctrl.dismissLoop}</span>
          </Item>
        ) : (
          <Item disabled={isBusy} onSelect={confirmStopLoop} variant="destructive">
            <Codicon name="stop-circle" size="0.8rem" />
            <span>{ctrl.stopLoop}</span>
          </Item>
        )}
      </>
    )
  }

  return (
    <>
      <ContextMenu>
        <ContextMenuTrigger asChild>
          <div data-slot="session-control-loop">
            <StatusSection
              accessory={
                <DropdownMenu onOpenChange={setMenuOpen} open={menuOpen}>
                  <Tip label={ctrl.loopActions}>
                    <span className="inline-flex">
                      <DropdownMenuTrigger asChild>
                        <Button
                          aria-haspopup="menu"
                          aria-label={ctrl.loopActions}
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
                  <DropdownMenuContent align="end" className="w-40">
                    {renderMenuItems(false)}
                  </DropdownMenuContent>
                </DropdownMenu>
              }
              defaultCollapsed={true}
              icon={<Codicon className={iconClass} name="sync" size="0.8rem" />}
              label={headerLabel}
            >
              <div className="space-y-1 px-1 py-1 text-xs">
                <div className="text-foreground/92 leading-relaxed break-words">{loop.prompt}</div>
                <div className="text-[0.7rem] text-muted-foreground/80">
                  <span>
                    {ctrl.loopCadenceLabel}:{' '}
                    {loop.mode === 'self_paced' ? ctrl.loopSelfPaced : formatInterval(loop.interval_seconds, t)}
                  </span>
                </div>
                {loop.until && (
                  <div className="text-[0.7rem] text-muted-foreground/80">
                    <span>
                      {ctrl.loopUntilLabel}: {loop.until}
                    </span>
                  </div>
                )}
                {loop.deferred_by_goal && (
                  <div className="text-[0.7rem] italic text-muted-foreground/80">{ctrl.loopDeferredNotice}</div>
                )}
                {loop.awaiting_response && (
                  <div className="text-[0.7rem] italic text-muted-foreground/80">{ctrl.loopAwaitingResponse}</div>
                )}
                {loop.paused_reason && (
                  <div className="text-[0.7rem] italic text-muted-foreground/80">{loop.paused_reason}</div>
                )}
                {loop.last_stop_reason && (
                  <div className="text-[0.7rem] italic text-muted-foreground/80">{loop.last_stop_reason}</div>
                )}
              </div>
            </StatusSection>
          </div>
        </ContextMenuTrigger>
        <ContextMenuContent className="w-40">{renderMenuItems(true)}</ContextMenuContent>
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
