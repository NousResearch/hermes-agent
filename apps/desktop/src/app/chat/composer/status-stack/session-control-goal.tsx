import { memo, useCallback, useState } from 'react'

import { queueKickoffIfSessionBusy } from '@/app/session/hooks/use-prompt-actions/queue-if-busy'
import type { SubmitTextOptions } from '@/app/session/hooks/use-prompt-actions/utils'
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
import { Tip } from '@/components/ui/tooltip'
import { useI18n } from '@/i18n'
import {
  runSessionControlAction,
  type SessionControlAction,
  type SessionControlActionArgs,
  type SessionControlGoal
} from '@/store/session-control'

import type { ConfirmState } from './session-control-utils'

interface GoalSectionProps {
  goal: SessionControlGoal
  sessionId: string
  pendingAction: SessionControlAction | null
  onSubmit?: (value: string, options?: SubmitTextOptions) => Promise<boolean> | boolean
  onFeedback: (error: string | null, success: string | null) => void
}

export const SessionControlGoalSection = memo(function SessionControlGoalSection({
  goal,
  sessionId,
  pendingAction,
  onSubmit,
  onFeedback
}: GoalSectionProps) {
  const { t } = useI18n()
  const s = t.statusStack
  const ctrl = s.control

  const [detailsOpen, setDetailsOpen] = useState(false)
  const [addCriterionOpen, setAddCriterionOpen] = useState(false)
  const [addCriterionError, setAddCriterionError] = useState<string | null>(null)
  const [newCriterionText, setNewCriterionText] = useState('')
  const [confirmState, setConfirmState] = useState<ConfirmState | null>(null)
  const [menuOpen, setMenuOpen] = useState(false)

  const isBusy = Boolean(pendingAction)

  const handleAction = useCallback(
    async (
      action: SessionControlAction,
      args?: SessionControlActionArgs,
      onFailure?: (message: string) => void
    ): Promise<boolean> => {
      onFeedback(null, null)

      try {
        const dispatch = await runSessionControlAction(sessionId, action, args)

        if (dispatch.type === 'send') {
          if (!dispatch.message || !onSubmit) {
            onFeedback(ctrl.continuationFailed, null)
            onFailure?.(ctrl.continuationFailed)

            return false
          }

          // The backend has already resumed the goal; if a turn is running the
          // kickoff must queue (same as a typed `/goal resume`, slash.ts), not
          // read as a failure.
          const queued = queueKickoffIfSessionBusy({
            displayText: dispatch.display ?? undefined,
            sessionId,
            text: dispatch.message
          })

          if (queued !== 'idle') {
            const copy = queued === 'queued' ? ctrl.continuationQueued : ctrl.continuationBusy
            onFeedback(queued === 'queued' ? null : copy, queued === 'queued' ? copy : null)

            return queued === 'queued'
          }

          const submitted = await onSubmit(dispatch.message, {
            displayKind: 'hidden',
            sessionId
          })

          if (!submitted) {
            onFeedback(ctrl.continuationFailed, null)
            onFailure?.(ctrl.continuationFailed)

            return false
          }
        }

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
    [sessionId, onSubmit, onFeedback, ctrl]
  )

  const copyCriterionText = useCallback(
    async (text: string) => {
      try {
        await navigator.clipboard.writeText(text)
        onFeedback(null, ctrl.copySuccess)
      } catch {
        onFeedback(ctrl.copyFailure, null)
      }
    },
    [onFeedback, ctrl]
  )

  const visibleState: 'waiting' | 'active' | 'paused' | 'done' = goal.wait_barrier
    ? 'waiting'
    : goal.status === 'paused'
      ? 'paused'
      : goal.status === 'done'
        ? 'done'
        : 'active'

  const iconClass =
    goal.last_verdict === 'blocked'
      ? 'text-red-500'
      : visibleState === 'done'
        ? 'text-muted-foreground/70'
        : visibleState === 'active'
          ? 'text-emerald-500'
          : 'text-amber-500'

  const stateLabel =
    goal.last_verdict === 'blocked'
      ? s.goalBlocked
      : visibleState === 'waiting'
        ? s.goalWaiting
        : visibleState === 'paused'
          ? s.goalPaused
          : visibleState === 'done'
            ? s.goalDone
            : s.goalActive

  const headerLabel =
    visibleState === 'done'
      ? `${stateLabel} · ${ctrl.goalDoneTurns(goal.turns_used)}`
      : goal.max_turns > 0
        ? `${stateLabel} · ${ctrl.goalActiveTurns(goal.turns_used, goal.max_turns)}`
        : goal.turns_used > 0
          ? `${stateLabel} · ${ctrl.goalTurn(goal.turns_used)}`
          : stateLabel

  const confirmClearGoal = () => {
    setConfirmState({
      title: ctrl.clearGoalConfirmTitle,
      description: ctrl.clearGoalConfirmBody,
      destructive: true,
      confirmLabel: ctrl.clearGoal,
      onConfirm: async () => {
        await handleAction('goal.clear')
      }
    })
  }

  const confirmRemoveCriterion = (index: number) => {
    setConfirmState({
      title: ctrl.removeCriterionConfirmTitle(index),
      description: ctrl.removeCriterionConfirmBody(index),
      destructive: true,
      confirmLabel: ctrl.removeCriterion(index),
      onConfirm: async () => {
        await handleAction('subgoal.remove', { index })
      }
    })
  }

  const confirmClearCriteria = () => {
    setConfirmState({
      title: ctrl.clearCriteriaConfirmTitle,
      description: ctrl.clearCriteriaConfirmBody,
      destructive: true,
      confirmLabel: ctrl.clearCriteria,
      onConfirm: async () => {
        await handleAction('subgoal.clear')
      }
    })
  }

  const openAddCriterion = () => {
    setAddCriterionError(null)
    setAddCriterionOpen(true)
  }

  const renderMenuItems = (isContext = false) => {
    const Item = isContext ? ContextMenuItem : DropdownMenuItem
    const Sep = isContext ? ContextMenuSeparator : DropdownMenuSeparator

    return (
      <>
        {hasDetails && (
          <Item disabled={isBusy} onSelect={() => setDetailsOpen(true)}>
            <Codicon name="eye" size="0.8rem" />
            <span>{ctrl.viewDetails}</span>
          </Item>
        )}
        {visibleState !== 'done' && (
          <Item disabled={isBusy} onSelect={openAddCriterion}>
            <Codicon name="add" size="0.8rem" />
            <span>{ctrl.addCriterion}</span>
          </Item>
        )}
        {(hasDetails || visibleState !== 'done') && <Sep />}
        {visibleState === 'active' && (
          <Item disabled={isBusy} onSelect={() => void handleAction('goal.pause')}>
            <Codicon name="debug-pause" size="0.8rem" />
            <span>{ctrl.pauseGoal}</span>
          </Item>
        )}
        {visibleState === 'paused' && (
          <Item disabled={isBusy} onSelect={() => void handleAction('goal.resume')}>
            <Codicon name="play" size="0.8rem" />
            <span>{ctrl.resumeGoal}</span>
          </Item>
        )}
        {visibleState === 'waiting' && (
          <>
            <Item disabled={isBusy} onSelect={() => void handleAction('goal.unwait')}>
              <Codicon name="play" size="0.8rem" />
              <span>{ctrl.resumeNow}</span>
            </Item>
            <Item disabled={isBusy} onSelect={() => void handleAction('goal.pause')}>
              <Codicon name="debug-pause" size="0.8rem" />
              <span>{ctrl.pauseGoal}</span>
            </Item>
          </>
        )}
        <Item disabled={isBusy} onSelect={confirmClearGoal} variant="destructive">
          <Codicon name="trash" size="0.8rem" />
          <span>{ctrl.clearGoal}</span>
        </Item>
      </>
    )
  }

  const hasDetails = Boolean(
    goal.contract.outcome ||
    goal.contract.verification ||
    goal.contract.constraints ||
    goal.contract.boundaries ||
    goal.contract.stop_when ||
    goal.wait_barrier ||
    goal.gates.length > 0
  )

  return (
    <>
      <ContextMenu>
        <ContextMenuTrigger asChild>
          <div data-slot="session-control-goal">
            <StatusSection
              accessory={
                <DropdownMenu onOpenChange={setMenuOpen} open={menuOpen}>
                  <Tip label={ctrl.goalActions}>
                    <span className="inline-flex">
                      <DropdownMenuTrigger asChild>
                        <Button
                          aria-haspopup="menu"
                          aria-label={ctrl.goalActions}
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
              defaultCollapsed={false}
              icon={<Codicon className={iconClass} name="target" size="0.8rem" />}
              label={headerLabel}
            >
              <div className="space-y-1.5 px-1 py-1">
                {/* Full goal title */}
                <div className="text-xs font-normal leading-relaxed text-foreground/92 break-words">{goal.title}</div>

                {/* Optional reasons */}
                {!detailsOpen && goal.wait_barrier && (
                  <div className="text-[0.7rem] italic text-muted-foreground/80">
                    {goal.wait_barrier.reason
                      ? `${ctrl.waitBarrierTitle}: ${goal.wait_barrier.reason}`
                      : ctrl.waitBarrierTitle}
                  </div>
                )}
                {!goal.wait_barrier && goal.paused_reason && (
                  <div className="text-[0.7rem] italic text-muted-foreground/80">{goal.paused_reason}</div>
                )}
                {!goal.wait_barrier && !goal.paused_reason && goal.last_reason && (
                  <div className="text-[0.7rem] italic text-muted-foreground/80">{goal.last_reason}</div>
                )}

                {/* View details button */}
                {hasDetails && (
                  <div>
                    <Button
                      className="text-[0.7rem] text-muted-foreground/75 hover:text-foreground/90"
                      onClick={() => setDetailsOpen(true)}
                      size="micro"
                      type="button"
                      variant="text"
                    >
                      {ctrl.viewDetails}
                    </Button>
                  </div>
                )}

                {/* Criteria subsection */}
                <div className="mt-2 border-t border-(--ui-stroke-tertiary)/40 pt-1.5">
                  <div className="flex items-center justify-between pb-1 text-[0.68rem] font-medium text-muted-foreground/75">
                    <div className="flex items-center gap-1.5">
                      <span aria-hidden="true" className={`inline-flex ${iconClass}`} data-slot="criteria-state-marker">
                        <Codicon name="target" size="0.68rem" />
                      </span>
                      <span>{ctrl.criteriaHeader(goal.subgoals.length)}</span>
                    </div>
                    <div className="flex items-center gap-1">
                      <Button
                        className="text-[0.68rem] text-muted-foreground/75 hover:text-foreground/90"
                        disabled={isBusy}
                        onClick={openAddCriterion}
                        size="micro"
                        type="button"
                        variant="text"
                      >
                        {ctrl.addCriterion}
                      </Button>
                      {goal.subgoals.length > 0 && (
                        <Button
                          className="text-[0.68rem] text-muted-foreground/60 hover:text-destructive"
                          disabled={isBusy}
                          onClick={confirmClearCriteria}
                          size="micro"
                          type="button"
                          variant="text"
                        >
                          {ctrl.clearCriteria}
                        </Button>
                      )}
                    </div>
                  </div>

                  {goal.subgoals.length > 0 ? (
                    <div className="space-y-1">
                      {goal.subgoals.map((subgoal, idx) => {
                        const index = idx + 1

                        return (
                          <div
                            className="group/criterion flex items-start justify-between gap-1.5 rounded px-1 py-0.5 text-xs text-foreground/85 hover:bg-(--ui-control-active-background)/40"
                            key={`${index}-${subgoal}`}
                          >
                            <div className="flex min-w-0 flex-1 items-start gap-1.5 leading-relaxed">
                              <span className="shrink-0 text-muted-foreground/60 tabular-nums">{index}.</span>
                              <span className="break-words">{subgoal}</span>
                            </div>
                            <div className="flex shrink-0 items-center gap-0.5 opacity-80 group-hover/criterion:opacity-100">
                              <Tip label={ctrl.copyCriterion(index)}>
                                <Button
                                  aria-label={ctrl.copyCriterion(index)}
                                  className="size-6 rounded text-muted-foreground/60 hover:text-foreground/90"
                                  onClick={() => void copyCriterionText(subgoal)}
                                  size="icon-xs"
                                  type="button"
                                  variant="ghost"
                                >
                                  <Codicon name="copy" size="0.7rem" />
                                </Button>
                              </Tip>
                              <Tip label={ctrl.removeCriterion(index)}>
                                <Button
                                  aria-label={ctrl.removeCriterion(index)}
                                  className="size-6 rounded text-muted-foreground/60 hover:text-destructive"
                                  disabled={isBusy}
                                  onClick={() => confirmRemoveCriterion(index)}
                                  size="icon-xs"
                                  type="button"
                                  variant="ghost"
                                >
                                  <Codicon name="close" size="0.7rem" />
                                </Button>
                              </Tip>
                            </div>
                          </div>
                        )
                      })}
                    </div>
                  ) : null}
                </div>
              </div>
            </StatusSection>
          </div>
        </ContextMenuTrigger>
        <ContextMenuContent className="w-44">{renderMenuItems(true)}</ContextMenuContent>
      </ContextMenu>

      {/* Add Criterion Dialog */}
      <Dialog
        onOpenChange={open => {
          if (!open) {
            setAddCriterionOpen(false)
            setAddCriterionError(null)
          }
        }}
        open={addCriterionOpen}
      >
        <DialogContent className="max-w-md">
          <form
            onSubmit={async e => {
              e.preventDefault()
              const text = newCriterionText.trim()

              if (!text || isBusy) {
                return
              }

              setAddCriterionError(null)
              const ok = await handleAction('subgoal.add', { text }, setAddCriterionError)

              if (ok) {
                setNewCriterionText('')
                setAddCriterionOpen(false)
              }
            }}
          >
            <DialogHeader>
              <DialogTitle>{ctrl.addCriterionDialogTitle}</DialogTitle>
              <DialogDescription>{ctrl.addCriterionPlaceholder}</DialogDescription>
            </DialogHeader>
            <div className="py-2">
              <textarea
                aria-label={ctrl.criterionLabel}
                autoFocus
                className="w-full rounded-md border border-(--ui-stroke-secondary) bg-transparent p-2 text-xs text-foreground placeholder:text-muted-foreground/60 focus:outline-none focus:ring-1 focus:ring-primary"
                disabled={isBusy}
                onChange={e => setNewCriterionText(e.target.value)}
                placeholder={ctrl.addCriterionPlaceholder}
                rows={3}
                value={newCriterionText}
              />
            </div>
            {addCriterionError && (
              <div className="text-xs text-destructive" role="alert">
                {addCriterionError}
              </div>
            )}
            <DialogFooter>
              <Button disabled={isBusy} onClick={() => setAddCriterionOpen(false)} type="button" variant="ghost">
                {t.common.cancel}
              </Button>
              <Button disabled={isBusy || !newCriterionText.trim()} type="submit">
                {ctrl.add}
              </Button>
            </DialogFooter>
          </form>
        </DialogContent>
      </Dialog>

      {/* Goal Details Dialog */}
      <Dialog onOpenChange={open => !open && setDetailsOpen(false)} open={detailsOpen}>
        <DialogContent className="max-w-lg">
          <DialogHeader>
            <DialogTitle>{ctrl.goalDetailsTitle}</DialogTitle>
          </DialogHeader>
          <div className="space-y-3 py-2 text-xs">
            <div>
              <div className="font-semibold text-muted-foreground/75 uppercase tracking-wider text-[0.68rem]">
                {ctrl.objectiveLabel}
              </div>
              <div className="mt-0.5 text-foreground leading-relaxed break-words">{goal.title}</div>
            </div>

            {/* Contract fields */}
            {goal.contract.outcome && (
              <div>
                <div className="font-semibold text-muted-foreground/75 uppercase tracking-wider text-[0.68rem]">
                  {ctrl.contractOutcome}
                </div>
                <div className="mt-0.5 text-foreground/90 leading-relaxed">{goal.contract.outcome}</div>
              </div>
            )}
            {goal.contract.verification && (
              <div>
                <div className="font-semibold text-muted-foreground/75 uppercase tracking-wider text-[0.68rem]">
                  {ctrl.contractVerification}
                </div>
                <div className="mt-0.5 text-foreground/90 leading-relaxed">{goal.contract.verification}</div>
              </div>
            )}
            {goal.contract.constraints && (
              <div>
                <div className="font-semibold text-muted-foreground/75 uppercase tracking-wider text-[0.68rem]">
                  {ctrl.contractConstraints}
                </div>
                <div className="mt-0.5 text-foreground/90 leading-relaxed">{goal.contract.constraints}</div>
              </div>
            )}
            {goal.contract.boundaries && (
              <div>
                <div className="font-semibold text-muted-foreground/75 uppercase tracking-wider text-[0.68rem]">
                  {ctrl.contractBoundaries}
                </div>
                <div className="mt-0.5 text-foreground/90 leading-relaxed">{goal.contract.boundaries}</div>
              </div>
            )}
            {goal.contract.stop_when && (
              <div>
                <div className="font-semibold text-muted-foreground/75 uppercase tracking-wider text-[0.68rem]">
                  {ctrl.contractStopWhen}
                </div>
                <div className="mt-0.5 text-foreground/90 leading-relaxed">{goal.contract.stop_when}</div>
              </div>
            )}

            {/* Wait Barrier */}
            {goal.wait_barrier && (
              <div>
                <div className="font-semibold text-muted-foreground/75 uppercase tracking-wider text-[0.68rem]">
                  {ctrl.waitBarrierTitle}
                </div>
                <div className="mt-0.5 text-foreground/90 leading-relaxed">
                  {goal.wait_barrier.type === 'until' &&
                    ctrl.waitUntil(new Date(goal.wait_barrier.until_at * 1000).toLocaleTimeString())}
                  {goal.wait_barrier.type === 'session' && ctrl.waitSession(goal.wait_barrier.target)}
                  {goal.wait_barrier.type === 'pid' && ctrl.waitPid(goal.wait_barrier.target)}
                  {goal.wait_barrier.reason && (
                    <div className="italic text-muted-foreground/80 mt-0.5">{goal.wait_barrier.reason}</div>
                  )}
                </div>
              </div>
            )}

            {/* Quality Gates */}
            {goal.gates.length > 0 && (
              <div>
                <div className="font-semibold text-muted-foreground/75 uppercase tracking-wider text-[0.68rem]">
                  {ctrl.qualityGatesTitle}
                </div>
                <div className="mt-1 space-y-1.5">
                  {goal.gates.map((gate, i) => (
                    <div
                      className="rounded border border-(--ui-stroke-tertiary)/60 bg-(--ui-control-active-background)/20 p-2"
                      key={`${i}-${gate.command}`}
                    >
                      <div className="font-mono text-[0.7rem] text-foreground/95">{gate.command}</div>
                      <div className="mt-1 flex items-center gap-2 text-[0.68rem] text-muted-foreground/80">
                        <span>{ctrl.gateAttempts(gate.attempts, gate.max_retries)}</span>
                        <span className="text-muted-foreground/40">·</span>
                        <span>{ctrl.gateTimeout(gate.timeout_seconds)}</span>
                        <span className="text-muted-foreground/40">·</span>
                        <span>{ctrl.gateLastExit(gate.last_exit_code)}</span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
          <DialogFooter>
            <Button onClick={() => setDetailsOpen(false)} type="button">
              {t.common.close}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Confirmation Dialog */}
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
