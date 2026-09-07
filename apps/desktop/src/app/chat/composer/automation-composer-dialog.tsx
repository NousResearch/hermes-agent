import { useStore } from '@nanostores/react'
import { useCallback, useEffect, useState } from 'react'

import { queueKickoffIfSessionBusy } from '@/app/session/hooks/use-prompt-actions/queue-if-busy'
import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle
} from '@/components/ui/dialog'
import { Field, FieldHint } from '@/components/ui/field'
import { Input } from '@/components/ui/input'
import { SegmentedControl } from '@/components/ui/segmented-control'
import { Textarea } from '@/components/ui/textarea'
import { useI18n } from '@/i18n'
import {
  $automationComposer,
  type AutomationType,
  closeAutomationComposer,
  invalidateOnSessionSwitch,
  setAutomationComposerType,
  submitAutomation
} from '@/store/automation-composer'
import { $activeSessionId } from '@/store/session'
import { $sessionControlBySession, refreshSessionControl } from '@/store/session-control'

type SegmentedValue = 'goal' | 'loop' | 'heartbeat'

interface AutomationComposerDialogProps {
  /** Submits a goal kickoff message to the captured session (mirrors the
   *  session-control goal card's send dispatch). */
  onSubmitText: (
    value: string,
    options?: { displayKind?: 'hidden'; sessionId?: string | null }
  ) => Promise<boolean> | boolean
  /** Routes to the existing Cron scheduler UI. */
  onOpenCron?: () => void
  /** The current conversation's display title, when known. */
  conversationTitle?: string | null
}

const TYPE_OPTIONS = [
  { id: 'goal', labelKey: 'goalLabel' },
  { id: 'loop', labelKey: 'loopLabel' },
  { id: 'heartbeat', labelKey: 'heartbeatLabel' }
] as const

export function AutomationComposerDialog({
  conversationTitle = null,
  onSubmitText,
  onOpenCron
}: AutomationComposerDialogProps) {
  const { t } = useI18n()
  const copy = t.automationComposer

  const state = useStore($automationComposer)
  const open = state.open
  const submitting = state.submitting
  const sessionId = state.sessionId
  const type = state.type
  const controls = useStore($sessionControlBySession)
  const entry = sessionId ? controls[sessionId] : undefined
  const existing = entry?.snapshot?.[type]
  const unavailable = !!entry && (entry.capability === 'unsupported' || !!entry.error || entry.loading)
  useEffect(() => {
    if (open && sessionId) {void refreshSessionControl(sessionId)}
  }, [open, sessionId])

  // Cancel/invalidate on session switch: a completion landing in a chat the
  // user moved away from must never happen.
  useEffect(() => {
    if (!open) {
      return
    }

    return $activeSessionId.listen(() => invalidateOnSessionSwitch())
  }, [open])

  const [prompt, setPrompt] = useState('')
  const [criteria, setCriteria] = useState<string[]>([])
  const [draftCriterion, setDraftCriterion] = useState('')
  const [maxTurns, setMaxTurns] = useState('')
  const [interval, setInterval] = useState('')
  const [runLimit, setRunLimit] = useState('')
  const [stopCondition, setStopCondition] = useState('')

  const toggleType = useCallback(
    (next: SegmentedValue) => {
      setAutomationComposerType(next)
    },
    []
  )

  const handleAddCriterion = () => {
    const value = draftCriterion.trim()

    if (!value) {
      return
    }

    setCriteria(current => [...current, value])
    setDraftCriterion('')
  }

  const handleSubmit = async () => {
    if (submitting || !sessionId || existing || unavailable) {
      return
    }

    const trimmedPrompt = prompt.trim()

    if (!trimmedPrompt) {
      return
    }

    try {
      await submitAutomation(type, buildArgs(type, {
        prompt: trimmedPrompt,
        criteria,
        maxTurns,
        interval,
        runLimit,
        stopCondition
      }), async dispatch => {

      // A goal's create dispatch is a `send`: the backend already persisted the
      // goal and handed us the continuation prompt. Mirror the goal card's
      // resume path — queue it if the session is busy, otherwise submit it.
      if (dispatch?.type === 'send' && dispatch.message) {
        const queued = queueKickoffIfSessionBusy({
          displayText: dispatch.display ?? undefined,
          sessionId,
          text: dispatch.message
        })

        if (queued === 'busy' || (queued === 'idle' && !await onSubmitText(dispatch.message, { displayKind: 'hidden', sessionId }))) {
          throw new Error(copy.kickoffFailure)
        }
      }
      })
    } catch {
      // The store keeps the dialog open with the error surfaced.
    }
  }

  const submitLabel =
    type === 'goal' ? copy.startGoal : type === 'loop' ? copy.startLoop : copy.createHeartbeat

  const firstRunText =
    type === 'goal' ? copy.firstRunGoal : type === 'loop' ? copy.firstRunLoop : copy.firstRunHeartbeat

  const idleText = type === 'loop' ? copy.idleLoop : copy.idleHeartbeat

  return (
    <Dialog onOpenChange={closeAutomationComposer} open={open}>
      <DialogContent bodyClassName="gap-5" className="max-w-lg">
        <DialogHeader>
          <DialogTitle icon={CodiconAutomation}>{copy.title}</DialogTitle>
          <DialogDescription>
            {conversationTitle ? copy.sessionScope(conversationTitle) : copy.title}
          </DialogDescription>
        </DialogHeader>
        <form
          className="grid gap-4"
          onSubmit={e => {
            e.preventDefault()
            void handleSubmit()
          }}
        >
          <SegmentedControl
            disabled={submitting}
            onChange={toggleType}
            options={TYPE_OPTIONS.map(option => ({
              id: option.id,
              label: copy[option.labelKey]
            }))}
            value={type}
          />

          {type === 'goal' && (
            <>
              <Field htmlFor="automation-goal-prompt" label={copy.goalPromptLabel}>
                <Textarea
                  id="automation-goal-prompt"
                  onChange={e => setPrompt(e.target.value)}
                  placeholder={copy.goalPromptPlaceholder}
                  value={prompt}
                />
              </Field>

              <div className="grid gap-1.5">
                <div className="flex items-baseline gap-2">
                  <label className="text-xs font-medium text-foreground" htmlFor="automation-criterion">
                    {copy.goalCriteriaLabel}
                  </label>
                </div>
                <div className="flex items-center gap-2">
                  <Input
                    id="automation-criterion"
                    onChange={e => setDraftCriterion(e.target.value)}
                    onKeyDown={e => {
                      if (e.key === 'Enter') {
                        e.preventDefault()
                        handleAddCriterion()
                      }
                    }}
                    placeholder={copy.goalCriteriaPlaceholder}
                    value={draftCriterion}
                  />
                  <Button onClick={handleAddCriterion} size="sm" type="button" variant="secondary">
                    <Codicon name="add" size="0.875rem" />
                    {copy.addCriterion}
                  </Button>
                </div>
                {criteria.length > 0 && (
                  <ul className="grid gap-1">
                    {criteria.map((criterion, index) => (
                      <li
                        className="flex items-center justify-between gap-2 rounded-md bg-(--ui-bg-tertiary) px-2 py-1.5 text-sm"
                        key={`${criterion}-${index}`}
                      >
                        <span className="min-w-0 truncate">{criterion}</span>
                        <button
                          aria-label={copy.removeCriterion(index + 1)}
                          className="shrink-0 text-muted-foreground hover:text-destructive"
                          onClick={() => setCriteria(current => current.filter((_, i) => i !== index))}
                          type="button"
                        >
                          <Codicon name="close" size="0.875rem" />
                        </button>
                      </li>
                    ))}
                  </ul>
                )}
              </div>

              <details><summary className="cursor-pointer text-sm font-medium">{copy.advanced}</summary>
              <Field htmlFor="automation-max-turns" label={copy.maxTurnsLabel} optional optionalLabel={copy.maxTurnsOptional}>
                <Input
                  id="automation-max-turns"
                  max="1000"
                  min="1"
                  onChange={e => setMaxTurns(e.target.value)}
                  type="number"
                  value={maxTurns}
                />
              </Field>
              </details>
            </>
          )}

          {type === 'loop' && (
            <>
              <Field htmlFor="automation-loop-prompt" label={copy.loopPromptLabel}>
                <Textarea
                  id="automation-loop-prompt"
                  onChange={e => setPrompt(e.target.value)}
                  placeholder={copy.loopPromptPlaceholder}
                  value={prompt}
                />
              </Field>

              <Field htmlFor="automation-loop-interval" label={copy.intervalLabel}>
                <Input
                  id="automation-loop-interval"
                  max="604800"
                  min="1"
                  onChange={e => setInterval(e.target.value)}
                  required
                  type="number"
                  value={interval}
                />
                <FieldHint>{copy.intervalSeconds}</FieldHint>
              </Field>

              <Field
                htmlFor="automation-run-limit"
                label={copy.runLimitLabel}
                optional
                optionalLabel={copy.runLimitOptional}
              >
                <Input
                  id="automation-run-limit"
                  max="10000"
                  min="1"
                  onChange={e => setRunLimit(e.target.value)}
                  type="number"
                  value={runLimit}
                />
              </Field>

              <Field
                htmlFor="automation-stop-condition"
                label={copy.stopConditionLabel}
                optional
                optionalLabel={copy.stopConditionOptional}
              >
                <Input
                  id="automation-stop-condition"
                  onChange={e => setStopCondition(e.target.value)}
                  placeholder={copy.stopConditionPlaceholder}
                  value={stopCondition}
                />
              </Field>
            </>
          )}

          {type === 'heartbeat' && (
            <>
              <Field htmlFor="automation-heartbeat-prompt" label={copy.heartbeatPromptLabel}>
                <Textarea
                  id="automation-heartbeat-prompt"
                  onChange={e => setPrompt(e.target.value)}
                  placeholder={copy.heartbeatPromptPlaceholder}
                  value={prompt}
                />
              </Field>

              <Field htmlFor="automation-heartbeat-interval" label={copy.intervalLabel}>
                <Input
                  id="automation-heartbeat-interval"
                  max="604800"
                  min="60"
                  onChange={e => setInterval(e.target.value)}
                  required
                  type="number"
                  value={interval}
                />
                <FieldHint>{copy.intervalSeconds}</FieldHint>
              </Field>
            </>
          )}

          <div className="grid gap-1 text-xs text-muted-foreground">
            <p>{firstRunText}</p>
            {type !== 'goal' && <p>{idleText}</p>}
          </div>

          {!sessionId && <FieldHint error>{copy.noSession}</FieldHint>}
          {unavailable && <FieldHint error>{copy.unavailable}</FieldHint>}
          {existing && <div className="grid gap-2 rounded-md border p-3 text-sm">
            <p>{'title' in existing ? existing.title : existing.prompt}</p>
            <p>{copy.duplicateError}</p>
            <Button onClick={closeAutomationComposer} type="button" variant="secondary">{copy.manageExisting}</Button>
          </div>}
          {state.error && <FieldHint error>{state.error}</FieldHint>}

          <div className="grid gap-1.5">
            <p className="text-[0.66rem] leading-4 text-muted-foreground">{copy.cronExplain}</p>
            {onOpenCron && (
              <button
                className="inline-flex items-center gap-1 text-[0.66rem] font-medium text-primary hover:underline"
                disabled={submitting}
                onClick={() => { closeAutomationComposer(); onOpenCron() }}
                type="button"
              >
                <Codicon name="calendar" size="0.75rem" />
                {copy.cronLink}
              </button>
            )}
          </div>

          <DialogFooter>
            <Button onClick={closeAutomationComposer} type="button" variant="ghost">
              {t.common.cancel}
            </Button>
            <Button disabled={submitting || !sessionId || !!existing || unavailable || !prompt.trim()} type="submit">
              {submitLabel}
            </Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  )
}

function buildArgs(
  type: AutomationType,
  values: {
    prompt: string
    criteria: string[]
    maxTurns: string
    interval: string
    runLimit: string
    stopCondition: string
  }
) {
  if (type === 'goal') {
    return {
      prompt: values.prompt,
      criteria: values.criteria,
      ...(values.maxTurns ? { max_turns: Number(values.maxTurns) } : {})
    }
  }

  if (type === 'loop') {
    return {
      prompt: values.prompt,
      interval_seconds: Number(values.interval),
      ...(values.runLimit ? { run_limit: Number(values.runLimit) } : {}),
      ...(values.stopCondition.trim() ? { stop_condition: values.stopCondition.trim() } : {})
    }
  }

  return {
    prompt: values.prompt,
    interval_seconds: Number(values.interval)
  }
}

function CodiconAutomation() {
  return <Codicon name="sparkle" size="1rem" />
}