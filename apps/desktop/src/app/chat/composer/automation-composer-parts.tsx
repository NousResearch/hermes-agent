import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import {
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle
} from '@/components/ui/dialog'
import { Field, FieldHint } from '@/components/ui/field'
import { Input } from '@/components/ui/input'
import { Textarea } from '@/components/ui/textarea'
import type { AutomationType } from '@/store/automation-composer'
import type {
  SessionControlEntry,
  SessionControlGoal,
  SessionControlHeartbeat,
  SessionControlLoop
} from '@/store/session-control'

import type { AutomationCopy, AutomationFormApi, AutomationLabels } from './automation-composer-form'

/** The per-type field groups, split out of the dialog so each one stays a readable unit. */
function GoalFields({ copy, form }: { copy: AutomationCopy; form: AutomationFormApi }) {
  const { values } = form

  return (
    <>
      <Field htmlFor="automation-goal-prompt" label={copy.goalPromptLabel}>
        <Textarea
          id="automation-goal-prompt"
          onChange={e => form.set('prompt', e.target.value)}
          placeholder={copy.goalPromptPlaceholder}
          value={values.prompt}
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
            onChange={e => form.set('draftCriterion', e.target.value)}
            onKeyDown={e => {
              if (e.key === 'Enter') {
                e.preventDefault()
                form.addCriterion()
              }
            }}
            placeholder={copy.goalCriteriaPlaceholder}
            value={values.draftCriterion}
          />
          <Button onClick={form.addCriterion} size="sm" type="button" variant="secondary">
            <Codicon name="add" size="0.875rem" />
            {copy.addCriterion}
          </Button>
        </div>
        {values.criteria.length > 0 && (
          <ul className="grid gap-1">
            {values.criteria.map((criterion, index) => (
              <li
                className="flex items-center justify-between gap-2 rounded-md bg-(--ui-bg-tertiary) px-2 py-1.5 text-sm"
                key={`${criterion}-${index}`}
              >
                <span className="min-w-0 truncate">{criterion}</span>
                <button
                  aria-label={copy.removeCriterion(index + 1)}
                  className="shrink-0 text-muted-foreground hover:text-destructive"
                  onClick={() => form.removeCriterion(index)}
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
          onChange={e => form.set('maxTurns', e.target.value)}
          type="number"
          value={values.maxTurns}
        />
      </Field>
      </details>
    </>
  )
}

function LoopFields({
  copy,
  form,
  loopMinInterval,
  intervalTooShort
}: {
  copy: AutomationCopy
  form: AutomationFormApi
  intervalTooShort: boolean
  loopMinInterval: number | undefined
}) {
  const { values } = form

  return (
    <>
      <Field htmlFor="automation-loop-prompt" label={copy.loopPromptLabel}>
        <Textarea
          id="automation-loop-prompt"
          onChange={e => form.set('prompt', e.target.value)}
          placeholder={copy.loopPromptPlaceholder}
          value={values.prompt}
        />
      </Field>

      <Field htmlFor="automation-loop-interval" label={copy.intervalLabel}>
        <Input
          id="automation-loop-interval"
          max="604800"
          min={loopMinInterval ?? 1}
          onChange={e => form.set('interval', e.target.value)}
          required
          type="number"
          value={values.interval}
        />
        <FieldHint>{copy.intervalSeconds}</FieldHint>
        {intervalTooShort && loopMinInterval !== undefined && (
          <FieldHint error>{copy.loopMinIntervalError(loopMinInterval)}</FieldHint>
        )}
      </Field>

      <Field htmlFor="automation-run-limit" label={copy.runLimitLabel} optional optionalLabel={copy.runLimitOptional}>
        <Input
          id="automation-run-limit"
          max="10000"
          min="1"
          onChange={e => form.set('runLimit', e.target.value)}
          type="number"
          value={values.runLimit}
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
          onChange={e => form.set('stopCondition', e.target.value)}
          placeholder={copy.stopConditionPlaceholder}
          value={values.stopCondition}
        />
      </Field>
    </>
  )
}

function HeartbeatFields({ copy, form }: { copy: AutomationCopy; form: AutomationFormApi }) {
  const { values } = form

  return (
    <>
      <Field htmlFor="automation-heartbeat-prompt" label={copy.heartbeatPromptLabel}>
        <Textarea
          id="automation-heartbeat-prompt"
          onChange={e => form.set('prompt', e.target.value)}
          placeholder={copy.heartbeatPromptPlaceholder}
          value={values.prompt}
        />
      </Field>

      <Field htmlFor="automation-heartbeat-interval" label={copy.intervalLabel}>
        <Input
          id="automation-heartbeat-interval"
          max="604800"
          min="60"
          onChange={e => form.set('interval', e.target.value)}
          required
          type="number"
          value={values.interval}
        />
        <FieldHint>{copy.intervalSeconds}</FieldHint>
      </Field>
    </>
  )
}

interface AutomationFieldsProps {
  copy: AutomationCopy
  form: AutomationFormApi
  intervalTooShort: boolean
  loopMinInterval: number | undefined
  type: AutomationType
}

export function AutomationFields({ copy, form, intervalTooShort, loopMinInterval, type }: AutomationFieldsProps) {
  switch (type) {
    case 'loop':
      return (
        <LoopFields
          copy={copy}
          form={form}
          intervalTooShort={intervalTooShort}
          loopMinInterval={loopMinInterval}
        />
      )

    case 'heartbeat':
      return <HeartbeatFields copy={copy} form={form} />

    default:
      return <GoalFields copy={copy} form={form} />
  }
}

interface AutomationHeaderProps {
  conversationTitle: string | null
  copy: AutomationCopy
  isEdit: boolean
}

function CodiconAutomation() {
  return <Codicon name="sparkle" size="1rem" />
}

export function AutomationHeader({ conversationTitle, copy, isEdit }: AutomationHeaderProps) {
  const title = isEdit ? copy.editTitle : copy.title

  return (
    <DialogHeader>
      <DialogTitle icon={CodiconAutomation}>{title}</DialogTitle>
      <DialogDescription>
        {conversationTitle ? copy.sessionScope(conversationTitle) : title}
      </DialogDescription>
    </DialogHeader>
  )
}

interface AutomationNoticesProps {
  copy: AutomationCopy
  entry: SessionControlEntry | undefined
  /** The automation already on this session for the selected type, in create mode only. */
  existing: SessionControlGoal | SessionControlHeartbeat | SessionControlLoop | null | undefined
  isEdit: boolean
  onManageExisting: () => void
  onOpenCron: (() => void) | undefined
  sessionId: string | null
  stateError: string | null
  submitting: boolean
  unavailable: boolean
}

export function AutomationNotices({
  copy,
  entry,
  existing,
  isEdit,
  onManageExisting,
  onOpenCron,
  sessionId,
  stateError,
  submitting,
  unavailable
}: AutomationNoticesProps) {
  return (
    <>
      {!sessionId && <FieldHint error>{copy.noSession}</FieldHint>}
      {unavailable && <FieldHint error>{copy.unavailable}</FieldHint>}
      {existing && (
        <div className="grid gap-2 rounded-md border p-3 text-sm">
          <p>{'title' in existing ? existing.title : existing.prompt}</p>
          <p>{copy.duplicateError}</p>
          <Button onClick={onManageExisting} type="button" variant="secondary">
            {copy.manageExisting}
          </Button>
        </div>
      )}
      {stateError && <FieldHint error>{stateError}</FieldHint>}
      {entry?.actionError && <FieldHint error>{entry.actionError}</FieldHint>}

      {!isEdit && (
        <div className="grid gap-1.5">
          <p className="text-[0.66rem] leading-4 text-muted-foreground">{copy.cronExplain}</p>
          {onOpenCron && (
            <button
              className="inline-flex items-center gap-1 text-[0.66rem] font-medium text-primary hover:underline"
              disabled={submitting}
              onClick={onOpenCron}
              type="button"
            >
              <Codicon name="calendar" size="0.75rem" />
              {copy.cronLink}
            </button>
          )}
        </div>
      )}
    </>
  )
}

interface AutomationFooterProps {
  cancelLabel: string
  isEdit: boolean
  labels: AutomationLabels
  onCancel: () => void
  onPause: () => void
  pauseDisabled: boolean
  submitDisabled: boolean
}

export function AutomationFooter({
  cancelLabel,
  isEdit,
  labels,
  onCancel,
  onPause,
  pauseDisabled,
  submitDisabled
}: AutomationFooterProps) {
  return (
    <DialogFooter>
      <Button onClick={onCancel} type="button" variant="ghost">
        {cancelLabel}
      </Button>
      {isEdit && (
        <Button
          aria-label={labels.pause}
          disabled={pauseDisabled}
          onClick={onPause}
          type="button"
          variant="secondary"
        >
          {labels.pause}
        </Button>
      )}
      <Button disabled={submitDisabled} type="submit">
        {labels.submit}
      </Button>
    </DialogFooter>
  )
}

/** The consequence of creating an automation here: a goal starts work now, a loop or heartbeat
 *  only fires while this conversation is live. */
export function AutomationConsequence({ labels, showIdle }: { labels: AutomationLabels; showIdle: boolean }) {
  return (
    <div className="grid gap-1 text-xs text-muted-foreground">
      <p>{labels.firstRun}</p>
      {showIdle && labels.idle && <p>{labels.idle}</p>}
    </div>
  )
}
