import { useStore } from '@nanostores/react'
import { useCallback, useEffect, useState } from 'react'

import { Dialog, DialogContent } from '@/components/ui/dialog'
import { SegmentedControl } from '@/components/ui/segmented-control'
import { useI18n } from '@/i18n'
import {
  $automationComposer,
  closeAutomationComposer,
  invalidateOnSessionSwitch,
  setAutomationComposerType,
  submitAutomation
} from '@/store/automation-composer'
import { $activeSessionId } from '@/store/session'
import { $sessionControlBySession, refreshSessionControl, runSessionControlAction, type SessionControlAction } from '@/store/session-control'

import {
  automationLabels,
  buildArgs,
  deliverKickoff,
  existingAutomation,
  isUnavailable,
  loopIntervalTooShort,
  pauseBlocked,
  submitBlocked,
  useAutomationForm,
  type AutomationCopy,
  type OnSubmitText
} from './automation-composer-form'
import {
  AutomationConsequence,
  AutomationFields,
  AutomationFooter,
  AutomationHeader,
  AutomationNotices
} from './automation-composer-parts'

type SegmentedValue = 'goal' | 'loop' | 'heartbeat'

interface AutomationComposerDialogProps {
  /** The conversation's display title captured when the dialog opened. */
  conversationTitle?: string | null
  /** Routes to the existing Cron scheduler UI. */
  onOpenCron?: () => void
  /** Submits a goal kickoff message to the captured session (mirrors the
   *  session-control goal card's send dispatch). */
  onSubmitText: OnSubmitText
}

const TYPE_OPTIONS = [
  { id: 'goal', labelKey: 'goalLabel' },
  { id: 'loop', labelKey: 'loopLabel' },
  { id: 'heartbeat', labelKey: 'heartbeatLabel' }
] as const satisfies readonly { id: SegmentedValue; labelKey: keyof AutomationCopy }[]

export function AutomationComposerDialog({
  conversationTitle = null,
  onSubmitText,
  onOpenCron
}: AutomationComposerDialogProps) {
  const { t } = useI18n()
  const copy = t.automationComposer

  const state = useStore($automationComposer)
  const { open, submitting, sessionId, type } = state
  const isEdit = state.mode === 'edit'
  const controls = useStore($sessionControlBySession)
  const entry = sessionId ? controls[sessionId] : undefined
  const existing = existingAutomation(entry, type, isEdit)
  const unavailable = isUnavailable(entry)

  const form = useAutomationForm()
  const { replace, values } = form
  const [pausing, setPausing] = useState(false)
  const [prefilledEditKey, setPrefilledEditKey] = useState<string | null>(null)

  const loopMinInterval = entry?.snapshot?.loop_min_interval_seconds
  const intervalTooShort = loopIntervalTooShort(type, values.interval, loopMinInterval)
  const labels = automationLabels(copy, type, isEdit)

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

  // Fill the form from the live snapshot once per edit target: a later snapshot
  // refresh must never overwrite what the user has typed.
  useEffect(() => {
    if (!open) {
      if (prefilledEditKey !== null) {
        setPrefilledEditKey(null)
      }

      return
    }

    if (!isEdit || !entry?.snapshot || !sessionId) {
      return
    }

    const editKey = `${sessionId}:${type}`

    if (prefilledEditKey === editKey) {
      return
    }

    const snap = entry.snapshot[type]

    if (!snap) {
      return
    }

    if ('title' in snap) {
      form.replace({ prompt: snap.title, criteria: [...snap.subgoals], maxTurns: String(snap.max_turns) })
    } else {
      form.replace({
        prompt: snap.prompt,
        interval: String(snap.interval_seconds),
        ...('times' in snap ? { runLimit: snap.times ? String(snap.times) : '', stopCondition: snap.until || '' } : {})
      })
    }

    setPrefilledEditKey(editKey)
  }, [open, isEdit, sessionId, type, entry?.snapshot, prefilledEditKey, replace])

  const toggleType = useCallback((next: SegmentedValue) => {
    setAutomationComposerType(next)
  }, [])

  const handlePause = useCallback(async () => {
    if (!sessionId || pauseBlocked({ pausing, sessionId, submitting, unavailable })) {
      return
    }

    const pauseAction = `${type}.pause` as SessionControlAction

    setPausing(true)

    try {
      await runSessionControlAction(sessionId, pauseAction)
      void refreshSessionControl(sessionId)
    } catch {
      // The error is surfaced via the session-control store; the dialog stays open.
    } finally {
      setPausing(false)
    }
  }, [sessionId, pausing, submitting, unavailable, type])

  const blocked = submitBlocked({
    existing,
    intervalTooShort,
    isEdit,
    pausing,
    prompt: values.prompt,
    sessionId,
    submitting,
    unavailable
  })

  const handleSubmit = async () => {
    if (blocked || !sessionId) {
      return
    }

    try {
      await submitAutomation(type, buildArgs(type, isEdit, values), dispatch =>
        deliverKickoff(dispatch, { copy, onSubmitText, sessionId })
      )
    } catch {
      // The store keeps the dialog open with the error surfaced.
    }
  }

  const openCron = () => {
    closeAutomationComposer()
    onOpenCron?.()
  }

  return (
    <Dialog onOpenChange={closeAutomationComposer} open={open}>
      <DialogContent bodyClassName="gap-5" className="max-w-lg">
        <AutomationHeader conversationTitle={conversationTitle} copy={copy} isEdit={isEdit} />
        <form
          className="grid gap-4"
          onSubmit={e => {
            e.preventDefault()
            void handleSubmit()
          }}
        >
          <SegmentedControl
            disabled={submitting || isEdit}
            onChange={toggleType}
            options={TYPE_OPTIONS.map(option => ({
              id: option.id,
              label: copy[option.labelKey]
            }))}
            value={type}
          />

          <AutomationFields
            copy={copy}
            form={form}
            intervalTooShort={intervalTooShort}
            loopMinInterval={loopMinInterval}
            type={type}
          />

          {!isEdit && <AutomationConsequence labels={labels} showIdle={type !== 'goal'} />}

          <AutomationNotices
            copy={copy}
            entry={entry}
            existing={existing}
            isEdit={isEdit}
            onManageExisting={closeAutomationComposer}
            onOpenCron={onOpenCron ? openCron : undefined}
            sessionId={sessionId}
            stateError={state.error}
            submitting={submitting}
            unavailable={unavailable}
          />

          <AutomationFooter
            cancelLabel={t.common.cancel}
            isEdit={isEdit}
            labels={labels}
            onCancel={closeAutomationComposer}
            onPause={() => void handlePause()}
            pauseDisabled={pauseBlocked({ pausing, sessionId, submitting, unavailable })}
            submitDisabled={blocked}
          />
        </form>
      </DialogContent>
    </Dialog>
  )
}
