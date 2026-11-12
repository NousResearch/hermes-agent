import { useStore } from '@nanostores/react'
import { useQuery } from '@tanstack/react-query'
import { useState } from 'react'

import { Button } from '@/components/ui/button'
import { Checkbox } from '@/components/ui/checkbox'
import { DialogFooter } from '@/components/ui/dialog'
import { Field, FieldHint } from '@/components/ui/field'
import { Input } from '@/components/ui/input'
import { Select, SelectContent, SelectGroup, SelectItem, SelectLabel, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Textarea } from '@/components/ui/textarea'
import { createCronJob, type CronJob, updateCronJob } from '@/hermes'
import { useI18n } from '@/i18n'
import { requestModelOptions } from '@/lib/model-options'
import { $interfaceMode } from '@/store/interface-mode'
import { notify } from '@/store/notifications'

import { CatchUpField } from '../cron/catch-up-field'
import { cronModelChoiceValue, jobIsScriptOnly, parseCronModelChoiceValue } from '../cron/cron-job-model'
import { truncateText } from '../cron/job-state'
import { SettingsDisclosure } from '../settings/settings-card'

import {
  createPayload,
  type DraftError,
  MODEL_DEFAULT,
  type RoutineDraft,
  updatePayload,
  validateDraft
} from './routine-draft'
import { AgentSelect, DeliveryField, ScheduleField } from './routine-fields'
import { routineTitle } from './routine-model'
import type { RoutineSource } from './routine-source'
import { useRoutinesCopy } from './routines-copy'

interface RoutineEditorFormProps {
  initial: RoutineDraft
  /** Editing this routine; absent for a new one. */
  job?: CronJob
  /** The agent cannot change (editing, or an agent's own Settings › Routines). */
  lockProfile: boolean
  source: RoutineSource
  onCancel: () => void
  onDone: (job: CronJob) => void
}

/** R4: the one routine editor, for a blank routine and for every edit. Agent, name, instruction, when
 *  to run, where results go, and the power settings under a collapsed Advanced. */
export function RoutineEditorForm({ initial, job, lockProfile, onCancel, onDone, source }: RoutineEditorFormProps) {
  const { t } = useI18n()
  const c = t.cron
  const copy = useRoutinesCopy()
  const developer = useStore($interfaceMode) === 'advanced'
  const [draft, setDraft] = useState(initial)
  const [error, setError] = useState<null | string>(null)
  const [saving, setSaving] = useState(false)
  const scriptOnly = job ? jobIsScriptOnly(job) : false
  const set = (patch: Partial<RoutineDraft>) => setDraft(current => ({ ...current, ...patch }))

  const DRAFT_ERRORS: Record<DraftError, string> = { prompt: copy.instructionRequired, schedule: copy.scheduleRequired }

  async function submit(event: React.FormEvent) {
    event.preventDefault()
    const invalid = validateDraft(draft, job)

    if (invalid) {
      setError(DRAFT_ERRORS[invalid])

      return
    }

    setSaving(true)
    setError(null)

    try {
      const saved = await source.mutate(() =>
        job
          ? updateCronJob(job.id, updatePayload(draft, job), source.scopeFor(job))
          : createCronJob(createPayload(draft), source.createScope(draft.profile))
      )

      if (saved) {
        notify({ kind: 'success', title: job ? c.updated : c.created, message: truncateText(routineTitle(saved), 60) })
        onDone(saved)
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : c.failedSave)
    } finally {
      setSaving(false)
    }
  }

  return (
    <form className="routines-form" onSubmit={submit}>
      {!lockProfile && (
        <Field htmlFor="routine-agent" label={copy.agent}>
          <AgentSelect id="routine-agent" onChange={profile => set({ profile })} value={draft.profile} />
        </Field>
      )}
      <Field htmlFor="routine-name" label={copy.name}>
        <Input
          id="routine-name"
          onChange={event => set({ name: event.target.value })}
          placeholder={copy.namePlaceholder}
          value={draft.name}
        />
      </Field>
      {scriptOnly && !draft.prompt ? (
        <Field label={copy.instruction}>
          <FieldHint>{copy.scriptRoutine}</FieldHint>
        </Field>
      ) : (
        <Field htmlFor="routine-instruction" label={copy.instruction}>
          <Textarea
            className="routines-instruction-input"
            id="routine-instruction"
            onChange={event => set({ prompt: event.target.value })}
            placeholder={copy.instructionPlaceholder}
            value={draft.prompt}
          />
        </Field>
      )}
      <ScheduleField onChange={pick => set({ pick })} pick={draft.pick} />
      <DeliveryField
        onChange={deliver => set({ deliver })}
        profile={draft.profile}
        source={source}
        value={draft.deliver}
      />
      <SettingsDisclosure description={copy.advancedSummary} title={copy.advanced}>
        <div className="routines-advanced">
          {!scriptOnly && <ModelField onChange={model => set({ model })} value={draft.model} />}
          <CatchUpField c={c} onChange={catchUp => set({ catchUp })} value={draft.catchUp} />
          <Field htmlFor="routine-run-limit" label={copy.runLimit}>
            <span className="routines-when-inline">
              <span>{copy.stopAfter}</span>
              <Input
                className="routines-when-number"
                id="routine-run-limit"
                inputMode="numeric"
                min={1}
                onChange={event => set({ runLimit: event.target.value.replace(/\D/g, '').slice(0, 4) })}
                placeholder="∞"
                type="number"
                value={draft.runLimit}
              />
              <span>{copy.runsUnit}</span>
            </span>
          </Field>
          <label className="routines-check">
            <Checkbox checked={draft.remember} onCheckedChange={next => set({ remember: next === true })} />
            <span className="routines-check-text">
              <span>{copy.rememberLastRun}</span>
              <span className="routines-field-note">{copy.rememberHint}</span>
            </span>
          </label>
          {developer && job && (
            <p className="routines-field-note">
              {copy.routineId}: <code>{job.id}</code>
            </p>
          )}
        </div>
      </SettingsDisclosure>
      {error && (
        <p className="routines-form-error" role="alert">
          {error}
        </p>
      )}
      <DialogFooter>
        <Button disabled={saving} onClick={onCancel} type="button" variant="outline">
          {t.common.cancel}
        </Button>
        <Button loading={saving} type="submit">
          {job ? copy.saveChanges : c.createAction}
        </Button>
      </DialogFooter>
    </form>
  )
}

/** Advanced › Model: the configured providers' available models, as the chat picker offers them. */
function ModelField({ onChange, value }: { onChange: (value: string) => void; value: string }) {
  const copy = useRoutinesCopy()

  const options = useQuery({
    queryKey: ['model-options', 'global'],
    queryFn: () => requestModelOptions({})
  })

  const providers = (options.data?.providers ?? []).filter(
    provider => provider.authenticated !== false && (provider.models ?? []).length > 0
  )

  // A pinned model that has left the catalog stays visible and re-selectable rather than blank.
  const known =
    value === MODEL_DEFAULT ||
    providers.some(provider => (provider.models ?? []).some(model => cronModelChoiceValue(provider.slug, model) === value))

  return (
    <Field htmlFor="routine-model" label={copy.model}>
      <Select onValueChange={onChange} value={value}>
        <SelectTrigger id="routine-model">
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          <SelectItem value={MODEL_DEFAULT}>{copy.modelDefault}</SelectItem>
          {!known && <SelectItem value={value}>{parseCronModelChoiceValue(value)?.model ?? value}</SelectItem>}
          {providers.map(provider => (
            <SelectGroup key={provider.slug}>
              <SelectLabel>{provider.name}</SelectLabel>
              {(provider.models ?? []).map(model => {
                const choice = cronModelChoiceValue(provider.slug, model)

                return (
                  <SelectItem key={choice} value={choice}>
                    {model}
                  </SelectItem>
                )
              })}
            </SelectGroup>
          ))}
        </SelectContent>
      </Select>
    </Field>
  )
}
