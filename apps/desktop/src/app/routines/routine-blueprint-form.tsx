import { useState } from 'react'

import { Button } from '@/components/ui/button'
import { DialogFooter } from '@/components/ui/dialog'
import { Field, FieldHint } from '@/components/ui/field'
import { type AutomationBlueprint, type CronJob, instantiateAutomationBlueprint } from '@/hermes'
import { useI18n } from '@/i18n'
import { notify } from '@/store/notifications'

import {
  BlueprintSlotControl,
  blueprintSlotHelp,
  cleanBlueprintFieldError,
  initialBlueprintValues
} from '../cron/blueprints'

import { AgentSelect, DeliveryField } from './routine-fields'
import { formatClock } from './routine-schedule'
import type { RoutineSource } from './routine-source'
import { useRoutinesCopy } from './routines-copy'

interface RoutineBlueprintFormProps {
  blueprint: AutomationBlueprint
  profile: string
  lockProfile: boolean
  source: RoutineSource
  onProfileChange: (profile: string) => void
  onCancel: () => void
  onDone: (job: CronJob) => void
}

/** R4 from a ready-made routine: the agent, then the blueprint's own fields (its time, days, topic…),
 *  filled by the backend into a real routine (`blueprints/instantiate`, the same create path). The
 *  delivery slot is the editor's Send results to. Everything else is editable once it exists. */
export function RoutineBlueprintForm({
  blueprint,
  lockProfile,
  onCancel,
  onDone,
  onProfileChange,
  profile,
  source
}: RoutineBlueprintFormProps) {
  const { locale, t } = useI18n()
  const copy = useRoutinesCopy()
  const [values, setValues] = useState<Record<string, string>>(() => initialBlueprintValues(blueprint))
  const [error, setError] = useState<null | string>(null)
  const [saving, setSaving] = useState(false)
  const set = (name: string, value: string) => setValues(current => ({ ...current, [name]: value }))

  async function submit(event: React.FormEvent) {
    event.preventDefault()
    setSaving(true)
    setError(null)

    try {
      const job = await source.mutate(() =>
        instantiateAutomationBlueprint({ blueprint: blueprint.key, values }, profile, source.connectionId)
      )

      if (job) {
        notify({ kind: 'success', title: t.cron.created, message: blueprint.title })
        onDone(job)
      }
    } catch (err) {
      // A 422 carries the slot's own message; show it inline.
      setError(cleanBlueprintFieldError(err instanceof Error ? err.message : String(err)))
    } finally {
      setSaving(false)
    }
  }

  return (
    <form className="routines-form" onSubmit={submit}>
      {!lockProfile && (
        <Field htmlFor="routine-agent" label={copy.agent}>
          <AgentSelect id="routine-agent" onChange={onProfileChange} value={profile} />
        </Field>
      )}
      {blueprint.fields.map(field => {
        const id = `routine-slot-${field.name}`

        // The backend's time hint ("24h local time, e.g. 08:00") contradicts the time input, which
        // shows the locale's clock: say the same thing in the field's own format.
        const help =
          field.type === 'time'
            ? copy.localTimeHint(formatClock(field.default || '09:00', locale))
            : blueprintSlotHelp(field)

        return field.name === 'deliver' ? (
          <DeliveryField
            key={field.name}
            onChange={next => set(field.name, next)}
            profile={profile}
            source={source}
            value={values[field.name] ?? 'local'}
          />
        ) : (
          <Field htmlFor={id} key={field.name} label={field.label}>
            <BlueprintSlotControl
              field={field}
              id={id}
              onChange={next => set(field.name, next)}
              value={values[field.name] ?? ''}
            />
            {help && <FieldHint>{help}</FieldHint>}
          </Field>
        )
      })}
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
          {t.cron.createAction}
        </Button>
      </DialogFooter>
    </form>
  )
}
