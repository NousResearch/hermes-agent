import { useQuery } from '@tanstack/react-query'
import { useCallback, useMemo, useState } from 'react'

import { PageLoader } from '@/components/page-loader'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue
} from '@/components/ui/select'
import { getAutomationBlueprints, instantiateAutomationBlueprint } from '@/hermes'
import type { AutomationBlueprint, AutomationBlueprintField, CronJob } from '@/hermes'
import { type Translations, useI18n } from '@/i18n'
import { asText } from '@/lib/text'
import { updateCronJobs } from '@/store/cron'
import { notify } from '@/store/notifications'

import { PanelEmpty, PanelPill } from '../overlays/panel'

// Initial form state for a blueprint = each field's default (or ''). Pure so the
// suite can assert the form seeds correctly without mounting React.
export function initialBlueprintValues(blueprint: AutomationBlueprint): Record<string, string> {
  const out: Record<string, string> = {}
  for (const field of blueprint.fields) {
    out[field.name] = field.default ?? ''
  }
  return out
}

// A slot-level validation error from the backend arrives as "422: <message>"
// (or "<code>: <message>"); strip the leading numeric code for inline display.
function cleanFieldError(message: string): string {
  return message.replace(/^\d+:\s*/, '')
}

function FieldInput({
  field,
  id,
  value,
  onChange
}: {
  field: AutomationBlueprintField
  id: string
  value: string
  onChange: (next: string) => void
}) {
  if (field.type === 'enum' || field.type === 'weekdays') {
    return (
      <Select onValueChange={onChange} value={value}>
        <SelectTrigger className="h-9 rounded-md" id={id}>
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          {field.options.map(option => (
            <SelectItem key={option} value={option}>
              {option}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
    )
  }

  if (field.type === 'time') {
    return <Input id={id} onChange={event => onChange(event.target.value)} type="time" value={value} />
  }

  return (
    <Input
      id={id}
      onChange={event => onChange(event.target.value)}
      placeholder={field.help || field.label}
      type="text"
      value={value}
    />
  )
}

function BlueprintCard({
  blueprint,
  c,
  profile,
  onCreated
}: {
  blueprint: AutomationBlueprint
  c: Translations['cron']
  profile: string
  onCreated: (job: CronJob) => void
}) {
  const [open, setOpen] = useState(false)
  const [values, setValues] = useState<Record<string, string>>(() => initialBlueprintValues(blueprint))
  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState<null | string>(null)

  const submit = useCallback(async () => {
    setSubmitting(true)
    setError(null)

    try {
      const job = await instantiateAutomationBlueprint({ blueprint: blueprint.key, values }, profile)
      onCreated(job)
      notify({ kind: 'success', title: c.blueprints.scheduled, message: asText(job.schedule_display) || blueprint.title })
      setOpen(false)
      setValues(initialBlueprintValues(blueprint))
    } catch (err) {
      // 422 carries the slot-level message; surface it inline on the form.
      setError(cleanFieldError(err instanceof Error ? err.message : String(err)))
    } finally {
      setSubmitting(false)
    }
  }, [blueprint, values, profile, onCreated, c])

  return (
    // A card container — NOT PanelBlock. PanelBlock is a height-capped, scrollable
    // <pre> for monospace code; using it here clipped each blueprint's copy at
    // max-h-48 and forced an inner scroll. A plain auto-height card lets the row
    // grow with its content and the whole gallery scrolls as one.
    <div className="rounded-md bg-foreground/5 p-3">
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <p className="text-sm font-medium text-foreground">{blueprint.title}</p>
          <p className="mt-0.5 text-xs leading-relaxed text-muted-foreground">{blueprint.description}</p>
          {blueprint.tags.length > 0 && (
            <div className="mt-2 flex flex-wrap gap-1">
              {blueprint.tags.map(tag => (
                <PanelPill key={tag}>{tag}</PanelPill>
              ))}
            </div>
          )}
        </div>
        <Button className="shrink-0" onClick={() => setOpen(prev => !prev)} size="sm" variant={open ? 'ghost' : 'outline'}>
          {open ? c.blueprints.cancel : c.blueprints.setUp}
        </Button>
      </div>

      {open && (
        <form
          className="mt-3 grid gap-3 border-t border-border/60 pt-3"
          onSubmit={event => {
            event.preventDefault()
            void submit()
          }}
        >
          {blueprint.fields.map(field => {
            const fieldId = `blueprint-${blueprint.key}-${field.name}`

            return (
              <div className="grid gap-1.5" key={field.name}>
                <label className="text-xs font-medium text-foreground" htmlFor={fieldId}>
                  {field.label}
                </label>
                <FieldInput
                  field={field}
                  id={fieldId}
                  onChange={next => setValues(prev => ({ ...prev, [field.name]: next }))}
                  value={values[field.name] ?? ''}
                />
                {field.help && field.type !== 'text' && (
                  <p className="text-[0.66rem] leading-4 text-muted-foreground">{field.help}</p>
                )}
              </div>
            )
          })}

          {error && (
            <p className="text-xs text-destructive" role="alert">
              {error}
            </p>
          )}

          <div>
            <Button disabled={submitting} size="sm" type="submit">
              {submitting ? c.blueprints.scheduling : c.blueprints.scheduleIt}
            </Button>
          </div>
        </form>
      )}
    </div>
  )
}

// Automation Blueprints gallery \u2014 the desktop counterpart to the dashboard's
// blueprint tab. Each card expands into an inline form (one field per typed
// slot); submitting POSTs to /api/cron/blueprints/instantiate, which fills the
// blueprint and creates the job via the same create_job path as a hand-written
// cron. The created job is spliced straight into the shared $cronJobs atom so
// the Jobs tab and sidebar reflect it immediately.
export function BlueprintsPanel({ profile }: { profile: string }) {
  const { t } = useI18n()
  const c = t.cron

  const blueprints = useQuery({
    queryKey: ['cron-blueprints', profile],
    queryFn: async () => (await getAutomationBlueprints()).blueprints
  })

  const handleCreated = useCallback((job: CronJob) => {
    // Merge, don't clobber: keep the existing rows and add/replace this one.
    updateCronJobs(rows => {
      const rest = rows.filter(row => row.id !== job.id)
      return [...rest, job]
    })
  }, [])

  const cards = useMemo(() => blueprints.data ?? [], [blueprints.data])

  if (blueprints.isLoading) {
    return <PageLoader label={c.blueprints.loading} />
  }

  if (blueprints.isError) {
    return <PanelEmpty description={c.blueprints.failedLoad} icon="warning" title={c.blueprints.failedLoad} />
  }

  if (cards.length === 0) {
    return <PanelEmpty description={c.blueprints.emptyDesc} icon="lightbulb" title={c.blueprints.emptyTitle} />
  }

  return (
    <div className="flex min-h-0 flex-1 flex-col gap-2 overflow-y-auto pr-1">
      {cards.map(blueprint => (
        <BlueprintCard blueprint={blueprint} c={c} key={blueprint.key} onCreated={handleCreated} profile={profile} />
      ))}
    </div>
  )
}
