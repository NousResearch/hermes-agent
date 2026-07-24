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
import { ListRow } from '../settings/primitives'

// The blueprint catalog is shared with the dashboard, so its deliver slot
// defaults to "origin" (the chat/home-channel a dashboard or gateway job was
// created from). Desktop has no origin chat and no home-channel picker, so
// "origin" would render unlabeled and, at runtime, deliver nowhere. Treat the
// desktop's native target ("local" = This desktop) as the default and hide the
// origin option — mirroring the manual cron editor, which only offers
// local/telegram/discord/slack/email.
const DELIVER_FIELD = 'deliver'
const DESKTOP_DELIVER_DEFAULT = 'local'

function isDeliverField(field: AutomationBlueprintField): boolean {
  return field.name === DELIVER_FIELD
}

// Options a desktop user can actually deliver to: drop "origin" (dashboard-only)
// and de-dupe. Everything else the backend offered (local + configured
// gateways) passes through.
function desktopDeliverOptions(options: string[]): string[] {
  return [...new Set(options.filter(option => option !== 'origin'))]
}

// Label a deliver value with the desktop's own delivery labels ("This desktop",
// "Telegram", …), falling back to the raw platform id for anything unmapped.
function deliverLabel(value: string, c: Translations['cron']): string {
  return c.deliveryLabels[value] ?? value
}

// Initial form state for a blueprint = each field's default (or ''). Pure so the
// suite can assert the form seeds correctly without mounting React. The deliver
// slot is special-cased: an "origin" default (or empty) becomes "local" so a
// desktop-created job delivers to This desktop instead of nowhere.
export function initialBlueprintValues(blueprint: AutomationBlueprint): Record<string, string> {
  const out: Record<string, string> = {}
  for (const field of blueprint.fields) {
    const seeded = field.default ?? ''
    out[field.name] = isDeliverField(field) && (seeded === '' || seeded === 'origin') ? DESKTOP_DELIVER_DEFAULT : seeded
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
  onChange,
  c
}: {
  field: AutomationBlueprintField
  id: string
  value: string
  onChange: (next: string) => void
  c: Translations['cron']
}) {
  if (field.type === 'enum' || field.type === 'weekdays') {
    const deliver = isDeliverField(field)
    const options = deliver ? desktopDeliverOptions(field.options) : field.options

    return (
      <Select onValueChange={onChange} value={value}>
        <SelectTrigger className="h-9 rounded-md" id={id}>
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          {options.map(option => (
            <SelectItem key={option} value={option}>
              {deliver ? deliverLabel(option, c) : option}
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
  // Only the blueprints slice of the cron strings is used here — narrow the
  // reference (and the submit dep below) instead of leaning on the whole
  // t.cron object.
  const b = c.blueprints

  const submit = useCallback(async () => {
    setSubmitting(true)
    setError(null)

    try {
      const job = await instantiateAutomationBlueprint({ blueprint: blueprint.key, values }, profile)
      onCreated(job)
      notify({ kind: 'success', title: b.scheduled, message: asText(job.schedule_display) || blueprint.title })
      setOpen(false)
      setValues(initialBlueprintValues(blueprint))
    } catch (err) {
      // 422 carries the slot-level message; surface it inline on the form.
      setError(cleanFieldError(err instanceof Error ? err.message : String(err)))
    } finally {
      setSubmitting(false)
    }
  }, [blueprint, values, profile, onCreated, b])

  return (
    // Keep this in the Panel family (matches the cron editor's in-surface
    // groupings) rather than a standalone card look — bg-(--ui-bg-quinary) is the
    // shared in-panel grouping token. NOT PanelBlock: that's a height-capped,
    // scrollable <pre> for monospace code and clipped each blueprint's copy.
    <div className="rounded-md bg-(--ui-bg-quinary) p-3">
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
          {open ? b.cancel : b.setUp}
        </Button>
      </div>

      {open && (
        <form
          className="mt-3 border-t border-border/60 pt-1"
          onSubmit={event => {
            event.preventDefault()
            void submit()
          }}
        >
          {blueprint.fields.map(field => {
            const fieldId = `blueprint-${blueprint.key}-${field.name}`
            // The backend deliver help is origin/dashboard-centric and even
            // contradicts desktop semantics ("local = save only" vs. This
            // desktop), and the relabeled dropdown is self-explanatory — skip it
            // for the deliver slot.
            const help = field.help && field.type !== 'text' && !isDeliverField(field) ? field.help : undefined

            return (
              // Shared settings/messaging field primitive so the slot form matches
              // the rest of the app's forms (label + description on the left, the
              // control on the right; stacks in a narrow pane).
              <ListRow
                action={
                  <FieldInput
                    c={c}
                    field={field}
                    id={fieldId}
                    onChange={next => setValues(prev => ({ ...prev, [field.name]: next }))}
                    value={values[field.name] ?? ''}
                  />
                }
                description={help}
                key={field.name}
                title={<label htmlFor={fieldId}>{field.label}</label>}
              />
            )
          })}

          {error && (
            <p className="mt-2 text-xs text-destructive" role="alert">
              {error}
            </p>
          )}

          <div className="mt-3">
            <Button disabled={submitting} size="sm" type="submit">
              {submitting ? b.scheduling : b.scheduleIt}
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
