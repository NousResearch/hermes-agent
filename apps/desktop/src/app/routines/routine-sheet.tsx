import { useStore } from '@nanostores/react'
import { useQuery } from '@tanstack/react-query'
import { useRef, useState } from 'react'

import { Button } from '@/components/ui/button'
import { Field, FieldHint } from '@/components/ui/field'
import { Input } from '@/components/ui/input'
import { Sheet, SheetContent, SheetDescription, SheetHeader, SheetTitle } from '@/components/ui/sheet'
import { Textarea } from '@/components/ui/textarea'
import { type AutomationBlueprint, createCronJob, type CronJob, getAutomationBlueprints, updateCronJob } from '@/hermes'
import { useI18n } from '@/i18n'
import { firstRunText, nextFirstRun } from '@/lib/routine-first-run'
import { $interfaceMode } from '@/store/interface-mode'
import { notify, notifyError } from '@/store/notifications'
import { $activeGatewayProfile } from '@/store/profile'

import { jobIsScriptOnly } from '../cron/cron-job-model'

import {
  blankDraft,
  createPayload,
  type DraftError,
  draftFromJob,
  draftFromTemplate,
  type RoutineDraft,
  updatePayload,
  validateDraft
} from './routine-draft'
import { DeliveryField, ScheduleField } from './routine-fields'
import { routineTitle } from './routine-model'
import { AdvancedFields, AgentField, type StartFrom, StartFromField, useHasAgentChoice } from './routine-sheet-fields'
import { ROUTINE_FORM_ID, RoutineSheetFooter } from './routine-sheet-footer'
import type { RoutineSource } from './routine-source'
import { RoutineTemplates } from './routine-templates'
import { useRoutinesCopy } from './routines-copy'
import { $routineSheet, closeRoutineSheet, refOf, sameRoutine, selectRoutine, type RoutineSheet as SheetState } from './routines-store'

/** Where each draft error sends focus: the field that has to change. */
const ERROR_FIELD: Record<DraftError, string> = { prompt: 'routine-instruction', schedule: 'routine-frequency' }

interface RoutineSheetProps {
  source: RoutineSource
  /** Delete was confirmed in the edit sheet's footer: the page holds it, with Undo. */
  onDelete: (job: CronJob) => void
}

/** The one place a routine is created or changed (rev 1394:21355): a sheet from the stage's end, full
 *  width on compact. Esc, the scrim, × and Cancel close it; focus returns to what opened it. */
export function RoutineSheet({ onDelete, source }: RoutineSheetProps) {
  const sheet = useStore($routineSheet)
  const editing = sheet.mode === 'edit' ? (source.jobs.find(job => sameRoutine(job, sheet.ref)) ?? null) : null
  // An edit whose routine left the list (deleted elsewhere) has nothing to show.
  const open = sheet.mode === 'create' || editing !== null
  // Opened by a store write, not a Radix trigger: remember what had focus so closing returns there.
  const opener = useRef<HTMLElement | null>(null)

  return (
    <Sheet onOpenChange={next => !next && closeRoutineSheet()} open={open}>
      <SheetContent
        className="routines-sheet"
        // The page's control language (buttons, fields, selects) applies inside the sheet too.
        data-page-surface=""
        onCloseAutoFocus={event => {
          if (opener.current?.isConnected) {
            event.preventDefault()
            opener.current.focus({ preventScroll: true })
          }

          opener.current = null
        }}
        onOpenAutoFocus={() => {
          opener.current = document.activeElement instanceof HTMLElement ? document.activeElement : null
        }}
        side="right"
      >
        {open && (
          <RoutineSheetBody
            job={editing}
            key={editing ? editing.id : 'create'}
            onDelete={onDelete}
            sheet={sheet}
            source={source}
          />
        )}
      </SheetContent>
    </Sheet>
  )
}

interface BodyProps {
  sheet: SheetState
  job: CronJob | null
  source: RoutineSource
  onDelete: (job: CronJob) => void
}

function initialDraft(sheet: SheetState, job: CronJob | null, fallbackProfile: string): RoutineDraft {
  if (job) {
    return draftFromJob(job)
  }

  return blankDraft((sheet.mode === 'create' && sheet.profile) || fallbackProfile || 'default')
}

/** The ready-made catalog for the agent a new routine is for, read once Template is chosen. */
function useTemplates(source: RoutineSource, profile: string, wanted: boolean) {
  return useQuery({
    enabled: wanted && Boolean(profile),
    queryFn: async () => (await getAutomationBlueprints(profile, source.connectionId)).blueprints,
    queryKey: ['cron-blueprints', source.connectionId ?? '', profile]
  })
}

function RoutineSheetBody({ job, onDelete, sheet, source }: BodyProps) {
  const { locale, t } = useI18n()
  const copy = useRoutinesCopy()
  const activeProfile = useStore($activeGatewayProfile)
  const developer = useStore($interfaceMode) === 'advanced'
  const agentChoice = useHasAgentChoice()
  const [draft, setDraft] = useState(() => initialDraft(sheet, job, activeProfile))
  const [start, setStart] = useState<null | StartFrom>(null)
  const [template, setTemplate] = useState<AutomationBlueprint | null>(null)
  const [error, setError] = useState<DraftError | null>(null)
  const [saving, setSaving] = useState(false)
  const [confirmingDelete, setConfirmingDelete] = useState(false)
  const templates = useTemplates(source, draft.profile, !job && start === 'template')
  const set = (patch: Partial<RoutineDraft>) => setDraft(current => ({ ...current, ...patch }))
  const choosing = !job && start === 'template' && !template
  const showFields = Boolean(job) || start === 'blank' || Boolean(template)

  const pickStart = (next: StartFrom) => {
    setStart(next)
    setTemplate(null)

    if (next === 'blank') {
      set({ skills: [] })
    }
  }

  const pickTemplate = (picked: AutomationBlueprint) => {
    setTemplate(picked)
    setDraft(current => draftFromTemplate(picked, current))
  }

  async function submit(event: React.FormEvent) {
    event.preventDefault()
    const invalid = validateDraft(draft, job ?? undefined)
    setError(invalid)

    if (invalid) {
      const field = document.getElementById(ERROR_FIELD[invalid])
      const control = field?.matches('input, textarea') ? field : field?.querySelector<HTMLElement>('button, input, textarea')
      control?.focus()

      return
    }

    setSaving(true)

    try {
      const saved = await save(job, draft, source)

      if (saved) {
        // A new routine's toast says when it first runs (rev 1406:24302).
        const first = job ? null : nextFirstRun(draft.pick, Date.now())
        const when = first ? copy.firstRun(firstRunText(first, locale, Date.now())) : ''

        notify({
          kind: 'success',
          message: [routineTitle(saved), when].filter(Boolean).join(' · '),
          title: job ? t.cron.updated : t.cron.created
        })
        closeRoutineSheet()
        selectRoutine(refOf(saved))
      }
    } catch (err) {
      notifyError(err, t.cron.failedSave)
    } finally {
      setSaving(false)
    }
  }

  return (
    <>
      <div className="routines-sheet-scroll">
        <SheetHeader className="routines-sheet-header">
          <SheetTitle>{job ? copy.editTitle(routineTitle(job)) : copy.createTitle}</SheetTitle>
          <SheetDescription className="sr-only">{job ? routineTitle(job) : copy.emptyDescription}</SheetDescription>
        </SheetHeader>
        <form className="routines-form" id={ROUTINE_FORM_ID} noValidate onSubmit={submit}>
          {agentChoice && (
            <AgentField locked={Boolean(job)} onChange={profile => set({ profile })} profile={draft.profile} />
          )}
          {!job && <StartFromField onChange={pickStart} value={start} />}
          {choosing && (
            <RoutineTemplates
              failed={templates.isError}
              loading={templates.isPending}
              onPick={pickTemplate}
              onRetry={() => void templates.refetch()}
              templates={templates.data ?? []}
            />
          )}
          {template && (
            <p className="routines-field-note">
              {copy.fromTemplate(template.title)}{' '}
              <Button onClick={() => setTemplate(null)} size="inline" type="button" variant="textStrong">
                {copy.actions.change}
              </Button>
            </p>
          )}
          {showFields && (
            <RoutineFields
              developer={developer}
              draft={draft}
              error={error}
              job={job}
              set={patch => {
                set(patch)
                setError(null)
              }}
              source={source}
            />
          )}
        </form>
      </div>
      <RoutineSheetFooter
        confirmingDelete={confirmingDelete}
        draft={draft}
        job={job}
        onConfirmDelete={setConfirmingDelete}
        onDelete={() => {
          if (job) {
            closeRoutineSheet()
            onDelete(job)
          }
        }}
        ready={showFields}
        saving={saving}
        source={source}
      />
    </>
  )
}

interface FieldsProps {
  draft: RoutineDraft
  set: (patch: Partial<RoutineDraft>) => void
  error: DraftError | null
  job: CronJob | null
  source: RoutineSource
  developer: boolean
}

/** Name, instruction, frequency, where results go, and Advanced. */
function RoutineFields({ developer, draft, error, job, set, source }: FieldsProps) {
  const copy = useRoutinesCopy()
  const scriptOnly = job ? jobIsScriptOnly(job) : false

  return (
    <>
      <Field htmlFor="routine-name" label={copy.name} optional optionalLabel={copy.optional}>
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
            aria-describedby={error === 'prompt' ? 'routine-instruction-error' : undefined}
            aria-invalid={error === 'prompt' || undefined}
            className="routines-instruction-input"
            id="routine-instruction"
            onChange={event => set({ prompt: event.target.value })}
            placeholder={copy.instructionPlaceholder}
            value={draft.prompt}
          />
          {error === 'prompt' && (
            <span id="routine-instruction-error">
              <FieldHint error>{copy.instructionRequired}</FieldHint>
            </span>
          )}
        </Field>
      )}
      <ScheduleField invalid={error === 'schedule'} onChange={pick => set({ pick })} pick={draft.pick} />
      <DeliveryField onChange={deliver => set({ deliver })} profile={draft.profile} source={source} value={draft.deliver} />
      <AdvancedFields draft={draft} routineId={developer && job ? job.id : undefined} scriptOnly={scriptOnly} set={set} />
    </>
  )
}

/** Create through the ordinary job path, or update only what changed; null when a switch made the
 *  answer stale. */
function save(job: CronJob | null, draft: RoutineDraft, source: RoutineSource): Promise<CronJob | null> {
  return source.mutate(() =>
    job
      ? updateCronJob(job.id, updatePayload(draft, job), source.scopeFor(job))
      : createCronJob(createPayload(draft), source.createScope(draft.profile))
  )
}
