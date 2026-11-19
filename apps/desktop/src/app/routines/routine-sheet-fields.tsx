import { useStore } from '@nanostores/react'
import { useQuery } from '@tanstack/react-query'

import { Checkbox } from '@/components/ui/checkbox'
import { Codicon } from '@/components/ui/codicon'
import { Field } from '@/components/ui/field'
import { Input } from '@/components/ui/input'
import { Select, SelectContent, SelectGroup, SelectItem, SelectLabel, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Tip } from '@/components/ui/tooltip'
import { useI18n } from '@/i18n'
import { requestModelOptions } from '@/lib/model-options'
import { $profiles } from '@/store/profile'

import { cronModelChoiceValue, parseCronModelChoiceValue } from '../cron/cron-job-model'
import { CATCH_UP_CHOICES, type CatchUpChoice } from '../cron/run-control'
import { SettingsDisclosure } from '../settings/settings-card'

import { MODEL_DEFAULT, type RoutineDraft } from './routine-draft'
import { AgentSelect } from './routine-fields'
import { useAgentLabel } from './routine-parts'
import { useRoutinesCopy } from './routines-copy'

/** A field's description, moved into a tooltip (rev 1399:22870): an info button that opens it on
 *  hover and on keyboard focus. */
export function InfoTip({ hint, label }: { hint: string; label: string }) {
  const copy = useRoutinesCopy()

  return (
    <Tip label={hint}>
      <button aria-label={copy.aboutField(label)} className="routines-info" type="button">
        <Codicon aria-hidden name="info" size="0.875rem" />
      </button>
    </Tip>
  )
}

/** Only an agent list longer than one needs a choice ("Agent picker only shows if they have multiple
 *  agents"). */
export function useHasAgentChoice(): boolean {
  return useStore($profiles).length > 1
}

interface AgentFieldProps {
  profile: string
  /** Editing: the agent shows, but a routine never moves to another agent. */
  locked: boolean
  onChange: (profile: string) => void
}

export function AgentField({ locked, onChange, profile }: AgentFieldProps) {
  const copy = useRoutinesCopy()
  const agentLabel = useAgentLabel()

  if (!locked) {
    return (
      <Field htmlFor="routine-agent" label={copy.agent}>
        <AgentSelect id="routine-agent" onChange={onChange} value={profile} />
      </Field>
    )
  }

  // A disabled control takes no pointer or focus, so it stays a read-only field that reads disabled:
  // hover and keyboard focus both reach its tooltip.
  return (
    <Field htmlFor="routine-agent" label={copy.agent}>
      <Tip label={copy.agentLocked}>
        <Input aria-disabled className="routines-locked" id="routine-agent" readOnly value={agentLabel(profile)} />
      </Tip>
    </Field>
  )
}

export type StartFrom = 'blank' | 'template'

const START_ICONS: Record<StartFrom, string> = { blank: 'file', template: 'folder-library' }

interface StartFromFieldProps {
  value: null | StartFrom
  onChange: (value: StartFrom) => void
}

/** Start from: Blank or Template, two tiles that are one radio group (arrow keys move between them). */
export function StartFromField({ onChange, value }: StartFromFieldProps) {
  const copy = useRoutinesCopy()

  const tiles: { id: StartFrom; title: string; hint: string }[] = [
    { hint: copy.blankHint, id: 'blank', title: copy.blank },
    { hint: copy.templateHint, id: 'template', title: copy.template }
  ]

  return (
    <fieldset className="routines-field">
      <legend>{copy.startFrom}</legend>
      <div className="routines-tiles">
        {tiles.map(tile => (
          <label className="routines-tile" data-checked={value === tile.id || undefined} key={tile.id}>
            <input
              checked={value === tile.id}
              className="sr-only"
              name="routine-start-from"
              onChange={() => onChange(tile.id)}
              // A second click on Template returns to the list after one was picked.
              onClick={() => value === tile.id && onChange(tile.id)}
              type="radio"
              value={tile.id}
            />
            <Codicon aria-hidden className="routines-tile-icon" name={START_ICONS[tile.id]} size="1.25rem" />
            <span className="routines-tile-text">
              <span className="routines-tile-title">{tile.title}</span>
              <span className="routines-tile-hint">{tile.hint}</span>
            </span>
          </label>
        ))}
      </div>
    </fieldset>
  )
}

interface AdvancedFieldsProps {
  draft: RoutineDraft
  set: (patch: Partial<RoutineDraft>) => void
  /** A script-only routine has no model to pick. */
  scriptOnly: boolean
  /** Developer mode on an existing routine: its id. */
  routineId?: string
}

/** Advanced, collapsed by default: model, missed runs, run limit, remember last run; each description
 *  in its field's tooltip. */
export function AdvancedFields({ draft, routineId, scriptOnly, set }: AdvancedFieldsProps) {
  const { t } = useI18n()
  const c = t.cron.control
  const copy = useRoutinesCopy()

  return (
    <SettingsDisclosure className="routines-advanced-disclosure" title={copy.advanced}>
      <div className="routines-advanced">
        {!scriptOnly && <ModelField onChange={model => set({ model })} value={draft.model} />}
        <Field
          htmlFor="routine-catch-up"
          label={
            <span className="routines-label-row">
              {c.catchUpLabel}
              <InfoTip hint={copy.missedRunsHint} label={c.catchUpLabel} />
            </span>
          }
        >
          <Select onValueChange={next => set({ catchUp: next as CatchUpChoice })} value={draft.catchUp}>
            <SelectTrigger id="routine-catch-up">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {CATCH_UP_CHOICES.map(choice => (
                <SelectItem key={choice} value={choice}>
                  {c.catchUpChoices[choice]}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </Field>
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
        <div className="routines-label-row">
          <label className="routines-check">
            <Checkbox checked={draft.remember} onCheckedChange={next => set({ remember: next === true })} />
            <span>{copy.rememberLastRun}</span>
          </label>
          <InfoTip hint={copy.rememberHint} label={copy.rememberLastRun} />
        </div>
        {routineId && (
          <p className="routines-field-note">
            {copy.routineId}: <code>{routineId}</code>
          </p>
        )}
      </div>
    </SettingsDisclosure>
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
