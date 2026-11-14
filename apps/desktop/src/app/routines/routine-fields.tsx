import { useStore } from '@nanostores/react'
import { useQuery } from '@tanstack/react-query'

import { Checkbox } from '@/components/ui/checkbox'
import { FieldHint } from '@/components/ui/field'
import { Input } from '@/components/ui/input'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { getOwnerCronDeliveryTargets } from '@/hermes'
import { useI18n } from '@/i18n'
import { $interfaceMode } from '@/store/interface-mode'
import { $profiles } from '@/store/profile'

import { agentChatTarget, deliveryTargets, toggleDelivery } from './routine-model'
import { useAgentLabel } from './routine-parts'
import { describePick, FREQUENCIES, type Frequency, type SchedulePick } from './routine-schedule'
import type { RoutineSource } from './routine-source'
import { useRoutinesCopy } from './routines-copy'

interface AgentSelectProps {
  id: string
  value: string
  onChange: (profile: string) => void
}

/** The agent a new routine belongs to: this machine's agents, by their display names. */
export function AgentSelect({ id, onChange, value }: AgentSelectProps) {
  const profiles = useStore($profiles)
  const agentLabel = useAgentLabel()
  const names = profiles.map(profile => profile.name)
  // A routine's agent the list does not know yet still shows by its name.
  const listed = names.includes(value) ? names : [value, ...names]

  return (
    <Select onValueChange={onChange} value={value}>
      <SelectTrigger id={id}>
        <SelectValue />
      </SelectTrigger>
      <SelectContent>
        {listed.map(name => (
          <SelectItem key={name} value={name}>
            {agentLabel(name)}
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  )
}

const WEEKDAY_ORDER = ['1', '2', '3', '4', '5', '6', '0']
const NEEDS_TIME = new Set<Frequency>(['daily', 'monthly', 'once', 'weekdays', 'weekly'])

interface ScheduleFieldProps {
  pick: SchedulePick
  onChange: (pick: SchedulePick) => void
}

/** When to run (R4): a frequency, then only the detail it needs, read back in plain words. Custom
 *  takes plain words ("every 2 hours"); developer mode also documents cron expressions there. */
export function ScheduleField({ onChange, pick }: ScheduleFieldProps) {
  const { locale, t } = useI18n()
  const copy = useRoutinesCopy()
  const developer = useStore($interfaceMode) === 'advanced'
  const set = (patch: Partial<SchedulePick>) => onChange({ ...pick, ...patch })
  const words = { copy, dayName: (day: string) => t.cron.days[day] ?? day, locale }
  const readBack = describePick(pick, words)

  return (
    <fieldset className="routines-field">
      <legend>{copy.whenToRun}</legend>
      <div className="routines-when">
        <Select onValueChange={value => set({ frequency: value as Frequency })} value={pick.frequency}>
          <SelectTrigger aria-label={copy.whenToRun} className="routines-when-frequency">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {FREQUENCIES.map(frequency => (
              <SelectItem key={frequency} value={frequency}>
                {copy.frequencies[frequency]}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
        {pick.frequency === 'weekly' && (
          <Select onValueChange={weekday => set({ weekday })} value={pick.weekday}>
            <SelectTrigger aria-label={copy.frequencies.weekly} className="routines-when-day">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {WEEKDAY_ORDER.map(day => (
                <SelectItem key={day} value={day}>
                  {t.cron.days[day] ?? day}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        )}
        {pick.frequency === 'monthly' && (
          <label className="routines-when-inline">
            <span>{copy.onDay}</span>
            <Input
              className="routines-when-number"
              inputMode="numeric"
              max={31}
              min={1}
              onChange={event => set({ monthday: event.target.value.replace(/\D/g, '').slice(0, 2) })}
              type="number"
              value={pick.monthday}
            />
          </label>
        )}
        {pick.frequency === 'once' && (
          <Input
            aria-label={copy.frequencies.once}
            className="routines-when-date"
            onChange={event => set({ date: event.target.value })}
            type="date"
            value={pick.date}
          />
        )}
        {NEEDS_TIME.has(pick.frequency) && (
          <label className="routines-when-inline">
            <span>{copy.at}</span>
            <Input
              className="routines-when-time"
              onChange={event => set({ time: event.target.value })}
              type="time"
              value={pick.time}
            />
          </label>
        )}
        {readBack && (
          <span aria-live="polite" className="routines-readback">
            <span aria-hidden className="routines-dot" data-tone="good" />
            {readBack}
          </span>
        )}
      </div>
      {pick.frequency === 'custom' && (
        <>
          <Input
            aria-label={copy.frequencies.custom}
            className={developer ? 'font-mono' : undefined}
            onChange={event => set({ custom: event.target.value })}
            placeholder={copy.customPlaceholder}
            value={pick.custom}
          />
          <FieldHint>{developer ? copy.customHintDeveloper : copy.customHint}</FieldHint>
        </>
      )}
    </fieldset>
  )
}

interface DeliveryFieldProps {
  profile: string
  source: RoutineSource
  value: string
  onChange: (deliver: string) => void
}

/** Send results to (R4): the agent's own chat and the messaging channels its machine has connected,
 *  as checkboxes. Nothing checked keeps today's save-only behaviour: every run is kept in Runs. */
export function DeliveryField({ onChange, profile, source, value }: DeliveryFieldProps) {
  const { t } = useI18n()
  const copy = useRoutinesCopy()
  const agentLabel = useAgentLabel()
  const scope = source.createScope(profile)

  const targets = useQuery({
    queryKey: ['cron-delivery-targets', source.connectionId ?? '', profile],
    queryFn: () => getOwnerCronDeliveryTargets(scope)
  })

  const selected = deliveryTargets(value)

  // A routine's own chat, channels; other agents' chats are theirs. Saved routes the list does not
  // offer (an older target, another agent's chat) stay listed so an edit never drops them.
  const offered = (targets.data ?? []).filter(target => {
    const chat = agentChatTarget(target.id)

    return target.id !== 'local' && (chat === null || chat === '' || chat === profile)
  })

  const options = [
    ...offered,
    ...selected
      .filter(id => !offered.some(target => target.id === id))
      .map(id => ({ home_env_var: null, home_target_set: true, id, name: id }))
  ]

  const label = (id: string, name: string) => {
    const chat = agentChatTarget(id)

    return chat === null ? (t.cron.deliveryLabels[id] ?? name) : copy.agentChat(agentLabel(chat || profile))
  }

  return (
    <fieldset className="routines-field">
      <legend className="routines-field-legend">
        <span>{copy.sendResultsTo}</span>
        <span className="routines-field-note">{copy.everyRunKept}</span>
      </legend>
      <div className="routines-checks" role="group">
        {options.map(target => {
          const blocked = agentChatTarget(target.id) === null && !target.home_target_set

          return (
            <label className="routines-check" data-disabled={blocked || undefined} key={target.id}>
              <Checkbox
                checked={selected.includes(target.id)}
                disabled={blocked && !selected.includes(target.id)}
                onCheckedChange={next => onChange(toggleDelivery(value, target.id, next === true))}
              />
              <span className="routines-check-text">
                <span>{label(target.id, target.name)}</span>
                {blocked && <span className="routines-field-note">{copy.needsHomeChannel}</span>}
              </span>
            </label>
          )
        })}
        {!targets.isPending && !options.some(target => agentChatTarget(target.id) === null) && (
          <p className="routines-field-note">{copy.noChannels}</p>
        )}
        <p className="routines-field-note">{copy.nothingSelected}</p>
      </div>
    </fieldset>
  )
}
