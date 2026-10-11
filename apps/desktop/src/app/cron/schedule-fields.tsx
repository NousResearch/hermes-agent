import { Field, FieldHint } from '@/components/ui/field'
import { Input } from '@/components/ui/input'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import type { Translations } from '@/i18n'

import {
  partsWithTime,
  PRESET_FIELDS,
  presetScheduleExpr,
  SCHEDULE_OPTIONS,
  type ScheduleParts,
  scheduleParts,
  scheduleSummary,
  SHORTEST_MONTH_DAYS,
  timeInputValue,
  WEEKDAY_ORDER
} from './schedule'

const DAYS_OF_MONTH = Array.from({ length: 31 }, (_, index) => String(index + 1))

export interface ScheduleValue {
  preset: string
  schedule: string
}

// Frequency + the parts that preset pins (time, weekday, day of month, minute).
// The expression stays the single source of truth: every control reads its
// value out of `schedule` and writes a rebuilt expression back.
export function ScheduleFields({
  c,
  onChange,
  value
}: {
  c: Translations['cron']
  onChange: (next: ScheduleValue) => void
  value: ScheduleValue
}) {
  const { preset, schedule } = value
  const parts = scheduleParts(schedule)
  const fields = PRESET_FIELDS[preset] ?? []
  const option = SCHEDULE_OPTIONS.find(candidate => candidate.value === preset) ?? SCHEDULE_OPTIONS[0]

  const setParts = (next: ScheduleParts) => onChange({ preset, schedule: presetScheduleExpr(preset, next) ?? schedule })

  // Switching presets keeps the time already picked; "Custom" starts from the
  // current expression so it can be tweaked rather than retyped.
  const setPreset = (nextPreset: string) =>
    onChange({ preset: nextPreset, schedule: presetScheduleExpr(nextPreset, parts) ?? schedule })

  return (
    <div className="grid gap-2">
      <div className="grid items-start gap-3 sm:grid-cols-3">
        <Field htmlFor="cron-frequency" label={c.frequencyLabel}>
          <Select onValueChange={setPreset} value={preset}>
            <SelectTrigger className="h-9 rounded-md" id="cron-frequency">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {SCHEDULE_OPTIONS.map(candidate => (
                <SelectItem key={candidate.value} value={candidate.value}>
                  {c.scheduleLabels[candidate.value]}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </Field>

        {fields.includes('dayOfWeek') && (
          <Field htmlFor="cron-day-of-week" label={c.dayOfWeekLabel}>
            <Select
              onValueChange={next => setParts({ ...parts, dayOfWeek: Number(next) })}
              value={String(parts.dayOfWeek)}
            >
              <SelectTrigger className="h-9 rounded-md" id="cron-day-of-week">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {WEEKDAY_ORDER.map(day => (
                  <SelectItem key={day} value={day}>
                    {c.days[day]}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </Field>
        )}

        {fields.includes('dayOfMonth') && (
          <Field htmlFor="cron-day-of-month" label={c.dayOfMonthLabel}>
            <Select
              onValueChange={next => setParts({ ...parts, dayOfMonth: Number(next) })}
              value={String(parts.dayOfMonth)}
            >
              <SelectTrigger className="h-9 rounded-md" id="cron-day-of-month">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {DAYS_OF_MONTH.map(day => (
                  <SelectItem key={day} value={day}>
                    {day}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </Field>
        )}

        {fields.includes('time') && (
          <Field htmlFor="cron-time" label={c.timeLabel}>
            <Input
              id="cron-time"
              onChange={event => setParts(partsWithTime(parts, event.target.value))}
              type="time"
              value={timeInputValue(parts)}
            />
          </Field>
        )}

        {fields.includes('minute') && (
          <Field htmlFor="cron-minute" label={c.minuteLabel}>
            <Input
              id="cron-minute"
              max={59}
              min={0}
              onChange={event => {
                const minute = Number(event.target.value)

                if (event.target.value !== '' && Number.isInteger(minute) && minute >= 0 && minute <= 59) {
                  setParts({ ...parts, minute })
                }
              }}
              type="number"
              value={parts.minute}
            />
          </Field>
        )}
      </div>

      {preset === 'custom' ? (
        <Field htmlFor="cron-schedule" label={c.customScheduleLabel}>
          <Input
            className="font-mono"
            id="cron-schedule"
            onChange={event => onChange({ preset, schedule: event.target.value })}
            placeholder={c.customPlaceholder}
            value={schedule}
          />
          <FieldHint>{c.customHint}</FieldHint>
        </Field>
      ) : (
        <div className="rounded-md bg-(--ui-bg-quinary) px-3 py-2">
          <div className="flex flex-wrap items-center justify-between gap-2 text-xs">
            <span className="font-medium text-foreground">{scheduleSummary(option, schedule, c)}</span>
            <span className="font-mono text-muted-foreground">{schedule}</span>
          </div>
        </div>
      )}

      {fields.includes('dayOfMonth') && parts.dayOfMonth > SHORTEST_MONTH_DAYS && (
        <FieldHint>{c.skipsShortMonths(String(parts.dayOfMonth))}</FieldHint>
      )}
    </div>
  )
}
