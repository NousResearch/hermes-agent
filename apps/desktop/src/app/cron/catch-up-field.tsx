import { Field, FieldHint } from '@/components/ui/field'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import type { Translations } from '@/i18n'

import { CATCH_UP_CHOICES, type CatchUpChoice } from './run-control'

/** Cron editor: what happens to the runs a job missed while Hermes was not running. */
export function CatchUpField({
  c,
  onChange,
  value
}: {
  c: Translations['cron']
  onChange: (value: CatchUpChoice) => void
  value: CatchUpChoice
}) {
  return (
    <Field htmlFor="cron-catch-up" label={c.control.catchUpLabel}>
      <Select onValueChange={next => onChange(next as CatchUpChoice)} value={value}>
        <SelectTrigger className="h-9 rounded-md" id="cron-catch-up">
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          {CATCH_UP_CHOICES.map(choice => (
            <SelectItem key={choice} value={choice}>
              {c.control.catchUpChoices[choice]}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
      <FieldHint>{c.control.catchUpHint}</FieldHint>
    </Field>
  )
}
