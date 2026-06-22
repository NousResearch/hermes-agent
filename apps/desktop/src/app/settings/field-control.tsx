import { Input } from '@/components/ui/input'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Switch } from '@/components/ui/switch'
import { Textarea } from '@/components/ui/textarea'
import { Check } from '@/lib/icons'
import type { MemoryProviderField } from '@/types/hermes'

import { CONTROL_TEXT } from './constants'

// Fade the placeholder well below set values so example text never reads as data.
const FIELD_INPUT = `font-mono ${CONTROL_TEXT} placeholder:text-muted-foreground/45`

/** Generic control for one declared provider field, dispatching on its kind.
 *  Values are edited as strings; the backend coerces them to native types. */
export function FieldControl({
  field,
  value,
  onChange
}: {
  field: MemoryProviderField
  value: string
  onChange: (value: string) => void
}) {
  if (field.kind === 'bool') {
    return <Switch checked={value === 'true'} onCheckedChange={checked => onChange(checked ? 'true' : 'false')} />
  }

  if (field.kind === 'number') {
    return (
      <Input
        className={FIELD_INPUT}
        inputMode="numeric"
        onChange={event => onChange(event.target.value)}
        placeholder={field.placeholder}
        type="number"
        value={value}
      />
    )
  }

  if (field.kind === 'json') {
    return (
      <Textarea
        className={FIELD_INPUT}
        onChange={event => onChange(event.target.value)}
        placeholder={field.placeholder}
        spellCheck={false}
        value={value}
      />
    )
  }

  if (field.kind === 'select') {
    return (
      <Select onValueChange={onChange} value={value}>
        <SelectTrigger className={CONTROL_TEXT}>
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          {field.options.map(option => (
            <SelectItem key={option.value} value={option.value}>
              {option.label}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
    )
  }

  if (field.kind === 'secret') {
    return (
      <div className="flex flex-col gap-1">
        <Input
          className={`w-full ${FIELD_INPUT}`}
          onChange={event => onChange(event.target.value)}
          placeholder={field.is_set ? 'Leave blank to keep current value' : field.placeholder}
          type="password"
          value={value}
        />
        {field.is_set && (
          <span className="inline-flex items-center gap-1 self-start font-mono text-[0.65rem] text-(--ui-text-tertiary)">
            <Check className="size-3 text-(--ui-accent-secondary)" />
            set
          </span>
        )}
      </div>
    )
  }

  return (
    <Input
      className={FIELD_INPUT}
      onChange={event => onChange(event.target.value)}
      placeholder={field.placeholder}
      value={value}
    />
  )
}
