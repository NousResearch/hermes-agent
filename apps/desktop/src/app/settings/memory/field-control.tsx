import type { ReactNode } from 'react'

import { Input } from '@/components/ui/input'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Switch } from '@/components/ui/switch'
import { Textarea } from '@/components/ui/textarea'
import { Tip } from '@/components/ui/tooltip'
import { useI18n } from '@/i18n'
import { Check, Info } from '@/lib/icons'
import type { MemoryProviderField, MemoryProviderFieldKind } from '@/types/hermes'

import { CONTROL_TEXT } from '../constants'

// Fade the placeholder well below set values so example text never reads as data.
const FIELD_INPUT = `font-mono ${CONTROL_TEXT} placeholder:text-muted-foreground/45`

export function FieldTitle({ field }: { field: MemoryProviderField }) {
  const { t } = useI18n()

  if (!field.info) {
    return <>{field.label}</>
  }

  return (
    <span className="inline-flex items-center gap-1.5">
      {field.label}
      <Tip className="max-w-60 font-normal leading-snug whitespace-normal" label={field.info}>
        <Info aria-label={t.memoryProviders.about} className="size-3.5 text-muted-foreground/70" />
      </Tip>
    </span>
  )
}

interface ControlProps {
  field: MemoryProviderField
  value: string
  onChange: (value: string) => void
}

const textInput = ({ field, value, onChange }: ControlProps, extra: Record<string, unknown> = {}) => (
  <Input
    aria-label={field.label}
    aria-required={field.required}
    className={FIELD_INPUT}
    onChange={event => onChange(event.target.value)}
    placeholder={field.placeholder}
    required={field.required}
    value={value}
    {...extra}
  />
)

// Values are edited as strings; the backend coerces them to native types.
const controls: Record<
  MemoryProviderFieldKind,
  (props: ControlProps, keepSecret: string, secretSet: string) => ReactNode
> = {
  bool: ({ field, value, onChange }) => (
    <Switch
      aria-label={field.label}
      aria-required={field.required}
      checked={value === 'true'}
      onCheckedChange={checked => onChange(checked ? 'true' : 'false')}
    />
  ),
  number: props =>
    textInput(props, {
      inputMode: 'numeric',
      max: props.field.maximum ?? undefined,
      min: props.field.minimum ?? undefined,
      step: props.field.step ?? 'any',
      type: 'number'
    }),
  json: ({ field, value, onChange }) => (
    <Textarea
      aria-label={field.label}
      aria-required={field.required}
      className={FIELD_INPUT}
      onChange={event => onChange(event.target.value)}
      placeholder={field.placeholder}
      required={field.required}
      spellCheck={false}
      value={value}
    />
  ),
  select: ({ field, value, onChange }) => (
    <Select onValueChange={onChange} required={field.required} value={value}>
      <SelectTrigger aria-label={field.label} aria-required={field.required} className={CONTROL_TEXT}>
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
  ),
  // A blank secret keeps the stored one, so a set secret is never required again.
  secret: (props, keepSecret, secretSet) => (
    <div className="flex flex-col gap-1">
      {textInput(props, {
        className: `w-full ${FIELD_INPUT}`,
        placeholder: props.field.is_set ? keepSecret : props.field.placeholder,
        required: props.field.required && !props.field.is_set,
        type: 'password'
      })}
      {props.field.is_set && (
        <span className="inline-flex items-center gap-1 self-start font-mono text-[0.65rem] text-(--ui-text-tertiary)">
          <Check className="size-3 text-(--ui-accent-secondary)" />
          {secretSet}
        </span>
      )}
    </div>
  ),
  text: props => textInput(props)
}

export function FieldControl(props: ControlProps) {
  const { t } = useI18n()

  return (controls[props.field.kind] ?? controls.text)(props, t.memoryProviders.keepSecret, t.memoryProviders.secretSet)
}
