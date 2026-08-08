import { Input } from '@/components/ui/input'
import { SegmentedControl } from '@/components/ui/segmented-control'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Switch } from '@/components/ui/switch'
import { Textarea } from '@/components/ui/textarea'
import { Tip } from '@/components/ui/tooltip'
import { Check, Info } from '@/lib/icons'
import type { MemoryProviderField } from '@/types/hermes'

import { CONTROL_TEXT } from '../constants'
import { SearchableSelect } from '../searchable-select'

// Fade the placeholder well below set values so example text never reads as data.
const FIELD_INPUT = `font-mono ${CONTROL_TEXT} placeholder:text-muted-foreground/45`

// Field label with an optional info tooltip, shared by the panel and modal rows.
export function FieldTitle({ field }: { field: MemoryProviderField }) {
  if (!field.info) {
    return <>{field.label}</>
  }

  return (
    <span className="inline-flex items-center gap-1.5">
      {field.label}
      <Tip className="max-w-60 font-normal leading-snug whitespace-normal" label={field.info}>
        <Info aria-label={`About ${field.label}`} className="size-3.5 text-muted-foreground/70" />
      </Tip>
    </span>
  )
}

// Values are edited as strings; the backend coerces them to native types.
export function FieldControl({
  controlId,
  field,
  invalid,
  value,
  onChange,
  onCommit
}: {
  controlId?: string
  field: MemoryProviderField
  invalid?: boolean
  value: string
  onChange: (value: string) => void
  // Present on autosaving surfaces: discrete controls commit on change, text-like
  // controls commit on blur. Absent (the modal), edits stay drafts until Save.
  onCommit?: (value: string) => void
}) {
  const set = (next: string) => {
    onChange(next)
    onCommit?.(next)
  }

  const commitDraft = onCommit ? () => onCommit(value) : undefined

  if (field.kind === 'bool') {
    return (
      <Switch
        aria-invalid={invalid || undefined}
        checked={value === 'true'}
        disabled={field.read_only}
        id={controlId}
        onCheckedChange={checked => set(checked ? 'true' : 'false')}
      />
    )
  }

  if (field.kind === 'number') {
    return (
      <Input
        aria-invalid={invalid || undefined}
        aria-required={field.required || undefined}
        className={FIELD_INPUT}
        disabled={field.read_only}
        id={controlId}
        inputMode="numeric"
        onBlur={commitDraft}
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
        aria-invalid={invalid || undefined}
        aria-required={field.required || undefined}
        className={FIELD_INPUT}
        disabled={field.read_only}
        id={controlId}
        onBlur={commitDraft}
        onChange={event => onChange(event.target.value)}
        placeholder={field.placeholder}
        spellCheck={false}
        value={value}
      />
    )
  }

  if (field.kind === 'select') {
    if (field.searchable) {
      return (
        <SearchableSelect
          disabled={field.read_only}
          emptyMessage={field.placeholder || 'No options found.'}
          id={controlId}
          invalid={invalid}
          onChange={set}
          options={field.options.map(option => ({ label: option.label, value: option.value }))}
          placeholder={field.search_placeholder || 'Search options...'}
          required={field.required}
          value={value}
        />
      )
    }

    return (
      <Select disabled={field.read_only} onValueChange={set} value={value}>
        <SelectTrigger
          aria-invalid={invalid || undefined}
          aria-required={field.required || undefined}
          className={CONTROL_TEXT}
          id={controlId}
        >
          <SelectValue placeholder={field.placeholder || undefined} />
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

  if (field.kind === 'segmented') {
    return (
      <SegmentedControl
        className="w-full"
        disabled={field.read_only}
        onChange={set}
        options={field.options.map(option => ({ id: option.value, label: option.label }))}
        value={value}
      />
    )
  }

  if (field.kind === 'secret') {
    return (
      <div className="flex flex-col gap-1">
        <Input
          aria-invalid={invalid || undefined}
          aria-required={field.required || undefined}
          className={`w-full ${FIELD_INPUT}`}
          disabled={field.read_only}
          id={controlId}
          onBlur={commitDraft}
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
      aria-invalid={invalid || undefined}
      aria-required={field.required || undefined}
      className={FIELD_INPUT}
      disabled={field.read_only}
      id={controlId}
      onBlur={commitDraft}
      onChange={event => onChange(event.target.value)}
      placeholder={field.placeholder}
      value={value}
    />
  )
}
