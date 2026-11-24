import type * as React from 'react'

import { Input } from './input'

interface SanitizedInputProps extends Omit<React.ComponentProps<typeof Input>, 'onChange' | 'value'> {
  value: string
  onValueChange: (value: string) => void
  // A formatter from `@/lib/sanitize` (gitRef, slug, …) run on every keystroke.
  sanitize: (raw: string) => string
}

// An <Input> that can only ever hold a valid value: every keystroke is run
// through `sanitize`, so callers never have to validate-then-reject (a space in
// a branch name becomes "-" as you type instead of erroring at submit).
export function SanitizedInput({ value, onValueChange, sanitize, ...props }: SanitizedInputProps) {
  return <Input {...props} onChange={event => onValueChange(sanitize(event.target.value))} value={value} />
}
