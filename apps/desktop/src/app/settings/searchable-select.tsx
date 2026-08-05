import { useCallback, useRef, useState } from 'react'

import { Codicon } from '@/components/ui/codicon'
import { Command, CommandEmpty, CommandGroup, CommandInput, CommandItem, CommandList } from '@/components/ui/command'
import { controlVariants } from '@/components/ui/control'
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover'
import { cn } from '@/lib/utils'

export interface SearchableSelectOption {
  label: string
  value: string
}

type SearchableSelectOptionInput = SearchableSelectOption | string

/**
 * cmdk filter score for one option. Case-insensitive substring match, with
 * the final path segment (after the last "/") ranked above matches anywhere
 * else so "york" ranks "America/New_York" over "America/New_York/Special".
 * Exported for tests.
 */
export function rankSearchOption(option: string, search: string, keywords: string[] = []): number {
  const lower = search.toLowerCase()
  let score = 0

  for (const candidate of [option, ...keywords]) {
    const itemLower = candidate.toLowerCase()
    const slash = itemLower.lastIndexOf('/')

    if (slash !== -1 && itemLower.slice(slash + 1).includes(lower)) {
      score = Math.max(score, 2)
    } else if (itemLower.includes(lower)) {
      score = Math.max(score, 1)
    }
  }

  return score
}

/**
 * Searchable select for large option lists (e.g. ~590 IANA timezones).
 * Built on Popover + cmdk Command — the same stack as Shadcn's Combobox.
 *
 * The trigger renders like the existing closed `<Select>` but opens into a
 * searchable Command palette. Closed-world only: the user must pick from the
 * list; arbitrary text entry is not supported.
 *
 * `ConfigField` routes here when `schema.searchable === true`.
 */
export function SearchableSelect({
  value,
  onChange,
  options,
  placeholder = 'Search…',
  emptyMessage = 'No results found.',
  clearLabel,
  disabled = false,
  id,
  invalid = false,
  required = false
}: {
  value: string
  onChange: (value: string) => void
  options: SearchableSelectOptionInput[]
  placeholder?: string
  emptyMessage?: string
  /** When set, prepends a "clear" item that sets the value to ''.
   *  Matches the existing <Select> pattern of EMPTY_SELECT_VALUE + "(none)". */
  clearLabel?: string
  disabled?: boolean
  id?: string
  invalid?: boolean
  required?: boolean
}) {
  const [open, setOpen] = useState(false)
  const triggerRef = useRef<HTMLButtonElement>(null)

  const handleSelect = useCallback(
    (selected: string) => {
      onChange(selected)
      setOpen(false)
    },
    [onChange]
  )

  const normalizedOptions = options.map(option =>
    typeof option === 'string' ? { label: option, value: option } : option
  )

  const selectedOption = normalizedOptions.find(option => option.value === value)
  const displayValue = value !== '' && value !== undefined ? (selectedOption?.label ?? value) : placeholder

  return (
    <Popover onOpenChange={setOpen} open={open}>
      <PopoverTrigger asChild>
        <button
          aria-expanded={open}
          aria-haspopup="listbox"
          aria-invalid={invalid || undefined}
          aria-required={required || undefined}
          className={cn(
            controlVariants(),
            'flex items-center justify-between gap-2 whitespace-nowrap',
            !value && 'text-muted-foreground'
          )}
          data-slot="searchable-select-trigger"
          disabled={disabled}
          id={id}
          ref={triggerRef}
          role="combobox"
          type="button"
        >
          <span className="truncate">{displayValue}</span>
          <Codicon className="shrink-0 opacity-60" name={open ? 'chevron-up' : 'chevron-down'} size="1rem" />
        </button>
      </PopoverTrigger>
      <PopoverContent
        align="start"
        className="max-h-[var(--radix-popover-content-available-height)] w-[var(--radix-popover-trigger-width)] overflow-hidden p-0"
      >
        <Command filter={rankSearchOption}>
          <CommandInput autoFocus placeholder={placeholder} />
          <CommandList className="max-h-64 overscroll-contain">
            <CommandEmpty>{emptyMessage}</CommandEmpty>
            <CommandGroup>
              {clearLabel && (
                <CommandItem onSelect={() => handleSelect('')} value={clearLabel}>
                  <Codicon className={cn('mr-2 size-4', value === '' ? 'opacity-100' : 'opacity-0')} name="check" />
                  {clearLabel}
                </CommandItem>
              )}
              {normalizedOptions.map(option => (
                <CommandItem
                  key={option.value}
                  keywords={[option.label]}
                  onSelect={() => handleSelect(option.value)}
                  value={option.value}
                >
                  <Codicon
                    className={cn('mr-2 size-4', option.value === value ? 'opacity-100' : 'opacity-0')}
                    name="check"
                  />
                  {option.label}
                </CommandItem>
              ))}
            </CommandGroup>
          </CommandList>
        </Command>
      </PopoverContent>
    </Popover>
  )
}
