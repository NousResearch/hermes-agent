import { useCallback, useRef, useState } from 'react'

import { Codicon } from '@/components/ui/codicon'
import { Command, CommandEmpty, CommandGroup, CommandInput, CommandItem, CommandList } from '@/components/ui/command'
import { controlVariants } from '@/components/ui/control'
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover'
import { cn } from '@/lib/utils'

/**
 * cmdk filter score for one option. Case-insensitive substring match, with
 * the final path segment (after the last "/") ranked above matches anywhere
 * else so "york" ranks "America/New_York" over "America/New_York/Special".
 * Exported for tests.
 */
export function rankSearchOption(option: string, search: string): number {
  const lower = search.toLowerCase()
  const itemLower = option.toLowerCase()
  const slash = itemLower.lastIndexOf('/')

  if (slash !== -1 && itemLower.slice(slash + 1).includes(lower)) {
    return 2
  }

  if (itemLower.includes(lower)) {
    return 1
  }

  return 0
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
  clearLabel
}: {
  value: string
  onChange: (value: string) => void
  options: string[]
  placeholder?: string
  emptyMessage?: string
  /** When set, prepends a "clear" item that sets the value to ''.
   *  Matches the existing <Select> pattern of EMPTY_SELECT_VALUE + "(none)". */
  clearLabel?: string
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

  const displayValue = value !== '' && value !== undefined ? value : placeholder

  return (
    <Popover onOpenChange={setOpen} open={open}>
      <PopoverTrigger asChild>
        <button
          aria-expanded={open}
          aria-haspopup="listbox"
          className={cn(
            controlVariants(),
            'flex items-center justify-between gap-2 whitespace-nowrap',
            !value && 'text-muted-foreground'
          )}
          data-slot="searchable-select-trigger"
          ref={triggerRef}
          role="combobox"
          type="button"
        >
          <span className="truncate">{displayValue}</span>
          <Codicon className="shrink-0 opacity-60" name={open ? 'chevron-up' : 'chevron-down'} size="1rem" />
        </button>
      </PopoverTrigger>
      <PopoverContent align="start" className="w-[var(--radix-popover-trigger-width)] p-0">
        <Command filter={rankSearchOption}>
          <CommandInput autoFocus placeholder={placeholder} />
          <CommandList>
            <CommandEmpty>{emptyMessage}</CommandEmpty>
            <CommandGroup>
              {clearLabel && (
                <CommandItem onSelect={() => handleSelect('')} value={clearLabel}>
                  <Codicon className={cn('mr-2 size-4', value === '' ? 'opacity-100' : 'opacity-0')} name="check" />
                  {clearLabel}
                </CommandItem>
              )}
              {options.map(option => (
                <CommandItem key={option} onSelect={() => handleSelect(option)} value={option}>
                  <Codicon className={cn('mr-2 size-4', option === value ? 'opacity-100' : 'opacity-0')} name="check" />
                  {option}
                </CommandItem>
              ))}
            </CommandGroup>
          </CommandList>
        </Command>
      </PopoverContent>
    </Popover>
  )
}
