import { useCallback, useRef, useState } from 'react'

import { Command, CommandEmpty, CommandGroup, CommandInput, CommandItem, CommandList } from '@/components/ui/command'
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover'
import { Codicon } from '@/components/ui/codicon'
import { controlVariants } from '@/components/ui/control'
import { cn } from '@/lib/utils'

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
  emptyMessage = 'No results found.'
}: {
  value: string
  onChange: (value: string) => void
  options: string[]
  placeholder?: string
  emptyMessage?: string
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
          aria-expanded={open}
        >
          <span className="truncate">{displayValue}</span>
          <Codicon className="shrink-0 opacity-60" name={open ? 'chevron-up' : 'chevron-down'} size="1rem" />
        </button>
      </PopoverTrigger>
      <PopoverContent
        align="start"
        className="w-[var(--radix-popover-trigger-width)] p-0"
      >
        <Command
          filter={(value, search) => {
            // cmdk's default filter is case-insensitive substring match on
            // the item's value. For timezone-like values we also want to
            // match segments after "/" so "york" matches "America/New_York".
            const lower = search.toLowerCase()
            const itemLower = value.toLowerCase()
            // Prioritize city/region segment (after last "/") so "york" ranks
            // "America/New_York" above "America/New_York/Special".
            const slash = itemLower.lastIndexOf('/')
            if (slash !== -1 && itemLower.slice(slash + 1).includes(lower)) return 2
            if (itemLower.includes(lower)) return 1
            return 0
          }}
        >
          <CommandInput autoFocus placeholder={placeholder} />
          <CommandList>
            <CommandEmpty>{emptyMessage}</CommandEmpty>
            <CommandGroup>
              {options.map(option => (
                <CommandItem
                  key={option}
                  onSelect={() => handleSelect(option)}
                  value={option}
                >
                  <Codicon
                    className={cn('mr-2 size-4', option === value ? 'opacity-100' : 'opacity-0')}
                    name="check"
                  />
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
