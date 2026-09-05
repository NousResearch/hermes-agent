import { useCallback, useMemo, useRef, useState } from 'react'

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
 * cmdk filter for one item: the best score across the value and each keyword,
 * scored independently.
 *
 * Joining them into one haystack — `${item} ${keywords.join(' ')}` — was wrong
 * twice over. A search containing a space could match across the seam that the
 * join itself introduced, so "4 open" hit the pair ("gpt-4", "openai") that
 * neither member matches. And `rankSearchOption` locates the final path segment
 * with `lastIndexOf('/')`, which on a joined string can land inside a *keyword*:
 * ("anthropic/claude-opus-4", ["openai/gpt-4"]) measured "gpt-4" as the segment,
 * so "opus" silently dropped from rank 2 to rank 1 and lost its ordering.
 *
 * Scoring each candidate on its own also stops a keyword that repeats the value
 * from counting twice when cmdk breaks ties.
 *
 * Exported for tests.
 */
export function rankSearchCandidates(item: string, search: string, keywords?: string[]): number {
  return Math.max(rankSearchOption(item, search), ...(keywords ?? []).map(k => rankSearchOption(k, search)))
}

/** A single selectable entry. `label` overrides the raw `value` for display;
 *  `keywords` are extra search haystacks beyond the value (e.g. a model id's
 *  aliases), each scored separately by rankSearchOption. */
export interface SearchableSelectOption {
  value: string
  label?: string
  keywords?: string[]
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
  className
}: {
  value: string
  onChange: (value: string) => void
  options: readonly (string | SearchableSelectOption)[]
  placeholder?: string
  emptyMessage?: string
  /** When set, prepends a "clear" item that sets the value to ''.
   *  Matches the existing <Select> pattern of EMPTY_SELECT_VALUE + "(none)". */
  clearLabel?: string
  /** Extra classes merged onto the trigger (e.g. min-w-* sizing). */
  className?: string
}) {
  const [open, setOpen] = useState(false)
  const triggerRef = useRef<HTMLButtonElement>(null)

  const handleSelect = useCallback(
    (selected: string) => {
      // Radix's <Select> ignores re-picking the current value (no
      // onValueChange), and the converted pickers rely on that: a MoA slot's
      // provider re-selected unchanged must not schedule another autosave.
      if (selected !== value) {
        onChange(selected)
      }
      setOpen(false)
    },
    [onChange, value]
  )

  // Plain strings normalize to {value, label: value}. A selected value missing
  // from the list (e.g. a saved model the provider no longer reports) falls
  // back to the raw value so the trigger never renders as a blank box.
  // Memoized: harmless to rebuild at ~600 options, but the model pickers feed
  // this whole catalogs and every keystroke in the palette re-renders.
  const normalizedOptions: SearchableSelectOption[] = useMemo(
    () => options.map(option => (typeof option === 'string' ? { value: option, label: option } : option)),
    [options]
  )
  const selectedOption = normalizedOptions.find(option => option.value === value)
  const displayValue = !value ? placeholder : (selectedOption?.label ?? value)

  return (
    <Popover onOpenChange={setOpen} open={open}>
      <PopoverTrigger asChild>
        <button
          aria-expanded={open}
          aria-haspopup="listbox"
          className={cn(
            controlVariants(),
            'flex items-center justify-between gap-2 whitespace-nowrap',
            !value && 'text-muted-foreground',
            className
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
        <Command filter={rankSearchCandidates}>
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
              {normalizedOptions.map(option => (
                <CommandItem
                  key={option.value}
                  keywords={option.keywords}
                  onSelect={() => handleSelect(option.value)}
                  value={option.value}
                >
                  <Codicon
                    className={cn('mr-2 size-4', option.value === value ? 'opacity-100' : 'opacity-0')}
                    name="check"
                  />
                  {option.label ?? option.value}
                </CommandItem>
              ))}
            </CommandGroup>
          </CommandList>
        </Command>
      </PopoverContent>
    </Popover>
  )
}
