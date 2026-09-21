import { type ReactNode, useCallback, useState } from 'react'

import { Codicon } from '@/components/ui/codicon'
import { Command, CommandEmpty, CommandGroup, CommandInput, CommandItem, CommandList } from '@/components/ui/command'
import { controlVariants } from '@/components/ui/control'
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover'
import { cn } from '@/lib/utils'

/** One selectable row. Plain strings behave exactly as `{ value }`. */
export interface SearchableSelectOption {
  /** Extra search haystack beyond the label (provider names, model aliases). */
  keywords?: string[]
  /** What the row and the closed trigger show. Defaults to `value`. */
  label?: string
  /** The value emitted by onChange — a slug or id, never a label. */
  value: string
}

/** A labelled block of options; a missing label renders no heading. */
export interface SearchableSelectGroup {
  label?: ReactNode
  options: SearchableSelectOption[]
}

interface NormalizedOption {
  keywords: string[]
  label: string
  value: string
}

function normalizeOption(option: SearchableSelectOption | string): NormalizedOption {
  if (typeof option === 'string') {
    return { keywords: [], label: option, value: option }
  }

  return { keywords: option.keywords ?? [], label: option.label ?? option.value, value: option.value }
}

/**
 * Searchable select for large option lists (model ids, providers, ~590 IANA
 * timezones). Built on Popover + cmdk Command — the same stack as Shadcn's
 * Combobox, and the cmdk engine that already backs the composer model picker,
 * so filtering behaves identically across surfaces.
 *
 * The trigger renders like the closed `<Select>` it replaces but opens into a
 * searchable palette. Closed-world only: the user must pick from the list;
 * arbitrary text entry is not supported (Settings' `ComboboxInput` covers the
 * open-world case).
 *
 * Options are plain strings or rich `{ value, label?, keywords? }` rows, so a
 * provider can display its name while binding its slug, and models can carry
 * alias keywords (typing `kimi` finds `k3`) through cmdk's own filter — the
 * item haystack is value + label + keywords. Grouped options render under
 * headings; cmdk hides a heading whose rows all filter away.
 *
 * Generalizes the settings-local `SearchableSelect` (which grew out of the
 * timezone picker) with groups, rich options, and an out-of-catalog value
 * fallback; both currently share the same Popover+cmdk shape.
 */
export function SearchableSelect({
  ariaLabel,
  className,
  clearLabel,
  emptyMessage = 'No results found.',
  groups,
  onChange,
  options,
  placeholder = 'Search…',
  value
}: {
  ariaLabel?: string
  className?: string
  /** When set, prepends a "clear" item that selects the empty string. */
  clearLabel?: string
  emptyMessage?: string
  /** Grouped rendering; mutually exclusive with `options`. */
  groups?: SearchableSelectGroup[]
  onChange: (value: string) => void
  /** Flat list rendering. */
  options?: Array<SearchableSelectOption | string>
  placeholder?: string
  value: string
}) {
  const [open, setOpen] = useState(false)

  const normalizedGroups: Array<{ label?: ReactNode; options: NormalizedOption[] }> = (
    groups ?? [{ options: options ?? [] } as SearchableSelectGroup]
  ).map(group => ({ label: group.label, options: group.options.map(normalizeOption) }))

  const flatOptions = normalizedGroups.flatMap(group => group.options)
  // The closed trigger shows the selected row's label; an out-of-catalog value
  // (a custom slug, hand-edited config) falls back to the raw string so the
  // control never renders blank.
  const selected = flatOptions.find(option => option.value === value)
  const displayValue = selected?.label ?? (value ? value : placeholder)

  const handleSelect = useCallback(
    (next: string) => {
      setOpen(false)

      // Same-value no-op: cmdk fires onSelect unconditionally while Radix
      // Select suppresses re-picking the current value. Callers that react to
      // change (clearing the dependent model list, autosaving) lean on the
      // Radix semantics — re-picking what is already selected must be silent.
      if (next !== value) {
        onChange(next)
      }
    },
    [onChange, value]
  )

  return (
    <Popover onOpenChange={setOpen} open={open}>
      <PopoverTrigger asChild>
        <button
          aria-expanded={open}
          aria-haspopup="listbox"
          aria-label={ariaLabel}
          className={cn(
            controlVariants(),
            'flex cursor-pointer items-center justify-between gap-2 whitespace-nowrap',
            !value && 'text-muted-foreground',
            className
          )}
          data-slot="searchable-select-trigger"
          role="combobox"
          type="button"
        >
          <span className="truncate">{displayValue}</span>
          <Codicon className="shrink-0 opacity-60" name={open ? 'chevron-up' : 'chevron-down'} size="1rem" />
        </button>
      </PopoverTrigger>
      {/* min-w, not w: the trigger shrink-wraps to its current value, so a
          width pinned to it would clip long option labels. The popover keeps
          its own width and only grows to cover a wider trigger. */}
      <PopoverContent align="start" className="min-w-(--radix-popover-trigger-width) p-0">
        <Command>
          <CommandInput autoFocus placeholder={placeholder} />
          <CommandList>
            <CommandEmpty>{emptyMessage}</CommandEmpty>
            {clearLabel && (
              <CommandGroup>
                <CommandItem onSelect={() => handleSelect('')} value={clearLabel}>
                  <Codicon className={cn('mr-2 size-4', value === '' ? 'opacity-100' : 'opacity-0')} name="check" />
                  {clearLabel}
                </CommandItem>
              </CommandGroup>
            )}
            {normalizedGroups.map((group, groupIndex) => (
              <CommandGroup heading={group.label} key={groupIndex}>
                {group.options.map(option => (
                  <CommandItem
                    key={`${groupIndex}:${option.value}`}
                    keywords={option.keywords.length > 0 ? option.keywords : undefined}
                    onSelect={() => handleSelect(option.value)}
                    value={option.value}
                  >
                    <Codicon
                      className={cn('mr-2 size-4 shrink-0', option.value === value ? 'opacity-100' : 'opacity-0')}
                      name="check"
                    />
                    <span className="min-w-0 truncate">{option.label}</span>
                  </CommandItem>
                ))}
              </CommandGroup>
            ))}
          </CommandList>
        </Command>
      </PopoverContent>
    </Popover>
  )
}
