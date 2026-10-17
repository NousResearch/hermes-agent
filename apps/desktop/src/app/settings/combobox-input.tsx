import { type ReactNode, useId, useRef, useState } from 'react'

import { Codicon } from '@/components/ui/codicon'
import {
  Command,
  CommandEmpty,
  CommandInput,
  CommandItem,
  CommandItemCheck,
  CommandList
} from '@/components/ui/command'
import { Input } from '@/components/ui/input'
import { Popover, PopoverAnchor, PopoverContent } from '@/components/ui/popover'
import { useI18n } from '@/i18n'
import { isSubmitEnter } from '@/lib/ime'
import { cn } from '@/lib/utils'

import { rankSearchOption } from './searchable-select'

// cmdk highlights its first row whenever nothing is selected, which would make
// Enter swap free text for that row. A value no row carries keeps the list
// unhighlighted until the user actually arrows or points into it.
const IDLE = '\u0000idle'

/**
 * Free-input combobox for open-world fields (voice/model/font names): a plain
 * Input the user can type anything into, plus the app's menu listing ALL known
 * options. This is the only suggestion list — never `<Input list>` +
 * `<datalist>`: Chromium paints that as its own grey OS popup, out of step with
 * every other menu, and it filters by the field's current value, so a field
 * already holding a valid option (e.g. `gpt-4o-mini-tts`) suggested only that
 * one entry. Suggestions filter by substring while typing, but an exact-match
 * value shows the full list so an already-configured field still exposes
 * every alternative.
 */
export function ComboboxInput({
  'aria-label': ariaLabel,
  className,
  disabled,
  onChange,
  optionLabels,
  options,
  placeholder,
  renderOption,
  search,
  value
}: {
  'aria-label'?: string
  className?: string
  disabled?: boolean
  onChange: (value: string) => void
  optionLabels?: Record<string, string>
  options: readonly string[]
  placeholder?: string
  /** Custom row content, e.g. a font name set in its own face. */
  renderOption?: (option: string) => ReactNode
  /** Optional menu search keeps filtering separate from the editable value. */
  search?: { placeholder: string; emptyMessage: string }
  value: string
}) {
  const { t } = useI18n()
  const [open, setOpen] = useState(false)
  const [active, setActive] = useState(IDLE)
  const anchorRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLInputElement>(null)
  const commandRef = useRef<HTMLDivElement>(null)
  const listId = useId()

  const query = value.trim().toLowerCase()
  const isExact = options.some(option => option.toLowerCase() === query)

  const visible =
    query && !search && !isExact ? options.filter(option => option.toLowerCase().includes(query)) : options
  // An empty menu is just a floating border; stay closed until something matches.
  const expanded = open && !disabled && visible.length > 0

  const openList = (next: boolean) => {
    setOpen(next)
    setActive(IDLE)
  }

  // Focus stays in the Input, outside cmdk's root, so hand it the keys it
  // navigates with.
  const forwardKey = (key: string) =>
    commandRef.current?.dispatchEvent(new KeyboardEvent('keydown', { bubbles: true, cancelable: true, key }))

  return (
    <Popover onOpenChange={openList} open={expanded}>
      <PopoverAnchor asChild>
        <div className={cn('relative', className)} ref={anchorRef}>
          <Input
            aria-autocomplete="list"
            aria-controls={expanded ? listId : undefined}
            aria-expanded={expanded}
            aria-label={ariaLabel}
            className="w-full pr-7"
            disabled={disabled}
            onChange={e => {
              onChange(e.target.value)
              if (!search) openList(true)
            }}
            onFocus={() => {
              if (!search) openList(true)
            }}
            onKeyDown={e => {
              if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
                e.preventDefault()

                if (expanded) {
                  forwardKey(e.key)
                } else {
                  openList(true)
                }

                return
              }

              if (isSubmitEnter(e) && expanded && active !== IDLE) {
                e.preventDefault()
                forwardKey('Enter')

                return
              }

              if (e.key === 'Escape' || e.key === 'Tab' || isSubmitEnter(e)) {
                openList(false)
              }
            }}
            placeholder={placeholder}
            ref={inputRef}
            role="combobox"
            value={value}
          />
          <button
            aria-label={t.settings.config.showOptions}
            className="absolute inset-y-0 right-1.5 flex items-center text-(--ui-text-tertiary) disabled:opacity-50"
            disabled={disabled}
            onClick={() => {
              openList(!expanded)
              inputRef.current?.focus()
            }}
            tabIndex={-1}
            type="button"
          >
            <Codicon name={expanded ? 'chevron-up' : 'chevron-down'} size="1rem" />
          </button>
        </div>
      </PopoverAnchor>
      <PopoverContent
        align="start"
        className="min-w-(--radix-popover-trigger-width)"
        // The field and its chevron own open/close; a press there isn't "outside".
        onInteractOutside={e => {
          if (anchorRef.current?.contains(e.target as Node)) {
            e.preventDefault()
          }
        }}
        onOpenAutoFocus={e => {
          if (!search) e.preventDefault()
        }}
        variant="menu"
      >
        <Command
          filter={rankSearchOption}
          onValueChange={setActive}
          ref={commandRef}
          shouldFilter={Boolean(search)}
          value={active}
          variant="menu"
        >
          {search && <CommandInput autoFocus placeholder={search.placeholder} />}
          <CommandList id={listId}>
            {search && <CommandEmpty>{search.emptyMessage}</CommandEmpty>}
            {visible.map(option => (
              <CommandItem
                key={option}
                onSelect={() => {
                  onChange(option)
                  openList(false)
                }}
                value={option}
              >
                <span className="min-w-0 truncate">{renderOption?.(option) ?? optionLabels?.[option] ?? option}</span>
                <CommandItemCheck checked={option === value} />
              </CommandItem>
            ))}
          </CommandList>
        </Command>
      </PopoverContent>
    </Popover>
  )
}
