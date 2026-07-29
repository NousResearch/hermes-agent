import { useRef, useState } from 'react'

import { Codicon } from '@/components/ui/codicon'
import { Command, CommandGroup, CommandItem, CommandList } from '@/components/ui/command'
import { Input } from '@/components/ui/input'
import { Popover, PopoverAnchor, PopoverContent } from '@/components/ui/popover'
import { cn } from '@/lib/utils'

/**
 * Free-input combobox for open-world fields (voice/model names): a plain
 * Input the user can type anything into, plus a dropdown listing ALL known
 * options.
 *
 * Replaces the old `<Input list="…">` + `<datalist>` rendering for
 * FREE_INPUT_KEYS: native datalists filter by the field's current value, so a
 * field already holding a valid option (e.g. `gpt-4o-mini-tts`) suggested
 * only that one entry — users couldn't discover the other models/voices at
 * all (and on some platforms datalists barely render). Suggestions filter by
 * substring while typing, but an exact-match value shows the full list so an
 * already-configured field still exposes every alternative.
 */
export function ComboboxInput({
  value,
  onChange,
  options,
  optionLabels,
  placeholder,
  className
}: {
  value: string
  onChange: (value: string) => void
  options: string[]
  optionLabels?: Record<string, string>
  placeholder?: string
  className?: string
}) {
  const [open, setOpen] = useState(false)
  const inputRef = useRef<HTMLInputElement>(null)

  const query = value.trim().toLowerCase()
  const isExact = options.some(option => option.toLowerCase() === query)

  const visible =
    query && !isExact ? options.filter(option => option.toLowerCase().includes(query)) : options

  return (
    <Popover onOpenChange={setOpen} open={open}>
      <PopoverAnchor asChild>
        <div className={cn('relative', className)}>
          <Input
            className="w-full pr-7"
            onChange={e => {
              onChange(e.target.value)

              if (!open) {
                setOpen(true)
              }
            }}
            onFocus={() => setOpen(true)}
            onKeyDown={e => {
              if (e.key === 'Escape' || e.key === 'Enter' || e.key === 'Tab') {
                setOpen(false)
              }
            }}
            placeholder={placeholder}
            ref={inputRef}
            value={value}
          />
          <button
            aria-label="Show options"
            className="absolute inset-y-0 right-1.5 flex items-center text-muted-foreground"
            onClick={() => {
              setOpen(current => !current)
              inputRef.current?.focus()
            }}
            tabIndex={-1}
            type="button"
          >
            <Codicon name={open ? 'chevron-up' : 'chevron-down'} size="1rem" />
          </button>
        </div>
      </PopoverAnchor>
      <PopoverContent
        align="start"
        className="w-[var(--radix-popover-trigger-width)] p-0"
        onOpenAutoFocus={e => e.preventDefault()}
      >
        <Command shouldFilter={false}>
          <CommandList>
            {visible.length > 0 && (
              <CommandGroup>
                {visible.map(option => (
                  <CommandItem
                    key={option}
                    onSelect={() => {
                      onChange(option)
                      setOpen(false)
                    }}
                    value={option}
                  >
                    <Codicon className={cn('mr-2 size-4', option === value ? 'opacity-100' : 'opacity-0')} name="check" />
                    <span className="truncate">{optionLabels?.[option] ?? option}</span>
                  </CommandItem>
                ))}
              </CommandGroup>
            )}
          </CommandList>
        </Command>
      </PopoverContent>
    </Popover>
  )
}
