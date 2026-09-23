import { useRef, useState } from 'react'

import { Codicon } from '@/components/ui/codicon'
import { Command, CommandEmpty, CommandGroup, CommandInput, CommandItem, CommandList } from '@/components/ui/command'
import { controlVariants } from '@/components/ui/control'
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover'
import { useI18n } from '@/i18n'
import { cn } from '@/lib/utils'

import { categoryLabel, type CountedValue, UNCATEGORISED } from './derive-tools'

export interface CategoryPickerProps {
  categories: CountedValue[]
  onChange: (category: null | string) => void
  value: null | string
}

export function CategoryPicker({ categories, onChange, value }: CategoryPickerProps) {
  const { t } = useI18n()
  const copy = t.connectorsPage
  const labelFor = (name: string) => (name === UNCATEGORISED ? copy.uncategorised : categoryLabel(name))
  const selected = value === null ? null : categories.find(entry => entry.value === value)
  const [open, setOpen] = useState(false)
  const trigger = useRef<HTMLButtonElement>(null)

  const pick = (next: null | string) => {
    onChange(next)
    setOpen(false)
    trigger.current?.focus()
  }

  return (
    <Popover onOpenChange={setOpen} open={open}>
      <PopoverTrigger asChild>
        <button
          aria-expanded={open}
          aria-haspopup="listbox"
          className={cn(controlVariants({ size: 'xs' }), 'flex w-auto items-center gap-1.5 whitespace-nowrap')}
          ref={trigger}
          type="button"
        >
          <span className="truncate">
            {selected ? labelFor(selected.value) : copy.tools.categorySelect(categories.length)}
          </span>
          <Codicon className="shrink-0 opacity-60" name="chevron-down" size="0.875rem" />
        </button>
      </PopoverTrigger>
      <PopoverContent align="start" className="min-w-(--radix-popover-trigger-width) p-0">
        <Command>
          <CommandInput autoFocus placeholder={copy.filterCategory} />
          <CommandList>
            <CommandEmpty>{copy.tools.noMatch}</CommandEmpty>
            <CommandGroup>
              <CommandItem onSelect={() => pick(null)} value={copy.categoryAll}>
                <Codicon className={cn('mr-2 size-4', value === null ? 'opacity-100' : 'opacity-0')} name="check" />
                {copy.categoryAll}
              </CommandItem>
              {categories.map(entry => (
                <CommandItem key={entry.value} onSelect={() => pick(entry.value)} value={labelFor(entry.value)}>
                  <Codicon
                    className={cn('mr-2 size-4', entry.value === value ? 'opacity-100' : 'opacity-0')}
                    name="check"
                  />
                  <span className="min-w-0 flex-1 truncate">{labelFor(entry.value)}</span>
                  <span className="ml-2 shrink-0 tabular-nums text-(--ui-text-tertiary)">{entry.count}</span>
                </CommandItem>
              ))}
            </CommandGroup>
          </CommandList>
        </Command>
      </PopoverContent>
    </Popover>
  )
}
