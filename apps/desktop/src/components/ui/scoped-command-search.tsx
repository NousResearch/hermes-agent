import { useEffect, useLayoutEffect, useRef, useState } from 'react'

import { Command, CommandEmpty, CommandGroup, CommandInput, CommandItem, CommandList } from '@/components/ui/command'
import { ChevronRight } from '@/lib/icons'

export interface ScopedCommandSearchItem {
  description?: string
  id: string
  keywords?: string[]
  label: string
}

/**
 * Embeddable cmdk search whose caller owns the catalog and destination.
 * Passing only the current pane's items keeps results scoped without teaching
 * this component about settings, credentials, or routes.
 */
export function ScopedCommandSearch({
  emptyMessage,
  items,
  onSelect,
  onValueChange,
  placeholder,
  value
}: ScopedCommandSearchProps) {
  const inputRef = useRef<HTMLInputElement>(null)
  const rootRef = useRef<HTMLDivElement>(null)
  const [dismissed, setDismissed] = useState(false)
  const open = Boolean(value.trim()) && !dismissed

  useLayoutEffect(() => {
    inputRef.current?.setAttribute('aria-expanded', String(open))
  }, [open])

  useEffect(() => {
    if (!open) {
      return
    }

    const dismissOutside = (event: PointerEvent) => {
      if (!rootRef.current?.contains(event.target as Node)) {
        setDismissed(true)
      }
    }

    globalThis.document.addEventListener('pointerdown', dismissOutside)

    return () => globalThis.document.removeEventListener('pointerdown', dismissOutside)
  }, [open])

  const updateValue = (next: string) => {
    setDismissed(false)
    onValueChange(next)
  }

  return (
    <Command
      className="relative h-auto overflow-visible rounded-none bg-transparent"
      label={placeholder}
      loop
      ref={rootRef}
    >
      <CommandInput
        aria-expanded={open}
        aria-label={placeholder}
        className="text-xs"
        onBlur={() => {
          globalThis.requestAnimationFrame(() => {
            if (!rootRef.current?.contains(globalThis.document.activeElement)) {
              setDismissed(true)
            }
          })
        }}
        onFocus={() => setDismissed(false)}
        onKeyDown={event => {
          if (event.key === 'Escape' && open) {
            event.preventDefault()
            event.stopPropagation()
            updateValue('')
          }
        }}
        onValueChange={updateValue}
        placeholder={placeholder}
        ref={inputRef}
        value={value}
      />

      <CommandList
        className="dt-portal-scrollbar absolute inset-x-0 top-full z-30 mt-1 max-h-64 rounded-[6px] border border-border bg-popover p-1 shadow-nous"
        hidden={!open}
      >
        <CommandEmpty>{emptyMessage}</CommandEmpty>
        <CommandGroup className="p-0">
          {open &&
            items.map(item => (
              <CommandItem
                className="group grid grid-cols-[minmax(0,1fr)_auto] gap-3 px-3 py-2.5"
                key={item.id}
                keywords={item.keywords}
                onSelect={() => onSelect(item)}
                value={`${item.label}\u0001${item.id}`}
              >
                <span className="min-w-0">
                  <span className="block truncate font-medium">{item.label}</span>
                  <span className="mt-0.5 block truncate text-xs text-muted-foreground">
                    {item.description || item.id}
                  </span>
                </span>
                <ChevronRight className="size-4 self-center text-muted-foreground opacity-0 transition-opacity group-data-[selected=true]:opacity-100" />
              </CommandItem>
            ))}
        </CommandGroup>
      </CommandList>
    </Command>
  )
}

interface ScopedCommandSearchProps {
  emptyMessage: string
  items: ScopedCommandSearchItem[]
  onSelect: (item: ScopedCommandSearchItem) => void
  onValueChange: (value: string) => void
  placeholder: string
  value: string
}
