import type { ReactNode, RefObject } from 'react'

import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { Input } from '@/components/ui/input'
import { Loader2, Search } from '@/lib/icons'
import { cn } from '@/lib/utils'

interface OverlaySearchInputProps {
  placeholder: string
  value: string
  onChange: (value: string) => void
  containerClassName?: string
  inputClassName?: string
  loading?: boolean
  onClear?: () => void
  inputRef?: RefObject<HTMLInputElement | null>
  trailingAction?: ReactNode
}

export function OverlaySearchInput({
  placeholder,
  value,
  onChange,
  containerClassName,
  inputClassName,
  loading = false,
  onClear,
  inputRef,
  trailingAction
}: OverlaySearchInputProps) {
  const clear = onClear ?? (() => onChange(''))
  const hasTrailing = Boolean(trailingAction)

  return (
    <div className={cn('relative', containerClassName)}>
      <Search className="pointer-events-none absolute left-3 top-1/2 z-1 size-3.5 -translate-y-1/2 text-muted-foreground/80" />
      <Input
        className={cn(
          'relative z-0 py-2 pl-8 text-[length:var(--conversation-text-font-size)]',
          hasTrailing || loading || value ? 'pr-16' : 'pr-8',
          inputClassName
        )}
        onChange={event => onChange(event.target.value)}
        placeholder={placeholder}
        ref={inputRef}
        value={value}
      />
      <div className="absolute right-1.5 top-1/2 z-1 flex -translate-y-1/2 items-center gap-0.5">
        {trailingAction}
        {loading ? (
          <Loader2 className="pointer-events-none size-3.5 animate-spin text-muted-foreground/70" />
        ) : value ? (
          <Button
            aria-label="Clear search"
            className="text-muted-foreground/85 hover:bg-accent/60 hover:text-foreground"
            onClick={clear}
            size="icon-xs"
            variant="ghost"
          >
            <Codicon name="close" size="0.875rem" />
          </Button>
        ) : null}
      </div>
    </div>
  )
}

export function PageSearchInput(props: OverlaySearchInputProps) {
  return (
    <OverlaySearchInput
      {...props}
      containerClassName={cn('mx-auto w-[min(36rem,calc(100%-2rem))] min-w-0', props.containerClassName)}
      inputClassName={cn('py-2 pl-8', props.inputClassName)}
    />
  )
}
