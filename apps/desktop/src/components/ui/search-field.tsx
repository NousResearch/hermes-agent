import { type ReactNode, type RefObject, useState } from 'react'

import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { Tip } from '@/components/ui/tooltip'
import { useI18n } from '@/i18n'
import { Loader2, Search } from '@/lib/icons'
import { cn } from '@/lib/utils'

interface SearchFieldProps {
  placeholder: string
  value: string
  onChange: (value: string) => void
  /**
   * Data-driven placeholder suggestions ("Try \u201ccreative\u201d") — one is picked at
   * random per mount, the nudge that search understands more than names.
   * Falls back to `placeholder` when absent/empty.
   */
  hints?: string[]
  containerClassName?: string
  inputClassName?: string
  loading?: boolean
  onClear?: () => void
  inputRef?: RefObject<HTMLInputElement | null>
  trailingAction?: ReactNode
  'aria-label'?: string
}

/**
 * Shared search field used everywhere (sessions sidebar, pages, overlays,
 * command center, cron). Quiet at rest, then gains the shared rounded hover and
 * focus surface. Width/placement come from `containerClassName`.
 */
export function SearchField({
  placeholder,
  value,
  onChange,
  hints,
  containerClassName,
  inputClassName,
  loading = false,
  onClear,
  inputRef,
  trailingAction,
  'aria-label': ariaLabel
}: SearchFieldProps) {
  const { t } = useI18n()
  const clear = onClear ?? (() => onChange(''))

  // One hint per mount, picked at random — fresh nudge every visit, no
  // mid-page carousel.
  const [hintIndex] = useState(() => Math.floor(Math.random() * 4096))
  const hintCount = hints?.length ?? 0
  const effectivePlaceholder = hintCount > 0 ? hints![hintIndex % hintCount] : placeholder

  return (
    <div
      className={cn(
        // min-w-0 is load-bearing: without it the content-sized input sets the
        // container's flex min-width and the field bulldozes its siblings
        // instead of shrinking to fit its context.
        'inline-flex min-w-0 max-w-full items-center gap-1.5 rounded-[var(--radius-sm)] border border-transparent px-2 transition-[background-color,border-color,box-shadow,color,opacity] duration-200 ease-[cubic-bezier(0.2,0.8,0.2,1)] hover:bg-(--chrome-action-hover) focus-within:border-(--ui-stroke-secondary) focus-within:bg-(--ui-bg-elevated) focus-within:shadow-sm',
        // Recede until the user reaches for it.
        !value && 'opacity-60 hover:opacity-100 focus-within:opacity-100',
        containerClassName
      )}
    >
      <Search className="pointer-events-none size-3.5 shrink-0 text-muted-foreground/70" />
      <input
        aria-label={ariaLabel ?? placeholder}
        className={cn(
          // `field-sizing: content` grows the input to fit the placeholder/typed
          // text; min-w-0 lets it shrink back below content size when the
          // context is narrower — long queries scroll inside the field.
          // Search stays compact inside headers while preserving readable contrast.
          'h-7 min-w-0 max-w-full bg-transparent text-xs text-foreground [field-sizing:content] placeholder:text-muted-foreground focus:outline-none',
          inputClassName
        )}
        onChange={event => onChange(event.target.value)}
        placeholder={effectivePlaceholder}
        ref={inputRef}
        type="text"
        value={value}
      />
      {trailingAction}
      {loading ? (
        <Loader2 className="pointer-events-none size-3.5 shrink-0 animate-spin text-muted-foreground/70" />
      ) : value ? (
        <Tip label={t.ui.search.clear}>
          <Button
            aria-label={t.ui.search.clear}
            className="shrink-0 text-muted-foreground/85 hover:bg-accent/60 hover:text-foreground"
            onClick={clear}
            size="icon-xs"
            variant="ghost"
          >
            <Codicon name="close" size="0.875rem" />
          </Button>
        </Tip>
      ) : null}
    </div>
  )
}
