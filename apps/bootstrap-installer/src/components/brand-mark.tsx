import { cn } from '../lib/utils'

// Brand badge: a simple "◆" glyph on a neutral tile, identical in light/dark.
// Ported from apps/desktop's BrandMark.
// TODO(rebrand): replace with the final HT AI Agent logo asset when available.
export function BrandMark({ className, ...props }: React.ComponentProps<'span'>) {
  return (
    <span
      aria-hidden
      className={cn(
        'inline-flex size-14 shrink-0 select-none items-center justify-center bg-neutral-900 text-white dark:bg-white dark:text-neutral-900',
        className
      )}
      {...props}
    >
      <span className="text-[60%] leading-none">◆</span>
    </span>
  )
}
