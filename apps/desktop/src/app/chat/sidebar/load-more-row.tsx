import { Codicon } from '@/components/ui/codicon'
import { GlyphSpinner } from '@/components/ui/glyph-spinner'
import { Tip } from '@/components/ui/tooltip'
import { useI18n } from '@/i18n'

interface SidebarLoadMoreRowProps {
  step: number
  onClick: () => void
  loading?: boolean
  variant?: 'icon' | 'row'
}

// Compact "load more" affordance shared by recents, messaging, and cron. Kept
// intentionally identical to workspace "show more" controls (ellipsis button)
// so pagination reads as one interaction everywhere. It hangs off the list
// instead of sitting in a row, so it repeats the row's trailing inset
// (SidebarRowShell's `pr-2`) to stay on the edge the rows stop at.
export function SidebarLoadMoreRow({ step, onClick, loading = false, variant = 'icon' }: SidebarLoadMoreRowProps) {
  const { t } = useI18n()
  const label = loading ? t.sidebar.loading : step > 0 ? t.sidebar.loadCount(step) : t.sidebar.loadMore
  // Row variant appends a trailing ellipsis as a "more" affordance. Strip any
  // ellipsis a locale already puts at the end of its string (ASCII `.`, the
  // horizontal ellipsis `…`, or the two-dot leader `‥`) so we never render
  // doubled punctuation. The ellipsis is UI chrome, not translator-owned text,
  // so it stays in code rather than in the i18n strings.
  const visibleLabel =
    variant === 'row' && !loading ? `${label.replace(/[.\u2026\u2025]+$/u, '')}…` : label

  return (
    <Tip label={visibleLabel}>
      <button
        aria-label={visibleLabel}
        className={
          variant === 'row'
            ? 'w-full rounded-md px-1.5 py-1 text-left text-[0.6875rem] text-(--ui-text-tertiary) transition-colors hover:bg-(--ui-control-hover-background) hover:text-foreground disabled:cursor-default disabled:opacity-60 disabled:hover:bg-transparent disabled:hover:text-(--ui-text-tertiary)'
            : 'mr-2 ml-auto grid size-5 place-items-center rounded-sm bg-transparent text-(--ui-text-tertiary) transition-colors hover:bg-(--ui-control-hover-background) hover:text-foreground disabled:cursor-default disabled:opacity-60 disabled:hover:bg-transparent disabled:hover:text-(--ui-text-tertiary)'
        }
        disabled={loading}
        onClick={onClick}
        type="button"
      >
        {loading ? (
          <GlyphSpinner ariaLabel={label} className="text-[0.75rem]" />
        ) : variant === 'row' ? (
          visibleLabel
        ) : (
          <Codicon name="ellipsis" size="0.75rem" />
        )}
      </button>
    </Tip>
  )
}
