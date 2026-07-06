import { Codicon } from '@/components/ui/codicon'
import { GlyphSpinner } from '@/components/ui/glyph-spinner'
import { useI18n } from '@/i18n'

interface SidebarLoadMoreRowProps {
  step: number
  onClick: () => void
  loading?: boolean
}

// Compact "load more" affordance shared by recents, messaging, and cron. Kept
// intentionally identical to workspace "show more" controls (ellipsis button)
// so pagination reads as one interaction everywhere.
export function SidebarLoadMoreRow({ step, onClick, loading = false }: SidebarLoadMoreRowProps) {
  const { t } = useI18n()
  const label = loading ? t.sidebar.loading : step > 0 ? t.sidebar.loadCount(step) : t.sidebar.loadMore

  return (
    <button
      aria-label={label}
      className="ml-auto grid size-5 place-items-center rounded-sm bg-transparent text-(--ui-text-tertiary) transition-colors hover:bg-(--ui-control-hover-background) hover:text-foreground disabled:cursor-default disabled:opacity-60 disabled:hover:bg-transparent disabled:hover:text-(--ui-text-tertiary)"
      disabled={loading}
      onClick={onClick}
      type="button"
    >
      {loading ? (
        <GlyphSpinner ariaLabel={label} className="text-[0.75rem]" />
      ) : (
        <Codicon name="ellipsis" size="0.75rem" />
      )}
    </button>
  )
}
