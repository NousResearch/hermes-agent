import { Codicon } from '@/components/ui/codicon'
import { useI18n } from '@/i18n'

interface SidebarLoadMoreRowProps {
  step: number
  onClick: () => void
  loading?: boolean
}

// "Load N more" affordance shared by the recents, messaging, and cron sections.
// The chevron sits in the same w-3.5 column the rows use for their dot, so it
// lines up with the list above.
export function SidebarLoadMoreRow({ step, onClick, loading = false }: SidebarLoadMoreRowProps) {
  const { t } = useI18n()
  const label = loading ? t.sidebar.loading : step > 0 ? t.sidebar.loadCount(step) : t.sidebar.loadMore

  return (
    <button
      className="flex min-h-5 items-center gap-1.5 self-start bg-transparent pl-2 text-left text-[0.6875rem] text-(--ui-text-tertiary) transition-colors duration-100 ease-out hover:text-foreground hover:transition-none disabled:cursor-default disabled:opacity-60 disabled:hover:text-(--ui-text-tertiary)"
      disabled={loading}
      onClick={onClick}
      type="button"
    >
      <span className="grid w-3.5 shrink-0 place-items-center">
        <Codicon className="opacity-70" name={loading ? 'loading' : 'chevron-down'} size="0.75rem" spinning={loading} />
      </span>
      <span>{label}</span>
    </button>
  )
}
