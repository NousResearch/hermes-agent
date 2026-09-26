/**
 * The dashboard's right-hand rail: one panel, three views of the same real
 * session — the raw activity log, the measured statistics, and the digest.
 *
 * The panel keeps its previous a11y contract exactly: a `complementary`
 * landmark on desktop, a labelled `dialog` when it floats as a drawer or a
 * bottom sheet, with the close button and Escape handling owned by the
 * dashboard. Adding views did not change how it is reached.
 */

import type { RefObject } from 'react'

import { Button } from '@/components/ui/button'
import { SegmentedControl } from '@/components/ui/segmented-control'
import { Activity, BarChart3, Bell, X } from '@/lib/icons'
import { cn } from '@/lib/utils'

import { JarvisActivityList } from './activity-panel'
import type { JarvisMetrics } from './metrics'
import type { JarvisNewsItem } from './news'
import { JarvisNewsView } from './news-view'
import type { JarvisInsightsTabCopy, JarvisInsightsView, JarvisNewsViewCopy, JarvisStatsCopy } from './panel-copy'
import { JarvisStatsView } from './stats-view'
import type { JarvisEvent } from './types'

export interface JarvisInsightsPanelProps {
  className?: string
  closeButtonRef?: RefObject<HTMLButtonElement | null>
  copy: {
    activity: { close: string; empty: string; title: string; types: Record<string, string> }
    news: JarvisNewsViewCopy
    stats: JarvisStatsCopy
    tabs: JarvisInsightsTabCopy
    viewsLabel: string
  }
  events: readonly JarvisEvent[]
  id?: string
  labelledBy?: string
  metrics: JarvisMetrics
  news: readonly JarvisNewsItem[]
  onClose?: () => void
  onOpenUpdate?: (target: 'backend' | 'client') => void
  onViewChange: (view: JarvisInsightsView) => void
  surface?: 'bottom-sheet' | 'drawer' | 'panel'
  view: JarvisInsightsView
}

export function JarvisInsightsPanel({
  className,
  closeButtonRef,
  copy,
  events,
  id,
  labelledBy,
  metrics,
  news,
  onClose,
  onOpenUpdate,
  onViewChange,
  surface = 'panel',
  view
}: JarvisInsightsPanelProps) {
  const overlay = surface !== 'panel'
  const unread = news.filter(item => item.tone === 'warn').length

  return (
    <aside
      aria-labelledby={labelledBy}
      className={cn(
        'jarvis-glass-strong flex min-h-0 flex-col text-(--ui-text-primary)',
        surface === 'panel' && 'border-l border-(--ui-stroke-tertiary)',
        overlay && 'shadow-nous border border-(--stroke-nous)',
        className
      )}
      data-activity-surface={surface}
      data-insights-view={view}
      id={id}
      role={overlay ? 'dialog' : 'complementary'}
    >
      <div className="flex min-h-11 items-center gap-2 px-4 py-3">
        <Activity className="size-4 text-(--ui-accent)" />
        <h2 className="min-w-0 flex-1 text-sm font-semibold" id={labelledBy}>
          {copy.activity.title}
        </h2>
        {onClose && (
          <Button
            aria-label={copy.activity.close}
            className="min-h-11 min-w-11"
            onClick={onClose}
            ref={closeButtonRef}
            size="icon"
            type="button"
            variant="ghost"
          >
            <X />
          </Button>
        )}
      </div>

      {/* min-h-12 on the track leaves its buttons 44px once the 2px inset
          padding is taken off — the product's touch-target floor (P0
          acceptance criterion 11). */}
      <SegmentedControl<JarvisInsightsView>
        className="mx-4 mb-3 min-h-12 w-[calc(100%-2rem)]"
        onChange={onViewChange}
        options={[
          { icon: Activity, id: 'activity', label: copy.tabs.activity },
          { icon: BarChart3, id: 'stats', label: copy.tabs.stats },
          { icon: Bell, id: 'news', label: unread > 0 ? `${copy.tabs.news} (${unread})` : copy.tabs.news }
        ]}
        value={view}
      />

      <div aria-label={copy.viewsLabel} className="min-h-0 flex-1 overflow-auto px-4 pb-4" role="group">
        {view === 'activity' && <JarvisActivityList copy={copy.activity} events={events} />}
        {view === 'stats' && <JarvisStatsView copy={copy.stats} metrics={metrics} />}
        {view === 'news' && <JarvisNewsView copy={copy.news} items={news} onOpenUpdate={onOpenUpdate} />}
      </div>
    </aside>
  )
}
