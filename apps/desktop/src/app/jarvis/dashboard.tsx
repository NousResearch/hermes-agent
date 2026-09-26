import { useStore } from '@nanostores/react'
import { type ReactNode, useEffect, useId, useMemo, useRef, useState } from 'react'
import { useNavigate } from 'react-router'

import { NEW_CHAT_ROUTE } from '@/app/routes'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { useI18n } from '@/i18n'
import { Activity, Maximize, Moon, Sun } from '@/lib/icons'
import { cn } from '@/lib/utils'

import { JarvisCore } from './core'
import { $jarvisFocusMode, $jarvisRailVisible, setJarvisFocusMode } from './focus-mode'
import { JarvisInsightsPanel } from './insights-panel'
import { deriveJarvisMetrics } from './metrics'
import type { JarvisNewsItem } from './news'
import type { JarvisInsightsView } from './panel-copy'
import { jarvisDaypart } from './pulse'
import { JarvisStatusStrip } from './status-strip'
import { JarvisTipsLauncher } from './tips'
import type { JarvisUiState } from './types'

type DashboardLayout = 'desktop' | 'mobile' | 'tablet'

/** Work in flight — nothing may open itself over it. */
const BUSY_PHASES = new Set<JarvisUiState['task']['phase']>(['approval', 'cancelling', 'planning', 'running'])

export interface JarvisDashboardProps {
  children: ReactNode
  className?: string
  connected: boolean
  /**
   * A fresh draft is showing the home hero, which carries its own orb and
   * greeting — the header drops both so the screen says each thing once.
   */
  home?: boolean
  layout?: DashboardLayout
  /** The digest built from real update status and real session events. */
  news?: readonly JarvisNewsItem[]
  onOpenUpdate?: (target: 'backend' | 'client') => void
  profileDisplayName?: string
  /** Cards stacked above the insights panel in the desktop rail. */
  rail?: ReactNode
  state: JarvisUiState
  voiceControls?: ReactNode
}

function greeting(
  copy: ReturnType<typeof useI18n>['t']['jarvisShell']['dashboard'],
  profileDisplayName?: string
): string {
  const name = profileDisplayName?.trim()

  return name ? copy.emptyGreeting(name) : copy.emptyGreeting()
}

function dashboardLayoutForViewport(): DashboardLayout {
  if (window.matchMedia?.('(max-width: 767px)').matches) {
    return 'mobile'
  }

  if (window.matchMedia?.('(max-width: 1149px)').matches) {
    return 'tablet'
  }

  return 'desktop'
}

function useDashboardLayout(override: DashboardLayout | undefined): DashboardLayout {
  const [layout, setLayout] = useState<DashboardLayout>(() => override ?? dashboardLayoutForViewport())

  useEffect(() => {
    if (override) {
      setLayout(override)

      return undefined
    }

    const media = [window.matchMedia?.('(max-width: 767px)'), window.matchMedia?.('(max-width: 1149px)')].filter(
      Boolean
    ) as MediaQueryList[]

    const update = () => setLayout(dashboardLayoutForViewport())

    // The override may have just been dropped, so resync before listening.
    update()

    for (const query of media) {
      query.addEventListener?.('change', update)
      query.addListener?.(update)
    }

    return () => {
      for (const query of media) {
        query.removeEventListener?.('change', update)
        query.removeListener?.(update)
      }
    }
  }, [override])

  return layout
}

/**
 * Home's top bar: today's date with a sun or moon for the part of the day on
 * the left, the tips deck and focus mode on the right.
 */
function HomeTopBar({ tips }: { tips: ReactNode }) {
  const { locale, t } = useI18n()
  const copy = t.jarvisShell.home
  const focus = useStore($jarvisFocusMode)
  const now = new Date()
  const daypart = jarvisDaypart(now)
  const DayIcon = daypart === 'evening' || daypart === 'night' ? Moon : Sun

  const date = new Intl.DateTimeFormat(locale, { day: 'numeric', month: 'short', weekday: 'short' }).format(now)

  return (
    <div className="flex w-full items-center gap-3">
      <div className="flex min-w-0 items-center gap-2 text-sm text-(--ui-text-secondary)">
        <DayIcon className="size-4 shrink-0 text-amber-400" />
        <span className="truncate first-letter:uppercase">{date}</span>
      </div>
      <div className="ml-auto flex items-center gap-2">
        {tips}
        <Button
          aria-pressed={focus}
          className={cn(
            'min-h-11 jarvis-glass jarvis-glass-hover rounded-full px-4 text-(--ui-text-primary)',
            focus && 'jarvis-nav-active'
          )}
          onClick={() => setJarvisFocusMode(!focus)}
          size="sm"
          title={copy.focusModeHint}
          type="button"
          variant="secondary"
        >
          <Maximize />
          {focus ? copy.focusModeExit : copy.focusMode}
        </Button>
      </div>
    </div>
  )
}

function ResultHeader({
  copy,
  profileDisplayName,
  state
}: {
  copy: ReturnType<typeof useI18n>['t']['jarvisShell']['dashboard']
  profileDisplayName?: string
  state: JarvisUiState
}) {
  const title = state.result?.trim()

  return (
    <header className="shrink-0 px-4 py-4 md:px-5">
      {title ? (
        // The result is the headline. Its "done" state is already on the
        // status strip above — repeating it here would say the same thing
        // twice on one screen.
        <h1 className="text-xl font-semibold leading-7 text-(--ui-text-primary)">{title}</h1>
      ) : (
        <p className="text-sm font-medium text-(--ui-text-secondary)">{greeting(copy, profileDisplayName)}</p>
      )}
    </header>
  )
}

export function JarvisDashboard({
  children,
  className,
  connected,
  home = false,
  layout: layoutOverride,
  news = [],
  onOpenUpdate,
  profileDisplayName,
  rail,
  state,
  voiceControls
}: JarvisDashboardProps) {
  const { locale, t } = useI18n()
  const navigate = useNavigate()
  const copy = t.jarvisShell.dashboard
  const layout = useDashboardLayout(layoutOverride)
  const focus = useStore($jarvisFocusMode)
  const showRail = layout === 'desktop' && !focus
  const [activityOpen, setActivityOpen] = useState(layout === 'desktop')
  const [view, setView] = useState<JarvisInsightsView>('activity')
  // The rail (desktop home cards) leaves the conversation column too narrow
  // for the full-size status orb beside the status pills.
  const compactCore = layout === 'mobile' && !BUSY_PHASES.has(state.task.phase)
  const activityPanelId = useId()
  const activityTitleId = useId()
  const activityToggleRef = useRef<HTMLButtonElement>(null)
  const activityCloseRef = useRef<HTMLButtonElement>(null)

  // Derived, not stored: the activity list is the single source of truth, so
  // the charts can never disagree with the log above them.
  const metrics = useMemo(() => deriveJarvisMetrics(state.activity), [state.activity])
  const attention = news.filter(item => item.tone === 'warn').length
  const busy = BUSY_PHASES.has(state.task.phase)
  const voiceActive = state.voice === 'listening' || state.voice === 'speaking'

  useEffect(() => {
    setActivityOpen(layout === 'desktop')
  }, [layout])

  // Tells the home hero whether quick access already has a place in the rail.
  const railCards = showRail && Boolean(rail)

  useEffect(() => {
    $jarvisRailVisible.set(railCards)

    return () => $jarvisRailVisible.set(false)
  }, [railCards])

  const tipsLauncher = (
    <JarvisTipsLauncher
      busy={busy}
      className={home ? 'jarvis-glass jarvis-glass-hover rounded-full px-4 text-(--ui-text-primary)' : undefined}
      hasHistory={state.activity.length > 0}
    />
  )

  const conversation = (
    <main
      aria-label={copy.conversationLabel}
      className="relative isolate flex min-h-0 min-w-0 flex-1 flex-col overflow-hidden bg-(--ui-chat-surface-background)"
      data-home={home ? 'true' : undefined}
    >
      {/* Home only: the Earth-from-orbit backdrop behind the orb. Decoration, never a hit target. */}
      {home ? <span aria-hidden="true" className="jarvis-space" /> : null}
      {/* Balanced, centred header: the orb in the middle of the conversation
          and its status beneath it. The orb stays compact while you read and
          grows — smoothly, see core.css — while you talk with Jarvis. */}
      <div className="flex shrink-0 flex-col items-center gap-3 px-4 pt-4 md:px-5">
        {home ? null : (
          <JarvisCore compact={compactCore && !voiceActive} live taskPhase={state.task.phase} voice={state.voice} />
        )}
        {home ? null : <button className="text-xs font-medium text-(--ui-text-tertiary) transition-colors hover:text-(--ui-accent)" onClick={() => navigate(NEW_CHAT_ROUTE)} type="button">{locale === 'pl' ? 'Na pulpit' : 'To dashboard'}</button>}
        {home ? <HomeTopBar tips={tipsLauncher} /> : null}
        <div className="flex flex-wrap items-center justify-center gap-2">
          {/* At rest on home the hero's own status line says it; the pills would
              only crowd the greeting. */}
          {home && connected && !busy && state.voice === 'idle' && state.activeTool === null ? null : (
            <JarvisStatusStrip className="justify-center" connected={connected} copy={copy.status} state={state} />
          )}
          {/* The deck is capability- and history-aware, so it lives here rather
              than behind a menu: it is the answer to "and now what?" that the
              empty greeting above raises. On home it sits in the top bar. */}
          {home ? null : tipsLauncher}
        </div>
      </div>
      {/* On home at rest the hero's talk button is the voice entry point; in a
          conversation the controls sit centred under the orb. */}
      {voiceControls && (!home || busy || state.voice !== 'idle') ? (
        <div className="mx-auto w-full max-w-2xl shrink-0 px-4 pt-3 md:px-5">{voiceControls}</div>
      ) : null}
      {home ? null : <ResultHeader copy={copy} profileDisplayName={profileDisplayName} state={state} />}
      <div className="min-h-0 flex-1 overflow-hidden">{children}</div>
    </main>
  )

  useEffect(() => {
    if (layout === 'desktop' || !activityOpen) {
      return undefined
    }

    activityCloseRef.current?.focus()

    const closeOnEscape = (event: KeyboardEvent) => {
      if (event.key !== 'Escape') {
        return
      }

      event.preventDefault()
      setActivityOpen(false)
      activityToggleRef.current?.focus()
    }

    document.addEventListener('keydown', closeOnEscape)

    return () => {
      document.removeEventListener('keydown', closeOnEscape)
    }
  }, [activityOpen, layout])

  const closeActivity = () => {
    setActivityOpen(false)
    activityToggleRef.current?.focus()
  }

  const insightsPanel = (
    <JarvisInsightsPanel
      className={cn(
        layout === 'desktop' &&
          (rail ? 'min-h-[22rem] shrink-0 rounded-xl border border-(--ui-stroke-tertiary)' : 'w-80'),
        layout === 'tablet' && 'absolute inset-y-4 right-4 z-20 w-80 rounded-md',
        layout === 'mobile' && 'absolute inset-x-3 bottom-16 z-20 max-h-[60vh] rounded-md'
      )}
      closeButtonRef={activityCloseRef}
      copy={{
        activity: copy.activity,
        news: copy.news,
        stats: copy.stats,
        tabs: copy.insightTabs,
        viewsLabel: copy.insightViewsLabel
      }}
      events={state.activity}
      id={activityPanelId}
      labelledBy={activityTitleId}
      metrics={metrics}
      news={news}
      onClose={layout === 'desktop' ? undefined : closeActivity}
      onOpenUpdate={onOpenUpdate}
      onViewChange={setView}
      surface={layout === 'desktop' ? 'panel' : layout === 'tablet' ? 'drawer' : 'bottom-sheet'}
      view={view}
    />
  )

  return (
    <section
      className={cn(
        'relative flex h-full min-h-0 min-w-0 overflow-hidden bg-(--ui-chat-surface-background) text-(--ui-text-primary)',
        layout === 'mobile' && 'flex-col',
        className
      )}
      data-jarvis-dashboard=""
      data-layout={layout}
      data-testid="jarvis-dashboard"
    >
      {conversation}
      {showRail &&
        (rail ? (
          // The rail scrolls as one column: the cards first, then the session's
          // own activity — so a long news list never squeezes the log away.
          <div
            className="flex w-80 shrink-0 flex-col gap-3 overflow-y-auto border-l border-(--ui-stroke-tertiary) p-3"
            data-testid="jarvis-rail"
          >
            {rail}
            {insightsPanel}
            <p className="mt-auto flex items-center justify-end gap-2 px-1 pt-2 text-xs text-(--ui-text-tertiary)">
              {t.jarvisShell.home.footerMotto}
              <span
                aria-hidden="true"
                className="size-2 rounded-full bg-(--ui-accent) shadow-[0_0_10px_var(--ui-accent)]"
              />
            </p>
          </div>
        ) : (
          insightsPanel
        ))}
      {layout !== 'desktop' && (
        <Button
          aria-controls={activityPanelId}
          aria-expanded={activityOpen}
          aria-label={copy.showActivity}
          className="absolute right-4 top-4 z-20 min-h-11"
          data-jarvis-activity-toggle
          onClick={() => setActivityOpen(open => !open)}
          ref={activityToggleRef}
          size="sm"
          type="button"
          variant="secondary"
        >
          <Activity />
          {copy.showActivity}
          {attention > 0 && (
            <Badge size="xs" variant="warn">
              {attention}
            </Badge>
          )}
        </Button>
      )}
      {layout !== 'desktop' && activityOpen && insightsPanel}
    </section>
  )
}
