import { type ReactNode, useEffect, useId, useRef, useState } from 'react'

import { Button } from '@/components/ui/button'
import { useI18n } from '@/i18n'
import { Activity } from '@/lib/icons'
import { cn } from '@/lib/utils'

import { JarvisActivityPanel } from './activity-panel'
import { JarvisCore } from './core'
import { JarvisStatusStrip } from './status-strip'
import type { JarvisUiState } from './types'

type DashboardLayout = 'desktop' | 'mobile' | 'tablet'

export interface JarvisDashboardProps {
  children: ReactNode
  className?: string
  connected: boolean
  layout?: DashboardLayout
  profileDisplayName?: string
  state: JarvisUiState
}

function greeting(copy: ReturnType<typeof useI18n>['t']['jarvisShell']['dashboard'], profileDisplayName?: string): string {
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
  layout: layoutOverride,
  profileDisplayName,
  state
}: JarvisDashboardProps) {
  const { t } = useI18n()
  const copy = t.jarvisShell.dashboard
  const layout = useDashboardLayout(layoutOverride)
  const [activityOpen, setActivityOpen] = useState(layout === 'desktop')
  const compactCore = layout !== 'desktop'
  const activityPanelId = useId()
  const activityTitleId = useId()
  const activityToggleRef = useRef<HTMLButtonElement>(null)
  const activityCloseRef = useRef<HTMLButtonElement>(null)

  useEffect(() => {
    setActivityOpen(layout === 'desktop')
  }, [layout])

  const conversation = (
    <main
      aria-label={copy.conversationLabel}
      className="flex min-h-0 min-w-0 flex-1 flex-col overflow-hidden bg-(--ui-chat-surface-background)"
    >
      <div className="flex shrink-0 flex-col gap-4 px-4 pt-4 md:flex-row md:items-center md:justify-between md:px-5">
        <JarvisCore audioLevel={0} compact={compactCore} taskPhase={state.task.phase} voice={state.voice} />
        <JarvisStatusStrip connected={connected} copy={copy.status} state={state} />
      </div>
      <ResultHeader copy={copy} profileDisplayName={profileDisplayName} state={state} />
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

  const activityPanel = (
    <JarvisActivityPanel
      className={cn(
        layout === 'desktop' && 'w-80',
        layout === 'tablet' && 'absolute inset-y-4 right-4 z-20 w-80 rounded-md',
        layout === 'mobile' && 'absolute inset-x-3 bottom-16 z-20 max-h-[60vh] rounded-md'
      )}
      closeButtonRef={activityCloseRef}
      copy={copy.activity}
      events={state.activity}
      id={activityPanelId}
      labelledBy={activityTitleId}
      onClose={layout === 'desktop' ? undefined : closeActivity}
      surface={layout === 'desktop' ? 'panel' : layout === 'tablet' ? 'drawer' : 'bottom-sheet'}
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
      {layout === 'desktop' && activityPanel}
      {layout !== 'desktop' && (
        <Button
          aria-controls={activityPanelId}
          aria-expanded={activityOpen}
          aria-label={copy.showActivity}
          className="absolute right-4 top-4 z-20 min-h-11"
          onClick={() => setActivityOpen(open => !open)}
          ref={activityToggleRef}
          size="sm"
          type="button"
          variant="secondary"
        >
          <Activity />
          {copy.showActivity}
        </Button>
      )}
      {layout !== 'desktop' && activityOpen && activityPanel}
    </section>
  )
}
