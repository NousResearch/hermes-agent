import { useStore } from '@nanostores/react'
import { type ReactNode, useEffect, useLayoutEffect, useMemo, useRef } from 'react'
import { useNavigate } from 'react-router-dom'

import { blurComposerInput } from '@/app/chat/composer/focus'
import { AGENTS_ROUTE } from '@/app/routes'
import { composerDockCard } from '@/components/chat/composer-dock'
import { StatusSection } from '@/components/chat/status-section'
import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { type Translations, useI18n } from '@/i18n'
import { cn } from '@/lib/utils'
import {
  $statusItemsBySession,
  type ComposerStatusItem,
  dismissBackgroundProcess,
  groupStatusItems,
  refreshBackgroundProcesses,
  type StatusGroup,
  stopBackgroundProcess
} from '@/store/composer-status'
import { $threadScrolledUp } from '@/store/thread-scroll'
import { openSessionInNewWindow } from '@/store/windows'

import { StatusItemRow } from './status-row'

// Slow safety-net poll for silent exits (processes without notify_on_complete
// emit no event when they die). Only armed while a running row is on screen.
const BACKGROUND_POLL_MS = 5_000

const groupLabel = (group: StatusGroup, s: Translations['statusStack']) => {
  if (group.type === 'todo') {
    return s.todos(group.items.filter(i => i.todoStatus === 'completed').length, group.items.length)
  }

  return group.type === 'subagent' ? s.subagents(group.items.length) : s.background(group.items.length)
}

interface ComposerStatusStackProps {
  /** The queue, built by the composer (it owns the queue's callbacks). Rendered
   *  as the last group so it stays fused to the composer like before. */
  queue: ReactNode
  sessionId: null | string
}

/**
 * The status "sink" above the composer: one card (the queue's chrome) holding
 * every session-scoped status — subagents, background tasks, queue — grouped by
 * type and separated by light dividers. Collapses to nothing when empty.
 */
export function ComposerStatusStack({ queue, sessionId }: ComposerStatusStackProps) {
  const { t } = useI18n()
  const navigate = useNavigate()
  const itemsBySession = useStore($statusItemsBySession)
  const scrolledUp = useStore($threadScrolledUp)

  const groups = useMemo(
    () => groupStatusItems(sessionId ? (itemsBySession[sessionId] ?? []) : []),
    [itemsBySession, sessionId]
  )

  // Seed from the registry on session open; event-driven refreshes (terminal /
  // process tool completions) live in use-message-stream.
  useEffect(() => {
    if (sessionId) {
      void refreshBackgroundProcesses(sessionId)
    }
  }, [sessionId])

  const hasRunningBackground = groups.some(g => g.type === 'background' && g.items.some(i => i.state === 'running'))

  useEffect(() => {
    if (!sessionId || !hasRunningBackground) {
      return
    }

    const timer = setInterval(() => void refreshBackgroundProcesses(sessionId), BACKGROUND_POLL_MS)

    return () => clearInterval(timer)
  }, [hasRunningBackground, sessionId])

  const openAgents = () => navigate(AGENTS_ROUTE)

  const openSubagent = (item: ComposerStatusItem) =>
    item.sessionId ? void openSessionInNewWindow(item.sessionId, { watch: true }) : openAgents()

  const sections: { key: string; node: ReactNode }[] = groups.map(group => ({
    key: group.type,
    node: (
      <StatusSection
        accessory={
          group.type === 'subagent' ? (
            <Button
              className="text-muted-foreground/75 hover:text-foreground/90"
              onClick={openAgents}
              size="micro"
              type="button"
              variant="text"
            >
              {t.statusStack.agents}
            </Button>
          ) : undefined
        }
        defaultCollapsed={group.type !== 'todo'}
        icon={
          group.type === 'todo' ? (
            <Codicon className="text-muted-foreground/70" name="checklist" size="0.8rem" />
          ) : undefined
        }
        label={groupLabel(group, t.statusStack)}
      >
        {group.items.map(item => (
          <StatusItemRow
            item={item}
            key={item.id}
            onDismiss={sessionId ? id => dismissBackgroundProcess(sessionId, id) : undefined}
            onOpen={() => openSubagent(item)}
            onStop={sessionId ? id => stopBackgroundProcess(sessionId, id) : undefined}
          />
        ))}
      </StatusSection>
    )
  }))

  if (queue) {
    sections.push({ key: 'queue', node: queue })
  }

  const visible = sections.length > 0
  const stackRef = useRef<HTMLDivElement | null>(null)

  // The stack is out of flow (overlays the thread), so the composer's measured
  // height never sees it. Publish our own measured height — bucketed like the
  // composer's, to avoid style invalidation churn — so the thread's
  // last-message clearance can add it and the stack never hides messages.
  useLayoutEffect(() => {
    const root = document.documentElement
    const el = stackRef.current

    if (!visible || !el) {
      root.style.removeProperty('--status-stack-measured-height')

      return
    }

    let last = -1

    const sync = () => {
      const bucket = Math.round(el.getBoundingClientRect().height / 8) * 8

      if (bucket !== last) {
        last = bucket
        root.style.setProperty('--status-stack-measured-height', `${bucket}px`)
      }
    }

    const observer = new ResizeObserver(sync)
    observer.observe(el)
    sync()

    return () => {
      observer.disconnect()
      root.style.removeProperty('--status-stack-measured-height')
    }
  }, [visible])

  if (!visible) {
    return null
  }

  return (
    <div
      className="absolute inset-x-0 bottom-full z-6 -mb-[9px] max-h-[40vh] overflow-y-auto"
      onPointerDownCapture={() => blurComposerInput()}
      ref={stackRef}
    >
      {/* The card paints the shared --composer-fill (rest / scrolled / focused
          all match the composer surface by construction); on scroll we only
          ghost the CONTENT — element opacity on the card would kill the blur. */}
      <div className={cn(composerDockCard('top'), 'mx-1 pt-0.5 pb-1')}>
        <div
          className={cn(
            'transition-opacity duration-200 ease-out',
            scrolledUp ? 'opacity-30 group-hover/composer:opacity-100' : 'opacity-100'
          )}
        >
          {sections.map(section => (
            <div key={section.key}>{section.node}</div>
          ))}
        </div>
      </div>
    </div>
  )
}
