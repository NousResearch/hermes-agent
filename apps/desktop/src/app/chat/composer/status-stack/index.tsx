import { useStore } from '@nanostores/react'
import { type ReactNode, useEffect, useMemo } from 'react'
import { useNavigate } from 'react-router-dom'

import { AGENTS_ROUTE } from '@/app/routes'
import { composerDockCard } from '@/components/chat/composer-dock'
import { StatusSection } from '@/components/chat/status-section'
import { Button } from '@/components/ui/button'
import { type Translations, useI18n } from '@/i18n'
import { cn } from '@/lib/utils'
import {
  $statusItemsBySession,
  dismissBackgroundProcess,
  groupStatusItems,
  refreshBackgroundProcesses,
  type StatusGroup,
  stopBackgroundProcess
} from '@/store/composer-status'
import { $threadScrolledUp } from '@/store/thread-scroll'

import { StatusItemRow } from './status-row'

// Slow safety-net poll for silent exits (processes without notify_on_complete
// emit no event when they die). Only armed while a running row is on screen.
const BACKGROUND_POLL_MS = 5_000

const groupLabel = (group: StatusGroup, s: Translations['statusStack']) =>
  group.type === 'subagent' ? s.subagents(group.items.length) : s.background(group.items.length)

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

  // Subagents don't have a standalone session to open yet, so both the header
  // accessory and row activation land on the Agents view (the live spawn tree).
  const openAgents = () => navigate(AGENTS_ROUTE)

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
        label={groupLabel(group, t.statusStack)}
      >
        {group.items.map(item => (
          <StatusItemRow
            item={item}
            key={item.id}
            onDismiss={sessionId ? id => dismissBackgroundProcess(sessionId, id) : undefined}
            onOpen={openAgents}
            onStop={sessionId ? id => stopBackgroundProcess(sessionId, id) : undefined}
          />
        ))}
      </StatusSection>
    )
  }))

  if (queue) {
    sections.push({ key: 'queue', node: queue })
  }

  if (sections.length === 0) {
    return null
  }

  return (
    <div className="absolute inset-x-0 bottom-full z-6 -mb-[9px] max-h-[40vh] overflow-y-auto">
      {/* Mirror the composer's fade: keep the blur constant (element opacity
          would kill it) and only shift the bg color on scroll; ghost the
          content separately. Drive the faded state from `scrolledUp` (not a
          group-data variant) so the group-hover/focus override always wins. */}
      <div
        className={cn(
          composerDockCard('top'),
          'mx-1 pt-0.5 pb-1 transition-colors duration-200 ease-out',
          scrolledUp &&
            'bg-[color-mix(in_srgb,var(--dt-card)_60%,transparent)] group-hover/composer:bg-[color-mix(in_srgb,var(--dt-card)_92%,transparent)] group-focus-within/composer:bg-[color-mix(in_srgb,var(--dt-card)_92%,transparent)]'
        )}
      >
        <div
          className={cn(
            'transition-opacity duration-200 ease-out',
            scrolledUp
              ? 'opacity-30 group-hover/composer:opacity-100 group-focus-within/composer:opacity-100'
              : 'opacity-100'
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
