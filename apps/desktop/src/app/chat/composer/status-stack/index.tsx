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
import { $previewStatusBySession, dismissPreviewArtifact } from '@/store/preview-status'
import { $threadScrolledUp } from '@/store/thread-scroll'
import { openSessionInNewWindow } from '@/store/windows'

import { PreviewStatusRow } from './preview-row'
import { StatusItemRow } from './status-row'

// Slow safety-net poll for silent exits (processes without notify_on_complete
// emit no event when they die). Only armed while a running row is on screen.
const BACKGROUND_POLL_MS = 5_000

// A localhost/loopback preview is only meaningful while its dev server is up, so
// we tie it to a live background process rather than persisting dismissals or
// letting dead URLs pile up. File previews (a real on-disk artifact) stand alone.
const isLocalhostPreview = (target: string): boolean => /\b(?:localhost|127\.0\.0\.1|0\.0\.0\.0)\b/i.test(target)

// Real codicons per group (no sparkles): a checklist for todos, the agent glyph
// for subagents, a background process glyph for background tasks.
const GROUP_ICON: Record<StatusGroup['type'], string> = {
  todo: 'checklist',
  subagent: 'agent',
  background: 'server-process'
}

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
  const previewsBySession = useStore($previewStatusBySession)
  const scrolledUp = useStore($threadScrolledUp)

  const groups = useMemo(
    () => groupStatusItems(sessionId ? (itemsBySession[sessionId] ?? []) : []),
    [itemsBySession, sessionId]
  )

  const previews = sessionId ? (previewsBySession[sessionId] ?? []) : []

  // Seed from the registry on session open; event-driven refreshes (terminal /
  // process tool completions) live in use-message-stream.
  useEffect(() => {
    if (sessionId) {
      void refreshBackgroundProcesses(sessionId)
    }
  }, [sessionId])

  const hasRunningBackground = groups.some(g => g.type === 'background' && g.items.some(i => i.state === 'running'))

  // Drop localhost previews once no dev server is left running — that's what made
  // dead `localhost:5174` chips stick around. On-disk file previews are kept.
  const visiblePreviews = previews.filter(item => hasRunningBackground || !isLocalhostPreview(item.target))

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

  // Preview links live as child rows of the background group — a localhost dev
  // server and its preview are the same thing — so they no longer float as an
  // odd, differently-indented standalone block under the stack.
  const previewRows =
    visiblePreviews.length > 0 && sessionId
      ? visiblePreviews.map(item => (
          <PreviewStatusRow item={item} key={item.id} onDismiss={id => dismissPreviewArtifact(sessionId, id)} />
        ))
      : []

  const hasBackgroundGroup = groups.some(g => g.type === 'background')

  const previewBlock = <div className="px-1 py-0.5">{previewRows}</div>

  const sections: { key: string; node: ReactNode }[] = []

  for (const group of groups) {
    sections.push({
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
          icon={<Codicon className="text-muted-foreground/70" name={GROUP_ICON[group.type]} size="0.8rem" />}
          label={groupLabel(group, t.statusStack)}
        >
          {group.items.map(item => (
            <StatusItemRow
              item={item}
              key={item.id}
              onDismiss={sessionId ? id => dismissBackgroundProcess(sessionId, id) : undefined}
              onOpen={() => openSubagent(item)}
              onStop={sessionId ? id => void stopBackgroundProcess(sessionId, id) : undefined}
            />
          ))}
        </StatusSection>
      )
    })

    // Preview links belong to the background group (a localhost dev server and
    // its preview are the same thing), but they must stay VISIBLE even when that
    // group is collapsed — the whole point is a one-tap open. Render them as an
    // always-visible block right after the background section, not as collapsible
    // children that get swallowed the moment a background task appears.
    if (group.type === 'background' && previewRows.length > 0) {
      sections.push({ key: 'preview', node: previewBlock })
    }
  }

  // No background group to host them (e.g. a standalone on-disk file preview):
  // still render them as their own always-visible block.
  if (previewRows.length > 0 && !hasBackgroundGroup) {
    sections.push({ key: 'preview', node: previewBlock })
  }

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
      // Sits in the overlay lane above the composer. The composer root has pt-2
      // before the actual surface; translate by that amount so the stack returns
      // to its original attachment point without intruding into the repo strip.
      className="absolute inset-x-0 bottom-full z-3 max-h-[40vh] translate-y-2 overflow-y-auto"
      onPointerDownCapture={() => blurComposerInput()}
      ref={stackRef}
    >
      {/* The card paints the shared --composer-fill (rest / scrolled / focused
          all match the composer surface by construction); on scroll we only
          ghost the CONTENT — element opacity on the card would kill the blur.
          Rounded top, square bottom; the bottom border is TRANSPARENT — the
          composer surface's visible top border (which sits at a higher z) is the
          single shared seam, so the two read as one fused capsule. */}
      <div
        className={cn(
          composerDockCard('top'),
          // Inset (mx-2) so the stack reads slightly narrower than the composer
          // surface below it — the original look.
          'mx-2 overflow-hidden rounded-b-none border-b border-b-transparent pt-0.5',
          'transition-opacity duration-200 ease-out',
          scrolledUp ? 'opacity-30 group-hover/composer:opacity-100' : 'opacity-100'
        )}
      >
        {sections.map(section => (
          <div key={section.key}>{section.node}</div>
        ))}
      </div>
    </div>
  )
}
