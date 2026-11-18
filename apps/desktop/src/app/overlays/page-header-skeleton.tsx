import '../page-search-shell.css'

import { CountSkeleton } from '@/components/ui/skeleton'
import { useI18n } from '@/i18n'
import { usePortalShell } from '@/store/shell-variant'

import { PageHeader } from '../redesign/page-header'

import { usePageFrameCopy } from './page-frame-copy'

// The parts of a page's loading frame that need no page shell (no panel, no layout store), so the
// workspace routes can wait in them without pulling the overlay machinery in.

/** What a page's frame shows before the page itself can render: the parts the page always has. */
export interface PageFrame {
  /** The page's sidebar panel title, on a page that has one. */
  panel?: string
  /** The H1 when it is known up front; otherwise a bar at the H1's size stands in for it. */
  title?: string
  /** The line under the H1 when it is known up front; otherwise a bar stands in for it. */
  description?: string
  /** The header leads with a tile (the agent avatar). */
  leading?: boolean
  /** The header carries a tab bar. */
  tabs?: boolean
}

/** The routed pages whose frame differs from the plain one (a header at the stage inset). */
export type PageFrameKind =
  'agent' | 'agents' | 'capabilities' | 'library' | 'messaging' | 'page' | 'schedules' | 'settings'

export function usePageFrame(kind: PageFrameKind): PageFrame {
  const { t } = useI18n()
  const copy = usePageFrameCopy()

  const frames: Record<PageFrameKind, PageFrame> = {
    agent: { panel: t.simplifiedAgents.agents, leading: true, tabs: true },
    agents: { title: t.profiles.title, description: copy.agentsDescription },
    capabilities: { title: t.sidebar.nav.capabilities, tabs: true },
    library: { title: t.simplifiedAgents.library, description: copy.libraryDescription, tabs: true },
    messaging: { title: t.sidebar.nav.messaging, description: copy.messagingDescription },
    page: {},
    schedules: { description: copy.schedulesDescription, panel: t.cron.title, title: t.cron.title },
    settings: { panel: t.commandCenter.settings }
  }

  return frames[kind]
}

/** The page header with its known parts and bars in place of the rest, at the loaded header's
 *  geometry, so the page that replaces it lands without moving anything. */
export function PageHeaderSkeleton({ className, frame = {} }: { className?: string; frame?: PageFrame }) {
  const { t } = useI18n()

  return (
    <PageHeader
      aria-busy
      aria-label={t.common.loading}
      className={className}
      description={frame.description ?? <CountSkeleton className="page-frame-skeleton-line" />}
      leading={frame.leading ? <span aria-hidden className="page-frame-skeleton-tile" /> : undefined}
      tabs={frame.tabs ? <div aria-hidden className="portal-page-tabs page-frame-skeleton-tabs" /> : undefined}
      title={frame.title ?? <CountSkeleton className="page-frame-skeleton-title" />}
    />
  )
}

/** A workspace page (Capabilities, Library, Messaging) while its code loads: the page column and
 *  its header, as PageSearchShell lays them out. */
export function WorkspacePageFallback({ frame }: { frame: PageFrameKind }) {
  const page = usePageFrame(frame)
  const portal = usePortalShell()

  if (!portal) {
    return null
  }

  return (
    <section
      className="page-redesigned flex h-full min-w-0 flex-col overflow-hidden bg-(--ui-chat-surface-background)"
      data-page-surface
    >
      <div className="shrink-0">
        <PageHeaderSkeleton className="page-header" frame={page} />
      </div>
    </section>
  )
}
