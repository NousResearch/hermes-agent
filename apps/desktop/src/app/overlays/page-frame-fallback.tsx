import { useLocation } from 'react-router'

import { useMediaQuery } from '@/hooks/use-media-query'
import { usePortalShell } from '@/store/shell-variant'

import { type PageFrameKind, PageHeaderSkeleton, usePageFrame } from './page-header-skeleton'
import { PAGE_SIDEBAR_NARROW, PageSidebarPanel } from './page-sidebar'
import { Panel } from './panel'

interface PageFrameFallbackProps {
  frame: PageFrameKind
  onClose: () => void
}

/** A routed overlay page while its code loads: the page frame itself (band, stage, the page's
 *  panel and header), never an empty window. Classic keeps its blank wait. */
export function PageFrameFallback({ frame, onClose }: PageFrameFallbackProps) {
  const page = usePageFrame(frame)
  const portal = usePortalShell()

  if (!portal) {
    return null
  }

  return (
    <Panel contentClassName="page-frame-fallback" onClose={onClose}>
      {page.panel && <PageSidebarPanel title={page.panel}>{null}</PageSidebarPanel>}
      <PageHeaderSkeleton frame={page} />
    </Panel>
  )
}

/** The Settings route serves both the global settings and an agent's page. */
export function SettingsFrameFallback({ onClose }: { onClose: () => void }) {
  const agent = new URLSearchParams(useLocation().search).has('agentProfile')

  return <PageFrameFallback frame={agent ? 'agent' : 'settings'} onClose={onClose} />
}

/** The Agents destination opens an agent's page, or below wide the agents list (AgentDirectory). */
export function AgentsFrameFallback({ onClose }: { onClose: () => void }) {
  const narrow = useMediaQuery(PAGE_SIDEBAR_NARROW)

  return <PageFrameFallback frame={narrow ? 'agents' : 'agent'} onClose={onClose} />
}
