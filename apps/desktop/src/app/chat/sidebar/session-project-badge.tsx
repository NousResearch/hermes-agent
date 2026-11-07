import type * as React from 'react'

import { Codicon } from '@/components/ui/codicon'
import { Tip } from '@/components/ui/tooltip'
import { sessionProjectLabel } from '@/lib/session-project-label'
import { useStoreSelector } from '@/lib/use-session-slice'
import { $projects } from '@/store/projects'
import type { SessionInfo } from '@/types/hermes'

/**
 * A one-line session row's project mark: a small folder glyph before the
 * title, shown when the session resolves to a project — the same resolver the
 * card header and the project-lane grouping read, so icon, tint and lane
 * membership cannot disagree. Card rows skip the badge: their header line
 * already names the project, and a second mark would read as duplication.
 * Home-less chats paint nothing.
 */
export function SessionProjectBadge({ card, session }: { card: boolean; session: SessionInfo }): React.ReactNode {
  const projectLabel = useStoreSelector($projects, projects =>
    card ? null : sessionProjectLabel(session, projects)
  )

  if (!projectLabel || card) {
    return null
  }

  return (
    <Tip label={projectLabel}>
      <Codicon aria-hidden="true" className="size-3.5 shrink-0 text-(--ui-text-quaternary)" name="folder" size="0.75rem" />
    </Tip>
  )
}
