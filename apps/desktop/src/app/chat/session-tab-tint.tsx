import { useStore } from '@nanostores/react'
import type { ReactNode } from 'react'

import { tabTintVars } from '@/lib/session-tab-tint'
import { $sessions, ownerLookupSessionRows, sessionMatchesStoredId } from '@/store/session'
import { $sessionTabTintById, sessionTabTintFor } from '@/store/session-color'
import type { SessionInfo } from '@/types/hermes'

export interface SessionTabTintProps {
  children: ReactNode
  /** The row, when the caller already holds it — a tile whose session is older
   *  than the paginated recents page has no entry in the shared map, and this is
   *  how it still resolves its own colour. */
  session?: null | SessionInfo
  /** Fallback resolution key: the STORED id every pane tile is keyed by. */
  storedSessionId?: null | string
}

/**
 * A session TAB's colour. `PaneTab` resolves its surface through the custom
 * properties `--pane-tab-strip-bg` / `--pane-tab-active-bg` /
 * `--pane-tab-active-accent`; this sets them on a `display: contents` wrapper,
 * from which they inherit onto the tab, whichever state it is in. Nothing about
 * the tab component changes, and no tab state needs to be known here.
 *
 * Self-subscribing on purpose: the colour changes when the user picks one, when
 * a group is recoloured through the plugin door, or when a project's colour
 * moves — none of which re-registers a pane, so a value read at `tabWrap` time
 * would paint the tab once and then lie. (`SessionStatusDot` solves the same
 * problem the same way, one tier down.)
 */
export function SessionTabTint({ children, session, storedSessionId }: SessionTabTintProps) {
  useStore($sessions)
  useStore($sessionTabTintById)

  const row =
    session ??
    (storedSessionId
      ? (ownerLookupSessionRows().find(candidate => sessionMatchesStoredId(candidate, storedSessionId)) ?? null)
      : null)

  const style = tabTintVars(sessionTabTintFor(row))

  // No colour means no wrapper: an unpainted tab keeps today's markup exactly,
  // so `--tab-bg` still resolves from the strip it sits in.
  if (!style) {
    return <>{children}</>
  }

  return (
    <span className="contents" style={style}>
      {children}
    </span>
  )
}
