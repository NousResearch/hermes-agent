import { computed } from 'nanostores'

import { sessionProjectColor } from '@/app/chat/sidebar/projects/workspace-groups'
import { Codecs, persistentAtom } from '@/lib/persisted'
import { $projectOwnerBySessionId, $projects } from '@/store/projects'
import { $sessions, sessionPinId } from '@/store/session'
import type { ProjectInfo, SessionInfo } from '@/types/hermes'

// Per-session color OVERRIDES — a user-picked color that wins over the inherited
// project color (#66565 layer 2). Desktop-local like pins, keyed by the DURABLE
// lineage id so a color survives auto-compression's session-id rotation. To take
// this to the TUI later, promote this one atom to a backend SessionInfo.color
// field — the resolver below and the picker UI stay exactly as they are.
export const $sessionColorOverrides = persistentAtom<Record<string, string>>(
  'hermes.desktop.sessionColors',
  {},
  Codecs.stringRecord
)

// TILE colours — the tab's own tier, written by whoever owns the grouping the
// user thinks in (the Personae pane colours a group; core has no group model).
// Deliberately SEPARATE from `$sessionColorOverrides`: a group tint must be able
// to paint a tab without rewriting — and so destroying — a colour the user
// picked by hand for that one session, which is a value the writer cannot read
// back and therefore cannot restore. Same durable keying, same store shape.
export const $sessionTileColors = persistentAtom<Record<string, string>>(
  'hermes.desktop.tileColors',
  {},
  Codecs.stringRecord
)

// Set a session's override (null clears it → falls back to the project color).
export function setSessionColorOverride(durableId: string, color: null | string): void {
  writeColor($sessionColorOverrides, durableId, color)
}

// Set (or clear) a session TAB's colour, below the explicit override and above
// the inherited project colour. `null` clears — writing '' would leave a key
// behind that paints nothing.
export function setSessionTileColor(durableId: string, color: null | string): void {
  writeColor($sessionTileColors, durableId, color)
}

function writeColor(atom: typeof $sessionColorOverrides, durableId: string, color: null | string): void {
  const prev = atom.get()

  if (color) {
    atom.set({ ...prev, [durableId]: color })
  } else if (durableId in prev) {
    const next = { ...prev }
    delete next[durableId]
    atom.set(next)
  }
}

// The resolved color for every session, keyed by live session id — the ONE
// source of truth both the sidebar rows and the pane tabs read, so the two
// surfaces can never drift. Recomputed only when the session list, projects, or
// overrides change (all cold atoms; the working/streaming pulse lives in
// $sessionStates, so a busy flip never rebuilds this), and every consumer reads
// it as an O(1) lookup rather than re-deriving membership per render.
//
// Precedence in one place: an explicit per-session override wins over the
// inherited project color. Agent-set color (#66565 layer 3) slots in here too.
function resolveSessionColor(
  session: SessionInfo,
  projects: ProjectInfo[],
  overrides: Record<string, string>,
  owners: ReadonlyMap<string, string>
): string | undefined {
  return overrides[sessionPinId(session)] ?? sessionProjectColor(session, projects, owners) ?? undefined
}

export const $sessionColorById = computed(
  [$sessions, $projects, $sessionColorOverrides, $projectOwnerBySessionId],
  (sessions, projects, overrides, owners) => {
    const map: Record<string, string> = {}

    for (const session of sessions) {
      const color = resolveSessionColor(session, projects, overrides, owners)

      if (color) {
        map[session.id] = color
      }
    }

    return map
  }
)

function resolveTabTint(
  session: SessionInfo,
  projects: ProjectInfo[],
  overrides: Record<string, string>,
  tileColors: Record<string, string>,
  owners: ReadonlyMap<string, string>
): string | undefined {
  const key = sessionPinId(session)

  // Most specific intent wins: the session's own pick, then the tab/group tier,
  // then the colour inherited from the project. The tile tier deliberately sits
  // BELOW the override so a group recolour can never repaint a session the user
  // coloured by hand.
  return overrides[key] ?? tileColors[key] ?? sessionProjectColor(session, projects, owners) ?? undefined
}

// The tab's colour for every session, keyed by live session id — same shape and
// same cold-atom discipline as `$sessionColorById`, so a tab can subscribe to one
// map and repaint when any tier changes. Separate from `$sessionColorById`
// because the tile tier must NOT move the sidebar row's dot: that dot says what
// the session IS, and a group tint is a statement about where it is filed.
export const $sessionTabTintById = computed(
  [$sessions, $projects, $sessionColorOverrides, $sessionTileColors, $projectOwnerBySessionId],
  (sessions, projects, overrides, tileColors, owners) => {
    const map: Record<string, string> = {}

    for (const session of sessions) {
      const color = resolveTabTint(session, projects, overrides, tileColors, owners)

      if (color) {
        map[session.id] = color
      }
    }

    return map
  }
)

// The tab colour for a single session object, through the same map the map is
// built from — a tile whose session is older than the paginated recents page
// misses the map, exactly like `sessionColorFor`.
export function sessionTabTintFor(session: null | SessionInfo | undefined): string | undefined {
  if (!session) {
    return undefined
  }

  return (
    $sessionTabTintById.get()[session.id] ??
    resolveTabTint(
      session,
      $projects.get(),
      $sessionColorOverrides.get(),
      $sessionTileColors.get(),
      $projectOwnerBySessionId.get()
    )
  )
}

// The color for a single session object (the tabs already hold the SessionInfo
// they render, so they resolve through the same map the sidebar reads). A row
// that isn't in `$sessions` — e.g. a project-tree session older than the
// paginated recents page, opened as a tab — misses the map, so fall back to the
// same resolver the map is built from.
export function sessionColorFor(session: null | SessionInfo | undefined): string | undefined {
  if (!session) {
    return undefined
  }

  return (
    $sessionColorById.get()[session.id] ??
    resolveSessionColor(session, $projects.get(), $sessionColorOverrides.get(), $projectOwnerBySessionId.get())
  )
}
