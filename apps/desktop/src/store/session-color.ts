import { computed } from 'nanostores'

import { sessionProjectColor } from '@/app/chat/sidebar/projects/workspace-groups'
import { $projects } from '@/store/projects'
import { $sessions } from '@/store/session'
import type { SessionInfo } from '@/types/hermes'

// The resolved color for every session, keyed by live session id — the ONE
// source of truth both the sidebar rows and the pane tabs read, so the two
// surfaces can never drift. Recomputed only when the session list or the
// projects change (both cold atoms; the working/streaming pulse lives in
// $sessionStates, so a busy flip never rebuilds this), and every consumer reads
// it as an O(1) lookup rather than re-deriving membership per render.
//
// Precedence lives in one place: today a session inherits its project's color;
// when per-session overrides / agent-set colors land (#66565 layers 2-3), fold
// them in ABOVE the project fallback here and every surface updates for free.
export const $sessionColorById = computed([$sessions, $projects], (sessions, projects) => {
  const map: Record<string, string> = {}

  for (const session of sessions) {
    const color = sessionProjectColor(session, projects)

    if (color) {
      map[session.id] = color
    }
  }

  return map
})

// The color for a single session object (the tabs already hold the SessionInfo
// they render, so they resolve through the same map the sidebar reads).
export function sessionColorFor(session: null | SessionInfo | undefined): string | undefined {
  return session ? $sessionColorById.get()[session.id] : undefined
}
