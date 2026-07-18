import { describe, expect, it } from 'vitest'

import { resumeSessionNavTarget, sessionRoute } from './routes'

// Regression coverage for #66875: clicking the most recent (main) session from a
// non-chat full-page route (Plugins/Artifacts/Messaging/Skills) did nothing,
// because focusOpenSession fronts a pane inside the hidden chat layout and the
// old handler skipped navigation whenever the session was already open.
describe('resumeSessionNavTarget', () => {
  const MAIN = 'session-main'
  const OTHER = 'session-other'
  const ARTIFACTS = '/artifacts'
  const CHAT = sessionRoute(MAIN)

  it('routes a not-yet-open session into main via its own route', () => {
    expect(resumeSessionNavTarget(OTHER, false, ARTIFACTS, MAIN)).toBe(sessionRoute(OTHER))
    expect(resumeSessionNavTarget(OTHER, false, CHAT, MAIN)).toBe(sessionRoute(OTHER))
  })

  it('does nothing when the open session is clicked while already on the chat view', () => {
    expect(resumeSessionNavTarget(MAIN, true, CHAT, MAIN)).toBeNull()
    expect(resumeSessionNavTarget(OTHER, true, CHAT, MAIN)).toBeNull()
  })

  it('routes back to chat when the open MAIN session is clicked from a full page (#66875)', () => {
    for (const page of ['/artifacts', '/skills', '/messaging', '/cron']) {
      expect(resumeSessionNavTarget(MAIN, true, page, MAIN)).toBe(sessionRoute(MAIN))
    }
  })

  it('keeps an open tile a tile — routes to MAIN, not the clicked tile, from a full page', () => {
    // OTHER is an already-open tile; focusOpenSession fronted it. We must reveal
    // the chat view without promoting the tile to the main session.
    expect(resumeSessionNavTarget(OTHER, true, ARTIFACTS, MAIN)).toBe(sessionRoute(MAIN))
  })

  it('falls back to the clicked id when there is no main session', () => {
    expect(resumeSessionNavTarget(OTHER, true, ARTIFACTS, null)).toBe(sessionRoute(OTHER))
  })
})
