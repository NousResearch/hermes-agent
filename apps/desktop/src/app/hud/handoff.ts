/**
 * HUD ⇄ app-window handoff.
 *
 * The gateway binds a session's event stream to exactly ONE socket — the last
 * one to submit or resume it (`session["transport"]`). The HUD is a full
 * renderer with its own socket, so entering HUD mode moves that binding to the
 * HUD and the app window stops hearing the session entirely: no deltas, no
 * turn-complete, no draft clear. Nothing to poll for either, since mid-turn
 * there is nothing persisted to re-pull.
 *
 * So leaving HUD mode is a re-home, not a window close. The app window resumes
 * the session the HUD ended on — the existing hydration path, which rebinds the
 * transport, reconciles the transcript, and picks up an in-flight turn — and
 * repaints its composer from the shared draft stash the HUD has been writing.
 */

import { useStore } from '@nanostores/react'
import { useEffect, useRef } from 'react'

import { reloadPersistedDrafts, requestComposerDraftSync } from '@/store/composer'
import { reportHudSession, watchHudState } from '@/store/hud'
import { $selectedStoredSessionId } from '@/store/session'
import { isHudWindow } from '@/store/windows'

import { openSession, type OpenSessionNavigate } from '../open-session'

interface HudHandoffParams {
  navigate: OpenSessionNavigate
  resumeSession: (storedSessionId: string) => unknown
}

/** App-window side: take the session back when the HUD goes away. Also keeps
 *  the titlebar toggle honest when the HUD is closed from its own side. */
export function useHudHandoff({ navigate, resumeSession }: HudHandoffParams): void {
  const paramsRef = useRef<HudHandoffParams>({ navigate, resumeSession })
  paramsRef.current = { navigate, resumeSession }

  useEffect(() => {
    // The HUD's own renderer mounts the same wiring; it is the window going
    // away, so it has nothing to re-home.
    if (isHudWindow()) {
      return
    }

    return watchHudState(hudSessionId => {
      // The HUD may have typed or sent since this window last read the stash.
      reloadPersistedDrafts()

      const selected = $selectedStoredSessionId.get()
      const target = hudSessionId ?? selected

      // The HUD switched sessions (or started one this window has never seen):
      // route to it and let the route resume do the rest, including loading
      // that session's draft as the composer's scope swaps.
      if (target && target !== selected) {
        openSession(target, paramsRef.current.navigate)

        return
      }

      // Same session, so the composer's scope never changes and its
      // per-session swap effect will never re-consult the stash. Repaint it.
      requestComposerDraftSync('reload')

      if (target) {
        void paramsRef.current.resumeSession(target)
      }
    })
  }, [])
}

/** HUD side: keep main told which session this window is on. */
export function useReportHudSession(): void {
  const selectedStoredSessionId = useStore($selectedStoredSessionId)

  useEffect(() => {
    if (isHudWindow()) {
      reportHudSession(selectedStoredSessionId)
    }
  }, [selectedStoredSessionId])
}
