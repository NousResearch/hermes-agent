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
import { focusOpenSession, sessionTileDelegate } from '@/store/session-states'
import { isHudWindow } from '@/store/windows'

import { getActiveComposer } from '../chat/composer/focus'
import { openSession, type OpenSessionNavigate } from '../open-session'
import { sessionRoute } from '../routes'

/** Session tiles route on `tile:<storedSessionId>` (see session-tile.tsx). */
const TILE_TARGET_PREFIX = 'tile:'

/**
 * The conversation HUD mode should open on: whichever chat surface the user is
 * actually looking at.
 *
 * `$selectedStoredSessionId` is the WORKSPACE pane's session, so reading it
 * alone sent the main tab into the HUD no matter which tile was fronted — the
 * tabs exist precisely so that isn't the same question. `getActiveComposer()`
 * already answers it for the focus bus, healing to the visible surface when its
 * cached claim is buried or gone, and a tile's routing key IS its stored
 * session id.
 */
export function hudTargetSessionId(): null | string {
  const target = getActiveComposer()
  const tile = target.startsWith(TILE_TARGET_PREFIX) ? target.slice(TILE_TARGET_PREFIX.length) : null

  return tile || $selectedStoredSessionId.get()
}

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

      // Somewhere other than the workspace pane. If it is an open tile, front
      // it and re-resume THROUGH the tile delegate: the ordinary resume path
      // enforces "a session is either main or a tile, never both" and would
      // close the tile to take it into main, quietly rearranging tabs the user
      // opened on purpose. Otherwise it's a session this window has never seen
      // — route to it and let the route resume do the rest, including loading
      // its draft as the composer's scope swaps.
      if (target && target !== selected) {
        const delegate = focusOpenSession(target) === 'tile' ? sessionTileDelegate() : null

        if (delegate) {
          void delegate.resumeTile(target).catch(() => undefined)

          return
        }

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

/** HUD side: follow a retarget. Asking for HUD mode from another tab while the
 *  HUD is already up switches the conversation showing in it. */
export function useHudGoto(navigate: OpenSessionNavigate): void {
  const navigateRef = useRef(navigate)
  navigateRef.current = navigate

  useEffect(() => window.hermesDesktop?.hud?.onGoto?.(id => navigateRef.current(sessionRoute(id))), [])
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
