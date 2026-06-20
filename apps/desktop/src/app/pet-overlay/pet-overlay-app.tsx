import { useStore } from '@nanostores/react'
import { useEffect, useRef, useState } from 'react'

import { PetBubble } from '@/components/pet/pet-bubble'
import { PetSprite } from '@/components/pet/pet-sprite'
import { Mail } from '@/lib/icons'
import { $petActivity, $petInfo, setPetInfo } from '@/store/pet'
import { setAwaitingResponse, setBusy } from '@/store/session'

/**
 * The pop-out overlay's only view: a transparent, draggable mascot with a mini
 * composer.
 *
 * This runs in a separate, gateway-less BrowserWindow (`?win=overlay`). It is a
 * pure puppet — the main renderer pushes the live pet state over IPC and we
 * mirror it into the same atoms the in-window pet reads, so `PetSprite` /
 * `PetBubble` render identically with zero extra logic.
 *
 * The window is a full rectangle but mostly transparent; we toggle OS-level
 * mouse click-through so only the sprite (or the open composer) is interactive
 * and the empty margins pass clicks through to whatever is behind.
 *
 * Gestures on the pet: drag to move it anywhere on screen (even outside the
 * app), shift-click to pop it back into the window, single-click to open a small
 * composer, double-click to toggle the app window (minimize ↔ restore). A mail
 * icon (shown only when a turn finished while you were away) raises the app on
 * the most recent thread.
 */

// Below this much pointer travel, a press counts as a click, not a drag.
const CLICK_SLOP_PX = 3
// A second click within this window is a double-click (raise app) and cancels
// the deferred single-click (open composer), so a double never flashes it open.
const DOUBLE_CLICK_MS = 250

interface DragState {
  startX: number
  startY: number
  offX: number
  offY: number
  width: number
  height: number
  moved: boolean
}

export function PetOverlayApp() {
  const info = useStore($petInfo)
  const [composerOpen, setComposerOpen] = useState(false)
  const [draft, setDraft] = useState('')
  // Mirrored from the main renderer: a finish landed while you were away.
  const [unread, setUnread] = useState(false)

  const dragRef = useRef<DragState | null>(null)
  const petRef = useRef<HTMLDivElement | null>(null)
  const inputRef = useRef<HTMLInputElement | null>(null)
  const ignoreRef = useRef(true)
  const composerOpenRef = useRef(false)
  const clickTimerRef = useRef<ReturnType<typeof setTimeout> | undefined>(undefined)

  const setIgnore = (ignore: boolean) => {
    if (ignoreRef.current !== ignore) {
      ignoreRef.current = ignore
      window.hermesDesktop?.petOverlay?.setIgnoreMouse(ignore)
    }
  }

  // Mirror pushed state into the shared atoms so PetSprite/PetBubble just work.
  useEffect(() => {
    const off = window.hermesDesktop?.petOverlay?.onState(payload => {
      setPetInfo(payload.info)
      $petActivity.set(payload.activity ?? {})
      setBusy(Boolean(payload.busy))
      setAwaitingResponse(Boolean(payload.awaiting))
      setUnread(Boolean(payload.unread))
    })

    // Tell the main renderer we're mounted so it pushes the current frame (the
    // subscribe-time pushes during open() can land before this view exists).
    window.hermesDesktop?.petOverlay?.control({ type: 'ready' })

    return off
  }, [])

  // Click-through: make only the sprite (or an open composer) interactive. With
  // ignore+forward, the renderer still receives mousemove so we can re-enable
  // hit-testing the moment the cursor returns to the pet.
  useEffect(() => {
    setIgnore(true)

    const onMove = (ev: MouseEvent) => {
      if (dragRef.current || composerOpenRef.current) {
        setIgnore(false)

        return
      }

      const el = petRef.current

      if (!el) {
        return
      }

      const r = el.getBoundingClientRect()
      const over = ev.clientX >= r.left && ev.clientX <= r.right && ev.clientY >= r.top && ev.clientY <= r.bottom
      setIgnore(!over)
    }

    window.addEventListener('mousemove', onMove)

    return () => {
      window.removeEventListener('mousemove', onMove)
      clearTimeout(clickTimerRef.current)
    }
  }, [])

  // The whole window must stay interactive while the composer is open (so the
  // input keeps focus); focus it on open. The overlay is a non-activating panel
  // (so it never steals the app's cmd/alt-tab anchor) — flip it focusable while
  // the composer needs the keyboard, then back to non-activating when it closes.
  useEffect(() => {
    composerOpenRef.current = composerOpen

    window.hermesDesktop?.petOverlay?.setFocusable(composerOpen)

    if (composerOpen) {
      setIgnore(false)
      // The OS window has to become key first (setFocusable + focus happen in
      // the main process), so focus the input on the next frame.
      requestAnimationFrame(() => inputRef.current?.focus())
    }
  }, [composerOpen])

  const onPetPointerDown = (e: React.PointerEvent) => {
    if (e.button !== 0) {
      return
    }

    ;(e.target as Element).setPointerCapture?.(e.pointerId)
    dragRef.current = {
      height: window.outerHeight,
      moved: false,
      offX: e.screenX - window.screenX,
      offY: e.screenY - window.screenY,
      startX: e.screenX,
      startY: e.screenY,
      width: window.outerWidth
    }
  }

  const onPetPointerMove = (e: React.PointerEvent) => {
    const drag = dragRef.current

    if (!drag) {
      return
    }

    if (Math.hypot(e.screenX - drag.startX, e.screenY - drag.startY) > CLICK_SLOP_PX) {
      drag.moved = true
    }

    window.hermesDesktop?.petOverlay?.setBounds({
      height: drag.height,
      width: drag.width,
      x: e.screenX - drag.offX,
      y: e.screenY - drag.offY
    })
  }

  const onPetPointerUp = (e: React.PointerEvent) => {
    const drag = dragRef.current
    dragRef.current = null
    ;(e.target as Element).releasePointerCapture?.(e.pointerId)

    if (!drag) {
      return
    }

    if (drag.moved) {
      // A drag cancels any deferred single-click so the composer can't pop open
      // after you reposition the pet.
      clearTimeout(clickTimerRef.current)
      clickTimerRef.current = undefined

      // Remember the spot on the desktop (screen coords) so the pet reopens here
      // next time / after a restart.
      window.hermesDesktop?.petOverlay?.control({
        bounds: { height: drag.height, width: drag.width, x: e.screenX - drag.offX, y: e.screenY - drag.offY },
        type: 'bounds'
      })

      return
    }

    // Shift-click always pops the pet back in (no double-click ambiguity).
    if (e.shiftKey) {
      window.hermesDesktop?.petOverlay?.control({ type: 'pop-in' })

      return
    }

    // Double-click toggles the app window (minimize ↔ restore); defer the
    // single-click composer toggle so a double never flashes the composer open.
    if (clickTimerRef.current) {
      clearTimeout(clickTimerRef.current)
      clickTimerRef.current = undefined
      window.hermesDesktop?.petOverlay?.control({ type: 'toggle-app' })

      return
    }

    clickTimerRef.current = setTimeout(() => {
      clickTimerRef.current = undefined
      setComposerOpen(open => !open)
    }, DOUBLE_CLICK_MS)
  }

  const send = () => {
    const text = draft.trim()

    if (text) {
      window.hermesDesktop?.petOverlay?.control({ text, type: 'submit' })
    }

    setDraft('')
    setComposerOpen(false)
  }

  const openApp = () => {
    // Hide the icon immediately; the main renderer also clears the source flag.
    setUnread(false)
    window.hermesDesktop?.petOverlay?.control({ type: 'open-app' })
  }

  if (!info.enabled || !info.spritesheetBase64) {
    return null
  }

  return (
    <div
      onPointerDown={e => {
        // Click on the transparent backdrop (not the pet/composer) dismisses
        // the composer.
        if (composerOpen && e.target === e.currentTarget) {
          setComposerOpen(false)
        }
      }}
      style={{
        alignItems: 'center',
        background: 'transparent',
        display: 'flex',
        flexDirection: 'column',
        height: '100vh',
        justifyContent: 'flex-end',
        paddingBottom: 24,
        userSelect: 'none',
        width: '100vw'
      }}
    >
      {composerOpen && (
        <input
          onChange={e => setDraft(e.target.value)}
          onKeyDown={e => {
            if (e.key === 'Enter' && !e.shiftKey) {
              e.preventDefault()
              send()
            } else if (e.key === 'Escape') {
              setComposerOpen(false)
            }
          }}
          placeholder="Message…"
          ref={inputRef}
          style={{
            background: 'var(--ui-bg-elevated)',
            border: '1px solid var(--ui-stroke-secondary)',
            borderRadius: 2,
            boxShadow: '0 6px 18px rgba(0,0,0,0.28)',
            color: 'var(--foreground)',
            fontSize: 12,
            marginBottom: 8,
            outline: 'none',
            padding: '4px 8px',
            width: 184
          }}
          value={draft}
        />
      )}

      <div
        onPointerDown={onPetPointerDown}
        onPointerMove={onPetPointerMove}
        onPointerUp={onPetPointerUp}
        ref={petRef}
        style={{
          alignItems: 'center',
          cursor: 'grab',
          display: 'flex',
          flexDirection: 'column',
          position: 'relative',
          touchAction: 'none'
        }}
      >
        <div style={{ marginBottom: 4 }}>
          <PetBubble />
        </div>
        <div style={{ lineHeight: 0, position: 'relative' }}>
          <PetSprite info={info} />

          {/* Mail icon: only when a finish landed while you were away. Jumps to
              the app's most recent thread. Anchored to the sprite (kept inside
              its box so the overlay's click-through hit-test still catches it);
              stopPropagation keeps a click from starting a window drag. */}
          {unread && (
            <button
              aria-label="Open in Hermes"
              onClick={openApp}
              onPointerDown={e => e.stopPropagation()}
              onPointerUp={e => e.stopPropagation()}
              style={{
                alignItems: 'center',
                background: 'var(--ui-bg-elevated)',
                border: '1px solid var(--ui-stroke-secondary)',
                borderRadius: 999,
                boxShadow: '0 4px 14px rgba(0,0,0,0.22)',
                color: 'var(--foreground)',
                cursor: 'pointer',
                display: 'inline-flex',
                height: 24,
                justifyContent: 'center',
                padding: 0,
                position: 'absolute',
                right: 0,
                top: 0,
                width: 24
              }}
              title="Open in Hermes"
              type="button"
            >
              <Mail style={{ height: 13, width: 13 }} />
            </button>
          )}
        </div>
      </div>
    </div>
  )
}
