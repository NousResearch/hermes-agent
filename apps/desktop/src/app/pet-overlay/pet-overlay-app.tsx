import { useStore } from '@nanostores/react'
import { useCallback, useEffect, useRef, useState } from 'react'

import { PetHeartField, playVibeHearts } from '@/components/chat/vibe-hearts'
import { PetBubble } from '@/components/pet/pet-bubble'
import { PetSprite } from '@/components/pet/pet-sprite'
import { type PetZoomAnchor, usePetZoomGesture } from '@/components/pet/use-pet-zoom-gesture'
import { useI18n } from '@/i18n'
import { Mail } from '@/lib/icons'
import { $petActivity, $petInfo, setPetInfo } from '@/store/pet'
import type { PetActionCenterState } from '@/store/pet-action-center'
import {
  anchoredOverlayBounds,
  overlayWindowTargetSize,
  type PetOverlayBounds,
  type PetOverlayMeasuredContent
} from '@/store/pet-overlay'
import { setAwaitingResponse, setBusy } from '@/store/session'

import { PetActionCenter } from './pet-action-center'

// Fallbacks mirror pet-sprite's defaults; the gateway normally sends real values.
const DEFAULT_FRAME_W = 192
const DEFAULT_FRAME_H = 208
const DEFAULT_SCALE = 0.33

const EMPTY_ACTION_CENTER_STATE: PetActionCenterState = {
  action: null,
  actionableCount: 0,
  attentionCount: 0,
  blockingCount: 0,
  items: [],
  secureInputCount: 0,
  selectedItemId: null
}

// Must match the root's paddingBottom — the sprite renders bottom-centered, this
// many px above the window's bottom edge. Used to anchor the resize.
const PET_PADDING_BOTTOM = 24

// A sprite pixel counts as "solid" (interactive) at/above this alpha (0-255).
// Low enough to catch anti-aliased edges, high enough that the faint halo around
// the art still clicks through.
const ALPHA_HIT_THRESHOLD = 16

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

interface QueuedZoomIntent {
  anchor: PetZoomAnchor
  scale: number
}

function parsedOverlayBounds(value: unknown): PetOverlayBounds | null {
  if (typeof value !== 'object' || value === null) {
    return null
  }

  const bounds = value as Partial<PetOverlayBounds>

  if (
    !Number.isFinite(bounds.x) ||
    !Number.isFinite(bounds.y) ||
    !Number.isFinite(bounds.width) ||
    !Number.isFinite(bounds.height) ||
    (bounds.width ?? 0) <= 0 ||
    (bounds.height ?? 0) <= 0
  ) {
    return null
  }

  return bounds as PetOverlayBounds
}

export function PetOverlayApp() {
  const { t } = useI18n()
  const info = useStore($petInfo)
  const [composerOpen, setComposerOpen] = useState(false)
  const [actionCenterOpen, setActionCenterOpen] = useState(false)
  const [actionCenterState, setActionCenterState] = useState<PetActionCenterState>(EMPTY_ACTION_CENTER_STATE)
  const [measuredContent, setMeasuredContent] = useState<PetOverlayMeasuredContent | null>(null)
  const [draft, setDraft] = useState('')
  // Mirrored from the main renderer: a finish landed while you were away.
  const [unread, setUnread] = useState(false)

  const dragRef = useRef<DragState | null>(null)
  const dragBoundsRef = useRef<PetOverlayBounds | null>(null)
  const dragRafRef = useRef<number | null>(null)
  // Preserve every wheel step until the resize frame consumes it. Each intent
  // carries an absolute screen cursor so it can be rebased after a pending
  // native resize moves the window underneath the gesture.
  const zoomIntentsRef = useRef<QueuedZoomIntent[]>([])
  const petRef = useRef<HTMLDivElement | null>(null)
  const inputRef = useRef<HTMLInputElement | null>(null)
  // Last mirrored reaction id — a bump means the main window fired a reaction.
  const lastReactionRef = useRef<number | null>(null)
  const ignoreRef = useRef(true)
  const keyboardInteractiveRef = useRef(false)
  const actionCenterOpenRef = useRef(false)
  const clickTimerRef = useRef<ReturnType<typeof setTimeout> | undefined>(undefined)
  const boundsRequestRef = useRef(0)

  // Last actual window bounds returned by the main process (clamped + applied).
  // The resize effect reads from here instead of `window.outerWidth/Height/
  // screenX/Y`, which can be stale for one or more frames after a setBounds
  // call — especially on macOS where the OS applies bounds asynchronously.
  // Without this, a rapid open/close of the action center can race: the second
  // resize reads pre-resize DOM geometry and anchors the pet off-position.
  const actualBoundsRef = useRef<PetOverlayBounds | null>(null)
  // Tracks the newest request until the main process confirms it. A content
  // change can return to the confirmed size while a grow request is still in
  // flight; without this, the grow may land after the close and leave the
  // overlay enlarged because the close looked like a local no-op.
  const pendingBoundsRef = useRef<PetOverlayBounds | null>(null)
  const boundsActiveRef = useRef(true)
  // Pending rAF for the resize effect, so rapid content changes (panel open
  // → items pushed → nested layer) coalesce into one setBounds call.
  const resizeRafRef = useRef<number | null>(null)

  const requestOverlayBounds = useCallback((bounds: PetOverlayBounds, persist: boolean) => {
    const api = window.hermesDesktop?.petOverlay

    if (!api || !boundsActiveRef.current) {
      return
    }

    const requestId = ++boundsRequestRef.current
    pendingBoundsRef.current = bounds

    void api
      .setBounds(bounds)
      .then(result => {
        if (!boundsActiveRef.current || requestId !== boundsRequestRef.current) {
          return
        }

        pendingBoundsRef.current = null

        if (result.ok && result.bounds) {
          actualBoundsRef.current = result.bounds

          if (persist) {
            api.control({ bounds: result.bounds, type: 'bounds' })
          }
        }
      })
      .catch(() => {
        if (boundsActiveRef.current && requestId === boundsRequestRef.current) {
          pendingBoundsRef.current = null
        }
      })
  }, [])

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
      setActionCenterState(payload.actionCenter)

      // Play a reaction on a new id (ignore the first sync, which just primes it).
      const reaction = payload.reaction ?? null

      if (lastReactionRef.current === null) {
        lastReactionRef.current = reaction?.id ?? 0
      } else if (reaction && reaction.id > lastReactionRef.current) {
        lastReactionRef.current = reaction.id

        if (reaction.kind === 'vibe') {
          playVibeHearts()
        }
      }
    })

    // Tell the main renderer we're mounted so it pushes the current frame (the
    // subscribe-time pushes during open() can land before this view exists).
    window.hermesDesktop?.petOverlay?.control({ type: 'ready' })

    return off
  }, [])

  // Electron can move/reclamp the native window independently when display
  // topology changes. Keep the renderer's confirmed basis synchronized, and
  // invalidate every outstanding response when the overlay is disabled or
  // unmounted so stale promises cannot persist old geometry.
  useEffect(() => {
    const active = Boolean(info.enabled && info.spritesheetBase64)
    const api = window.hermesDesktop?.petOverlay
    boundsActiveRef.current = active

    const offBounds = active
      ? api?.onBounds(payload => {
          const bounds = parsedOverlayBounds(payload)

          if (bounds) {
            actualBoundsRef.current = bounds
          }
        })
      : undefined

    if (!active) {
      actualBoundsRef.current = null
      pendingBoundsRef.current = null
      zoomIntentsRef.current = []
      dragBoundsRef.current = null
    }

    return () => {
      boundsActiveRef.current = false
      boundsRequestRef.current += 1
      pendingBoundsRef.current = null
      zoomIntentsRef.current = []
      dragBoundsRef.current = null

      if (resizeRafRef.current !== null) {
        cancelAnimationFrame(resizeRafRef.current)
        resizeRafRef.current = null
      }

      if (dragRafRef.current !== null) {
        cancelAnimationFrame(dragRafRef.current)
        dragRafRef.current = null
      }

      offBounds?.()
    }
  }, [info.enabled, info.spritesheetBase64])

  // Click-through: make only the *solid* sprite pixels (plus the bubble / mail
  // button / open composer) interactive — clicks on the transparent rectangle
  // around the art pass through to whatever's behind. With ignore+forward, the
  // renderer still receives mousemove so we can re-arm the moment the cursor
  // returns to a solid pixel.
  useEffect(() => {
    setIgnore(true)

    // True when the point sits on a solid sprite pixel or on the pet's other
    // interactive chrome (bubble, mail button). Over the canvas we sample the
    // rendered alpha; elsewhere inside the pet (bubble/button) we trust DOM
    // hit-testing. Anything else is transparent backdrop.
    const isInteractiveAt = (x: number, y: number): boolean => {
      const pet = petRef.current
      const target = document.elementFromPoint(x, y)

      if (!pet || !target || !pet.contains(target)) {
        return false
      }

      if (!(target instanceof HTMLCanvasElement)) {
        return true
      }

      const rect = target.getBoundingClientRect()

      if (rect.width === 0 || rect.height === 0) {
        return true
      }

      const ctx = target.getContext('2d')

      if (!ctx) {
        return true
      }

      const px = Math.floor((x - rect.left) * (target.width / rect.width))
      const py = Math.floor((y - rect.top) * (target.height / rect.height))

      try {
        return ctx.getImageData(px, py, 1, 1).data[3] >= ALPHA_HIT_THRESHOLD
      } catch {
        // Tainted/zero-size read — fail open so the pet stays grabbable.
        return true
      }
    }

    const onMove = (ev: MouseEvent) => {
      if (dragRef.current || keyboardInteractiveRef.current) {
        setIgnore(false)

        return
      }

      setIgnore(!isInteractiveAt(ev.clientX, ev.clientY))
    }

    window.addEventListener('mousemove', onMove)

    return () => {
      window.removeEventListener('mousemove', onMove)
      clearTimeout(clickTimerRef.current)
    }
  }, [])

  // Keep the whole window interactive while either keyboard surface is open.
  // Incoming action-center state only renders its trigger; the explicit trigger
  // click is the sole path that flips actionCenterOpen and makes the window key.
  useEffect(() => {
    const keyboardInteractive = composerOpen || actionCenterOpen
    keyboardInteractiveRef.current = keyboardInteractive
    actionCenterOpenRef.current = actionCenterOpen

    window.hermesDesktop?.petOverlay?.setFocusable(keyboardInteractive)

    if (keyboardInteractive) {
      setIgnore(false)
    }

    if (composerOpen) {
      // The OS window has to become key first (setFocusable + focus happen in
      // the main process), so focus the input on the next frame.
      const frameId = requestAnimationFrame(() => inputRef.current?.focus())

      return () => cancelAnimationFrame(frameId)
    }
  }, [composerOpen, actionCenterOpen])

  // Observe the complete interactive stack (ActionCenter + Bubble + Sprite),
  // not the compact sprite alone. ResizeObserver is optional: older/limited
  // Chromium builds safely remain at compact geometry.
  useEffect(() => {
    const target = petRef.current
    const ResizeObserverCtor = globalThis.ResizeObserver

    if (!target || typeof ResizeObserverCtor !== 'function') {
      setMeasuredContent(null)

      return
    }

    const observer = new ResizeObserverCtor(entries => {
      const rect = entries[0]?.contentRect
      const width = Number.isFinite(rect?.width) && (rect?.width ?? 0) > 0 ? Math.round(rect!.width) : 0
      const height = Number.isFinite(rect?.height) && (rect?.height ?? 0) > 0 ? Math.round(rect!.height) : 0

      setMeasuredContent(current =>
        current?.width === width && current.height === height ? current : { height, width }
      )
    })

    observer.observe(target)

    return () => observer.disconnect()
  }, [info.enabled, info.spritesheetBase64])

  const onPetPointerDown = (e: React.PointerEvent) => {
    if (e.button !== 0) {
      return
    }

    const baseBounds = pendingBoundsRef.current ?? actualBoundsRef.current ?? {
      height: window.outerHeight,
      width: window.outerWidth,
      x: window.screenX,
      y: window.screenY
    }

    ;(e.target as Element).setPointerCapture?.(e.pointerId)
    dragRef.current = {
      height: baseBounds.height,
      moved: false,
      offX: e.screenX - baseBounds.x,
      offY: e.screenY - baseBounds.y,
      startX: e.screenX,
      startY: e.screenY,
      width: baseBounds.width
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

    dragBoundsRef.current = {
      height: drag.height,
      width: drag.width,
      x: e.screenX - drag.offX,
      y: e.screenY - drag.offY
    }

    if (dragRafRef.current === null) {
      dragRafRef.current = requestAnimationFrame(() => {
        dragRafRef.current = null
        const bounds = dragBoundsRef.current
        dragBoundsRef.current = null

        if (bounds) {
          requestOverlayBounds(bounds, false)
        }
      })
    }
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
      const requestedBounds = {
        height: drag.height,
        width: drag.width,
        x: e.screenX - drag.offX,
        y: e.screenY - drag.offY
      }

      if (dragRafRef.current !== null) {
        cancelAnimationFrame(dragRafRef.current)
        dragRafRef.current = null
      }

      dragBoundsRef.current = null
      requestOverlayBounds(requestedBounds, true)

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

    const suppressComposer = actionCenterOpenRef.current

    clickTimerRef.current = setTimeout(() => {
      clickTimerRef.current = undefined

      if (!suppressComposer && !actionCenterOpenRef.current) {
        setComposerOpen(open => !open)
      }
    }, DOUBLE_CLICK_MS)
  }

  const onActionCenterOpenChange = (open: boolean) => {
    actionCenterOpenRef.current = open

    if (open) {
      clearTimeout(clickTimerRef.current)
      clickTimerRef.current = undefined
      setComposerOpen(false)
    }

    setActionCenterOpen(open)
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

  // Alt+wheel over the popped-out pet resizes it. The overlay has no gateway,
  // so paint the new scale locally for instant feedback, then ask the main
  // renderer to persist it (it pushes the reconciled scale back). Stash the
  // cursor anchor for the resize effect; the window itself is grown to fit there.
  const onScale = useCallback((next: number, anchor: PetZoomAnchor) => {
    zoomIntentsRef.current.push({ anchor, scale: next })
    setPetInfo({ ...$petInfo.get(), scale: next })
    window.hermesDesktop?.petOverlay?.control({ scale: next, type: 'scale' })
  }, [])

  usePetZoomGesture(petRef, onScale, Boolean(info.enabled && info.spritesheetBase64))

  // One resize pipeline owns scale and content measurement. The compact pet
  // size is the floor; an open action center can only grow it, so a scale push
  // can never race the observer and shrink around live content.
  //
  // Reads actual window bounds from `actualBoundsRef` (updated by every
  // setBounds response) instead of `window.outer*` / `window.screen*`, which
  // lag the OS by one+ frame on macOS. A rAF coalesces rapid content changes
  // (panel open → items → nested layer) into a single setBounds.
  useEffect(() => {
    if (!info.enabled || !info.spritesheetBase64) {
      return
    }

    if (resizeRafRef.current !== null) {
      cancelAnimationFrame(resizeRafRef.current)
    }

    resizeRafRef.current = requestAnimationFrame(() => {
      resizeRafRef.current = null

      const currentBounds: PetOverlayBounds = pendingBoundsRef.current ?? actualBoundsRef.current ?? {
        height: window.outerHeight,
        width: window.outerWidth,
        x: window.screenX,
        y: window.screenY
      }
      const zoomIntents = zoomIntentsRef.current.splice(0)
      let bounds = currentBounds

      if (zoomIntents.length > 0) {
        for (const intent of zoomIntents) {
          const targetSize = overlayWindowTargetSize(
            info.frameW ?? DEFAULT_FRAME_W,
            info.frameH ?? DEFAULT_FRAME_H,
            intent.scale,
            measuredContent
          )

          bounds = anchoredOverlayBounds({
            currentBounds: bounds,
            paddingBottom: PET_PADDING_BOTTOM,
            targetSize,
            wheelAnchor: {
              ...intent.anchor,
              clientX: intent.anchor.screenX - bounds.x,
              clientY: intent.anchor.screenY - bounds.y
            }
          })
        }
      } else {
        const targetSize = overlayWindowTargetSize(
          info.frameW ?? DEFAULT_FRAME_W,
          info.frameH ?? DEFAULT_FRAME_H,
          info.scale ?? DEFAULT_SCALE,
          measuredContent
        )

        if (targetSize.width === currentBounds.width && targetSize.height === currentBounds.height) {
          return
        }

        bounds = anchoredOverlayBounds({
          currentBounds,
          paddingBottom: PET_PADDING_BOTTOM,
          targetSize
        })
      }

      requestOverlayBounds(bounds, true)
    })

    return () => {
      if (resizeRafRef.current !== null) {
        cancelAnimationFrame(resizeRafRef.current)
        resizeRafRef.current = null
      }
    }
  }, [info.enabled, info.spritesheetBase64, info.scale, info.frameW, info.frameH, measuredContent, requestOverlayBounds])

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
        paddingBottom: PET_PADDING_BOTTOM,
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
          placeholder={t.pet.composerPlaceholder}
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
        data-pet-overlay-interactive-root=""
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
        <PetActionCenter onOpenChange={onActionCenterOpenChange} state={actionCenterState} />
        {/* The action center already owns live status while open. Keeping the
            separate speech bubble mounted there makes transient navigation
            state add/remove a second row, which forces the native overlay to
            grow, shrink, then grow again around the dialog. Suppress that
            duplicate surface so action-center geometry changes monotonically. */}
        {!actionCenterOpen && (
          <div style={{ marginBottom: 4 }}>
            <PetBubble />
          </div>
        )}
        <div style={{ lineHeight: 0, position: 'relative' }}>
          <PetSprite info={info} />

          {/* Hearts on the popped-out pet — identical to in-window. */}
          <PetHeartField
            petH={(info.frameH ?? DEFAULT_FRAME_H) * (info.scale ?? DEFAULT_SCALE)}
            petW={(info.frameW ?? DEFAULT_FRAME_W) * (info.scale ?? DEFAULT_SCALE)}
          />

          {/* Mail icon: only when a finish landed while you were away. Jumps to
              the app's most recent thread. Anchored to the sprite (kept inside
              its box so the overlay's click-through hit-test still catches it);
              stopPropagation keeps a click from starting a window drag. */}
          {unread && (
            <button
              aria-label={t.pet.actionCenter.openInApp}
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
              title={t.pet.actionCenter.openInApp}
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
