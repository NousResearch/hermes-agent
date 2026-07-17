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
  computeActionCenterAlignment,
  computeActionCenterPlacement,
  overlayWindowTargetSize,
  type PetActionCenterAlignment,
  type PetActionCenterPlacement,
  type PetOverlayBounds,
  type PetOverlayMeasuredContent,
  type PetOverlayOffset,
  petOverlayTargetOffset
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
  petAnchorOffset: PetOverlayOffset | null
}

interface QueuedZoomIntent {
  anchor: PetZoomAnchor
  petAnchor: PetOverlayBounds | null
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

function currentScreenWorkArea(): PetOverlayBounds {
  const currentScreen = globalThis.screen as Screen & { availLeft?: number; availTop?: number }

  return {
    height: currentScreen.availHeight,
    width: currentScreen.availWidth,
    x: currentScreen.availLeft ?? 0,
    y: currentScreen.availTop ?? 0
  }
}

export function PetOverlayApp() {
  const { t } = useI18n()
  const info = useStore($petInfo)
  const [composerOpen, setComposerOpen] = useState(false)
  const [actionCenterOpen, setActionCenterOpen] = useState(false)
  const [actionCenterAlignment, setActionCenterAlignment] = useState<PetActionCenterAlignment>('center')
  const [actionCenterPlacement, setActionCenterPlacement] = useState<PetActionCenterPlacement>('above')
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
  const zoomPetAnchorRef = useRef<PetOverlayBounds | null>(null)
  const petRef = useRef<HTMLDivElement | null>(null)
  const actionCenterAlignmentRef = useRef<PetActionCenterAlignment>('center')
  const actionCenterPlacementRef = useRef<PetActionCenterPlacement>('above')
  const petBodyRef = useRef<HTMLDivElement | null>(null)
  const petGroupRef = useRef<HTMLDivElement | null>(null)
  const petAnchorRef = useRef<PetOverlayBounds | null>(null)
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

  const updateActionCenterLayout = useCallback(
    (placement: PetActionCenterPlacement, alignment: PetActionCenterAlignment) => {
      actionCenterPlacementRef.current = placement
      actionCenterAlignmentRef.current = alignment
      setActionCenterPlacement(placement)
      setActionCenterAlignment(alignment)
    },
    []
  )

  const chooseActionCenterLayout = useCallback(
    (petBounds: PetOverlayBounds) => {
      updateActionCenterLayout(computeActionCenterPlacement(petBounds, currentScreenWorkArea()), 'center')
    },
    [updateActionCenterLayout]
  )

  const requestOverlayBounds = useCallback(
    (bounds: PetOverlayBounds, persist: boolean, petAnchorOffset: PetOverlayOffset | null = null) => {
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

            if (petAnchorOffset && petAnchorRef.current) {
              petAnchorRef.current = {
                ...petAnchorRef.current,
                x: result.bounds.x + petAnchorOffset.x,
                y: result.bounds.y + petAnchorOffset.y
              }
            }
          }
        })
        .catch(() => {
          if (boundsActiveRef.current && requestId === boundsRequestRef.current) {
            pendingBoundsRef.current = null
          }
        })
    },
    []
  )

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

    const baseBounds = pendingBoundsRef.current ??
      actualBoundsRef.current ?? {
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
      petAnchorOffset: petAnchorRef.current
        ? { x: petAnchorRef.current.x - baseBounds.x, y: petAnchorRef.current.y - baseBounds.y }
        : null,
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
          requestOverlayBounds(bounds, false, drag.petAnchorOffset)
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
      requestOverlayBounds(requestedBounds, true, drag.petAnchorOffset)

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

      const rect = petBodyRef.current?.getBoundingClientRect()

      if (rect && rect.width > 0 && rect.height > 0) {
        const baseBounds = pendingBoundsRef.current ??
          actualBoundsRef.current ?? {
            height: window.outerHeight,
            width: window.outerWidth,
            x: window.screenX,
            y: window.screenY
          }
        const anchor = {
          height: rect.height,
          width: rect.width,
          x: baseBounds.x + rect.left,
          y: baseBounds.y + rect.top
        }
        petAnchorRef.current = anchor
        chooseActionCenterLayout(anchor)
      }
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
    const rect = petBodyRef.current?.getBoundingClientRect()
    const baseBounds = pendingBoundsRef.current ??
      actualBoundsRef.current ?? {
        height: window.outerHeight,
        width: window.outerWidth,
        x: window.screenX,
        y: window.screenY
      }
    const currentPetAnchor =
      zoomPetAnchorRef.current ??
      (rect && rect.width > 0 && rect.height > 0
        ? {
            height: rect.height,
            width: rect.width,
            x: baseBounds.x + rect.left,
            y: baseBounds.y + rect.top
          }
        : null)
    const petAnchor = currentPetAnchor
      ? {
          height: currentPetAnchor.height * anchor.ratio,
          width: currentPetAnchor.width * anchor.ratio,
          x: anchor.screenX - (anchor.screenX - currentPetAnchor.x) * anchor.ratio,
          y: anchor.screenY - (anchor.screenY - currentPetAnchor.y) * anchor.ratio
        }
      : null

    zoomPetAnchorRef.current = petAnchor
    zoomIntentsRef.current.push({ anchor, petAnchor, scale: next })
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

      const currentBounds: PetOverlayBounds = pendingBoundsRef.current ??
        actualBoundsRef.current ?? {
          height: window.outerHeight,
          width: window.outerWidth,
          x: window.screenX,
          y: window.screenY
        }
      const zoomIntents = zoomIntentsRef.current.splice(0)
      const zoomIntent = zoomIntents.at(-1)
      const targetSize = overlayWindowTargetSize(
        info.frameW ?? DEFAULT_FRAME_W,
        info.frameH ?? DEFAULT_FRAME_H,
        zoomIntent?.scale ?? info.scale ?? DEFAULT_SCALE,
        measuredContent
      )
      const petRect = petBodyRef.current?.getBoundingClientRect()
      const petGroupRect = petGroupRef.current?.getBoundingClientRect()
      const targetPetAnchor = zoomIntent?.petAnchor ?? petAnchorRef.current
      const hasPetGeometry =
        targetPetAnchor &&
        petRect &&
        petGroupRect &&
        petRect.width > 0 &&
        petRect.height > 0 &&
        petGroupRect.width > 0 &&
        petGroupRect.height > 0
      let bounds = currentBounds
      let petOffset: PetOverlayOffset | null = null

      if (
        !zoomIntent &&
        targetSize.width === currentBounds.width &&
        targetSize.height === currentBounds.height &&
        !targetPetAnchor
      ) {
        return
      }

      if (hasPetGeometry && targetPetAnchor && petRect && petGroupRect) {
        const groupBounds = {
          height: petGroupRect.height,
          width: petGroupRect.width,
          x: petGroupRect.left,
          y: petGroupRect.top
        }
        const spriteBounds = {
          height: petRect.height,
          width: petRect.width,
          x: petRect.left,
          y: petRect.top
        }
        const alignment = computeActionCenterAlignment({
          petAnchor: targetPetAnchor,
          petGroupRect: groupBounds,
          petRect: spriteBounds,
          placement: actionCenterPlacementRef.current,
          targetSize,
          workArea: currentScreenWorkArea()
        })

        if (alignment !== actionCenterAlignmentRef.current) {
          updateActionCenterLayout(actionCenterPlacementRef.current, alignment)
        }

        petOffset = petOverlayTargetOffset({
          alignment,
          paddingBottom: PET_PADDING_BOTTOM,
          petGroupRect: groupBounds,
          petRect: spriteBounds,
          placement: actionCenterPlacementRef.current,
          targetSize
        })
        petAnchorRef.current = targetPetAnchor
        bounds = anchoredOverlayBounds({
          currentBounds,
          paddingBottom: PET_PADDING_BOTTOM,
          targetSize,
          petAnchor: targetPetAnchor,
          petOffset,
          placement: actionCenterPlacementRef.current
        })
      } else if (zoomIntents.length > 0) {
        for (const intent of zoomIntents) {
          const intentTargetSize = overlayWindowTargetSize(
            info.frameW ?? DEFAULT_FRAME_W,
            info.frameH ?? DEFAULT_FRAME_H,
            intent.scale,
            measuredContent
          )

          bounds = anchoredOverlayBounds({
            currentBounds: bounds,
            paddingBottom: PET_PADDING_BOTTOM,
            targetSize: intentTargetSize,
            wheelAnchor: {
              ...intent.anchor,
              clientX: intent.anchor.screenX - bounds.x,
              clientY: intent.anchor.screenY - bounds.y
            }
          })
        }
      } else {
        bounds = anchoredOverlayBounds({
          currentBounds,
          paddingBottom: PET_PADDING_BOTTOM,
          targetSize,
          petAnchor: targetPetAnchor,
          placement: actionCenterPlacementRef.current
        })
      }

      zoomPetAnchorRef.current = null
      requestOverlayBounds(bounds, true, petOffset)
    })

    return () => {
      if (resizeRafRef.current !== null) {
        cancelAnimationFrame(resizeRafRef.current)
        resizeRafRef.current = null
      }
    }
  }, [
    actionCenterAlignment,
    actionCenterOpen,
    actionCenterPlacement,
    info.enabled,
    info.spritesheetBase64,
    info.scale,
    info.frameW,
    info.frameH,
    measuredContent,
    requestOverlayBounds,
    updateActionCenterLayout
  ])

  if (!info.enabled || !info.spritesheetBase64) {
    return null
  }

  const horizontalPlacement = actionCenterPlacement === 'left' || actionCenterPlacement === 'right'
  const crossAxisAlignment =
    actionCenterAlignment === 'start' ? 'flex-start' : actionCenterAlignment === 'end' ? 'flex-end' : 'center'
  const outerAlignItems = horizontalPlacement
    ? actionCenterPlacement === 'left'
      ? 'flex-end'
      : 'flex-start'
    : crossAxisAlignment
  const outerJustifyContent = horizontalPlacement
    ? crossAxisAlignment
    : actionCenterPlacement === 'below'
      ? 'flex-start'
      : 'flex-end'
  const interactiveDirection =
    actionCenterPlacement === 'below'
      ? 'column-reverse'
      : actionCenterPlacement === 'left'
        ? 'row'
        : actionCenterPlacement === 'right'
          ? 'row-reverse'
          : 'column'

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
        alignItems: outerAlignItems,
        background: 'transparent',
        display: 'flex',
        flexDirection: 'column',
        height: '100vh',
        justifyContent: outerJustifyContent,
        paddingBottom: actionCenterPlacement === 'above' ? PET_PADDING_BOTTOM : 0,
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
        data-pet-overlay-alignment={actionCenterAlignment}
        data-pet-overlay-interactive-root=""
        data-pet-overlay-placement={actionCenterPlacement}
        onPointerDown={onPetPointerDown}
        onPointerMove={onPetPointerMove}
        onPointerUp={onPetPointerUp}
        ref={petRef}
        style={{
          alignItems: crossAxisAlignment,
          cursor: 'grab',
          display: 'flex',
          flexDirection: interactiveDirection,
          flexShrink: actionCenterOpen ? 0 : 1,
          position: 'relative',
          touchAction: 'none'
        }}
      >
        <PetActionCenter onOpenChange={onActionCenterOpenChange} state={actionCenterState} />
        <div
          data-pet-overlay-group=""
          ref={petGroupRef}
          style={{ alignItems: 'center', display: 'flex', flexDirection: 'column', flexShrink: 0 }}
        >
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
          <div data-pet-overlay-body="" ref={petBodyRef} style={{ lineHeight: 0, position: 'relative' }}>
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
    </div>
  )
}
