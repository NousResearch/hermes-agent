import { useStore } from '@nanostores/react'
import { useCallback, useEffect, useRef, useState } from 'react'

import { useGatewayRequest } from '@/app/gateway/hooks/use-gateway-request'
import { useOnProfileSwitch } from '@/app/hooks/use-on-profile-switch'
import { useRouteOverlayActive } from '@/app/hooks/use-route-overlay-active'
import { CompanionHeartField } from '@/components/chat/vibe-hearts'
import { CompanionBubble } from '@/components/companion/companion-bubble'
import { persistString, storedString } from '@/lib/storage'
import {
  $companionAtRest,
  $companionInfo,
  $companionRoam,
  $companionRoamDir,
  clearCompanionUnread,
  type CompanionInfo,
  type CompanionInfoMeta,
  companionProfile,
  hasCompanionSpriteForMeta,
  mergeCompanionInfoMeta,
  setCompanionInfo
} from '@/store/companion'
import { resetCompanionGallery, setCompanionScale } from '@/store/companion-gallery'
import { $companionOverlayActive, initCompanionOverlayBridge, popOutCompanion, restoreCompanionOverlay } from '@/store/companion-overlay'
import { $changeEventsAvailable, $companionChange } from '@/store/live-sync'
import { $gatewayState } from '@/store/session'
import { isSecondaryWindow } from '@/store/windows'
import { useTheme } from '@/themes/context'

import { COMPANION_STARTUP_RETRY_MS, companionInfoPollIntervalMs } from './companion-info-poll'
import { CompanionSprite, roamWalkRow } from './companion-sprite'
import { useCompanionRoam } from './use-companion-roam'
import { type CompanionZoomAnchor, useCompanionZoomGesture } from './use-companion-zoom-gesture'

// v2: positions are now top/left anchored (v1 stored bottom-anchored values,
// which dragged inverted). Bumping the key discards stale v1 coordinates.
const POSITION_KEY = 'hermes.desktop.companion-position.v2'

// Stand-in companion size for the pre-load clamp (real size flows in with `info`).
const NOMINAL_COMPANION_PX = 96

interface Point {
  x: number
  y: number
}

// Keep a w×h box fully inside the viewport. Pre-companion-load callers pass a nominal
// size; the live size flows in once `info` arrives.
function clampPoint(x: number, y: number, w: number, h: number): Point {
  return {
    x: Math.min(Math.max(0, x), Math.max(0, (window.innerWidth || 800) - w)),
    y: Math.min(Math.max(0, y), Math.max(0, (window.innerHeight || 600) - h))
  }
}

// The sprite art faces left by default, so mirror it when the companion's center sits
// on the left half of the window — it always faces inward, toward the content.
function facing(leftX: number, companionW: number): string {
  return leftX + companionW / 2 < (window.innerWidth || 800) / 2 ? 'scaleX(-1)' : 'none'
}

function loadPosition(): Point {
  try {
    const raw = storedString(POSITION_KEY)

    if (raw) {
      const parsed = JSON.parse(raw) as Point

      if (typeof parsed.x === 'number' && typeof parsed.y === 'number') {
        return clampPoint(parsed.x, parsed.y, NOMINAL_COMPANION_PX, NOMINAL_COMPANION_PX)
      }
    }
  } catch {
    // fall through to default
  }

  // Default: lower-left corner (top/left anchored).
  return clampPoint(24, (window.innerHeight || 600) - 220, NOMINAL_COMPANION_PX, NOMINAL_COMPANION_PX)
}

/**
 * In-window floating companiondex mascot. Always-on-top within the app, draggable,
 * and reactive to agent activity via `$companionState`. Fetches the active companion via
 * the shared `companion.info` RPC; renders nothing until a companion is installed +
 * enabled.
 *
 * Adopting a companion is fully in-app: type `/companion boba` in the composer. That
 * writes `display.companion.*` from the slash worker, so we keep polling `companion.info`
 * while no companion is active and the mascot pops in within a few seconds — no
 * reload, no CLI. Once a companion is live we still refresh more slowly so generated
 * companions rewritten on disk (or renamed/rebuilt by the hatch flow) repaint without
 * restarting the app.
 *
 * Event-capable backends also drive refreshes via `companion.changed`, but a slow
 * backstop poll stays in place: the watcher seeds the companion signature silently
 * at gateway boot and only broadcasts when it *moves*, and the one-shot
 * connect pull can race a still-warming `companion.info` (fail-open enabled:false).
 * Without the backstop the mascot stays hidden until Settings re-seeds it.
 *
 * Promotion to a separate frameless OS-level window is a follow-up — the
 * sprite + state logic here is reused as-is, only the host changes.
 */
export function FloatingCompanion() {
  const { requestGateway } = useGatewayRequest()
  const { resolvedMode } = useTheme()
  const gatewayState = useStore($gatewayState)
  const info = useStore($companionInfo)
  const changeEventsAvailable = useStore($changeEventsAvailable)
  const companionChange = useStore($companionChange)
  const overlayActive = useStore($companionOverlayActive)
  const roamEnabled = useStore($companionRoam)
  const atRest = useStore($companionAtRest)
  const roamDir = useStore($companionRoamDir)
  const routeOverlayOpen = useRouteOverlayActive()

  const [position, setPosition] = useState<Point>(loadPosition)
  const containerRef = useRef<HTMLDivElement | null>(null)
  // The facing mirror lives on the sprite wrapper, not the container, so the
  // speech bubble (a container child) never renders flipped/backwards.
  const spriteWrapRef = useRef<HTMLDivElement | null>(null)
  const companionW = (info.frameW ?? 192) * (info.scale ?? 0.33)
  const companionH = (info.frameH ?? 208) * (info.scale ?? 0.33)
  // Soft contact shadow, sized off the companion so every scale/species grounds the
  // same way (cf. lairp's per-actor feet ellipse). Lighter on light backgrounds.
  const shadowW = Math.round(companionW * 0.55)
  const shadowH = Math.max(3, Math.round(shadowW * 0.28))
  const shadowAlpha = resolvedMode === 'light' ? 0.2 : 0.55
  // Live drag offset (pointer → element top-left). Drag updates the DOM
  // directly to avoid a React re-render (and canvas reflow) per pointermove —
  // state is only committed on release.
  const dragRef = useRef<{ dx: number; dy: number; x: number; y: number } | null>(null)

  // Keep the *whole* companion on-screen at its current size, so growing it near an
  // edge can't leave the window cropping it. Shared by drag + the reclamp effect.
  const clamp = useCallback(({ x, y }: Point): Point => clampPoint(x, y, companionW, companionH), [companionW, companionH])

  // Fetch companion.info on connect. companion.changed re-runs this effect when the
  // signature moves; a slow backstop covers silent seed + cold-start races.
  // Older backends (no change_events) keep the legacy fast-while-inactive poll.
  const active = info.enabled && Boolean(info.spritesheetBase64)
  useEffect(() => {
    if (gatewayState !== 'open') {
      return
    }

    let cancelled = false

    // companion.changed already carries the meta payload — an enabled=false
    // broadcast clears the mascot with zero round-trips, and an unchanged
    // revision (scale-only move still changes the sig) short-circuits below
    // via hasCompanionSpriteForMeta + mergeCompanionInfoMeta.
    if (changeEventsAvailable && companionChange.tick > 0 && companionChange.meta?.enabled === false) {
      setCompanionInfo({ enabled: false })

      return
    }

    const pull = async () => {
      try {
        if (active) {
          try {
            const meta = await requestGateway<CompanionInfoMeta>('companion.info.meta', { profile: companionProfile() })

            if (cancelled || !meta) {
              return
            }

            if (!meta.enabled) {
              setCompanionInfo({ enabled: false })

              return
            }

            const current = $companionInfo.get()

            if (hasCompanionSpriteForMeta(current, meta)) {
              const merged = mergeCompanionInfoMeta(current, meta)

              if (merged !== current) {
                setCompanionInfo(merged)
              }

              return
            }
          } catch {
            // Older gateways may not have companion.info.meta yet; fall back to companion.info.
          }
        }

        // Send-once semantics (#54730): tell the gateway which spritesheet
        // revision we already hold so an unchanged multi-MB sheet is not
        // re-sent over the WebSocket on every backstop refresh.
        const held = $companionInfo.get()
        const knownRevision = held.enabled && held.spritesheetBase64 ? held.spritesheetRevision : undefined

        const next = await requestGateway<CompanionInfo & { spritesheetUnchanged?: boolean }>('companion.info', {
          knownRevision,
          profile: companionProfile()
        })

        if (!cancelled && next) {
          const current = $companionInfo.get()

          if (next.enabled && next.spritesheetUnchanged && !next.spritesheetBase64) {
            // Gateway confirmed our held sheet is current; keep the bytes.
            next.spritesheetBase64 = current.spritesheetBase64
          }

          if (
            next.enabled &&
            current.enabled &&
            current.slug === next.slug &&
            current.displayName === next.displayName &&
            current.scale === next.scale &&
            current.spritesheetRevision &&
            current.spritesheetRevision === next.spritesheetRevision
          ) {
            return
          }

          setCompanionInfo(next)
        }
      } catch {
        // cosmetic feature — never surface gateway errors
      }
    }

    const pullIfVisible = () => {
      if (document.visibilityState === 'visible') {
        void pull()
      }
    }

    void pull()
    window.addEventListener('focus', pull)

    // Cover the cold-start race where the first pull hit fail-open enabled:false
    // before the companion store was warm. Skip further retries once the mascot is live.
    const startupRetryTimers = COMPANION_STARTUP_RETRY_MS.map(delay =>
      window.setTimeout(() => {
        if (cancelled) {
          return
        }

        const current = $companionInfo.get()

        if (current.enabled && current.spritesheetBase64) {
          return
        }

        pullIfVisible()
      }, delay)
    )

    // Always keep a timer. Event-capable backends use the slow backstop (same
    // contract as cron/sessions in use-background-sync); legacy keeps the
    // historical fast-while-inactive cadence.
    const timer = window.setInterval(pullIfVisible, companionInfoPollIntervalMs(changeEventsAvailable, active))

    return () => {
      cancelled = true
      window.removeEventListener('focus', pull)

      for (const id of startupRetryTimers) {
        window.clearTimeout(id)
      }

      window.clearInterval(timer)
    }
  }, [gatewayState, active, changeEventsAvailable, companionChange, requestGateway])

  // Companions are per-profile. When the active profile changes, drop the previous
  // profile's mascot + gallery cache so the poll above refetches the new
  // profile's companion (its config + companions dir resolve per-profile on the backend).
  useOnProfileSwitch(() => {
    setCompanionInfo({ enabled: false })
    resetCompanionGallery()
  })

  // Wire the overlay control channel once, only in the primary window — the
  // pop-out overlay belongs to it (main.ts positions it against the main
  // window and routes control messages back to it).
  useEffect(() => {
    if (isSecondaryWindow()) {
      return
    }

    return initCompanionOverlayBridge()
  }, [])

  // Returning to the app (by any route, not just the mail icon) clears the companion's
  // "new message" hint — you've seen it now.
  useEffect(() => {
    if (isSecondaryWindow()) {
      return
    }

    const onFocus = () => clearCompanionUnread()
    window.addEventListener('focus', onFocus)

    return () => window.removeEventListener('focus', onFocus)
  }, [])

  // Restore a popped-out companion on boot, once the companion has loaded (so we never spawn
  // an empty overlay window). Primary window only; runs at most once.
  const restoredRef = useRef(false)
  // eslint-disable-next-line no-restricted-syntax -- legitimate non-atom ref write (see eslint rule comment)
  useEffect(() => {
    if (isSecondaryWindow() || restoredRef.current || !active) {
      return
    }

    restoredRef.current = true
    restoreCompanionOverlay()
  }, [active])

  // Never strand or crop the companion: re-clamp (and persist) whenever the viewport
  // shrinks or the companion's own size changes (wheel/slider). `clamp` carries the
  // current size, so depending on it covers both triggers.
  useEffect(() => {
    const reclamp = () =>
      setPosition(prev => {
        const next = clamp(prev)

        if (next.x === prev.x && next.y === prev.y) {
          return prev
        }

        persistString(POSITION_KEY, JSON.stringify(next))

        return next
      })

    reclamp()
    window.addEventListener('resize', reclamp)

    return () => window.removeEventListener('resize', reclamp)
  }, [clamp])

  const onPointerDown = useCallback((e: React.PointerEvent) => {
    const el = containerRef.current

    if (!el) {
      return
    }

    const rect = el.getBoundingClientRect()

    // Shift-click pops the companion out into a free-floating desktop overlay (it can
    // leave the window and stays visible while Hermes is minimized) instead of
    // starting an in-window drag. Primary window only — the overlay is anchored
    // to it.
    if (e.shiftKey && !isSecondaryWindow()) {
      popOutCompanion({ height: rect.height, width: rect.width, x: rect.left, y: rect.top })

      return
    }

    dragRef.current = { dx: e.clientX - rect.left, dy: e.clientY - rect.top, x: rect.left, y: rect.top }
    el.setPointerCapture(e.pointerId)
    el.style.cursor = 'grabbing'
  }, [])

  const onPointerMove = useCallback(
    (e: React.PointerEvent) => {
      const drag = dragRef.current
      const el = containerRef.current

      if (!drag || !el) {
        return
      }

      const next = clamp({ x: e.clientX - drag.dx, y: e.clientY - drag.dy })
      drag.x = next.x
      drag.y = next.y
      // Mutate the DOM directly — no setState, so no re-render while dragging. The
      // mirror follows the pointer across the midline for the same reason; it
      // rides the sprite wrapper so the bubble stays upright.
      el.style.left = `${next.x}px`
      el.style.top = `${next.y}px`

      if (spriteWrapRef.current) {
        spriteWrapRef.current.style.transform = facing(next.x, companionW)
      }
    },
    [clamp, companionW]
  )

  const onPointerUp = useCallback((e: React.PointerEvent) => {
    const drag = dragRef.current

    if (drag) {
      dragRef.current = null
      const committed = { x: drag.x, y: drag.y }
      setPosition(committed)
      persistString(POSITION_KEY, JSON.stringify(committed))
    }

    const el = containerRef.current

    if (el) {
      el.style.cursor = 'grab'
      el.releasePointerCapture?.(e.pointerId)
    }
  }, [])

  // Alt+wheel over the companion resizes it (persisted via the same path as the
  // settings slider). Zoom toward the cursor — shift the top-left so the pixel
  // under the pointer stays put — so the companion grows in place instead of running
  // off. The reclamp effect (via `clamp`) still guarantees it stays on-screen.
  const onScale = useCallback(
    (next: number, { clientX, clientY, ratio }: CompanionZoomAnchor) => {
      setCompanionScale(requestGateway, next)
      setPosition(prev => {
        const at = clampPoint(
          clientX - (clientX - prev.x) * ratio,
          clientY - (clientY - prev.y) * ratio,
          (info.frameW ?? 192) * next,
          (info.frameH ?? 208) * next
        )

        persistString(POSITION_KEY, JSON.stringify(at))

        return at
      })
    },
    [requestGateway, info.frameW, info.frameH]
  )

  useCompanionZoomGesture(containerRef, onScale, active && !overlayActive)

  // Commit a roamed-to position back to React state + storage when the wander
  // loop settles, so the inline style matches the DOM once the loop stops
  // driving it imperatively. Stable identity keeps the roam effect from
  // restarting every render.
  const commitRoamPosition = useCallback((point: Point) => {
    setPosition(point)
    persistString(POSITION_KEY, JSON.stringify(point))
  }, [])

  const isDragging = useCallback(() => dragRef.current !== null, [])

  // Roam only the in-window companion, only while it's idle (agent at rest) and not
  // popped out into the OS overlay. Activity pauses the wander; the companion reacts
  // in place, then resumes strolling when the turn ends.
  useCompanionRoam({
    commit: commitRoamPosition,
    containerRef,
    enabled: roamEnabled && active && !overlayActive && atRest,
    isInteracting: isDragging,
    loopMs: info.loopMs ?? 1100,
    overlayOpen: routeOverlayOpen,
    companionH,
    companionW
  })

  // While roaming, drive the directional run row + mirror from the travel
  // direction; at rest, fall back to the inward-facing static mascot.
  const walk = roamWalkRow(roamDir, info.stateRows)

  // While popped out, the desktop overlay window owns the mascot — hide the
  // in-window one so there aren't two.
  if (!info.enabled || !info.spritesheetBase64 || overlayActive) {
    return null
  }

  return (
    <div
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={onPointerUp}
      ref={containerRef}
      style={{
        cursor: 'grab',
        left: position.x,
        pointerEvents: 'auto',
        position: 'fixed',
        top: position.y,
        touchAction: 'none',
        userSelect: 'none',
        zIndex: 60
      }}
    >
      <div
        aria-hidden
        style={{
          background: `radial-gradient(ellipse at center, rgba(0,0,0,${shadowAlpha}) 0%, rgba(0,0,0,0) 70%)`,
          bottom: -shadowH * 0.4,
          height: shadowH,
          left: '50%',
          pointerEvents: 'none',
          position: 'absolute',
          transform: 'translateX(-50%)',
          width: shadowW,
          zIndex: 0
        }}
      />
      <div
        ref={spriteWrapRef}
        style={{
          lineHeight: 0,
          position: 'relative',
          transform: roamDir !== 0 ? (walk.mirror ? 'scaleX(-1)' : 'none') : facing(position.x, companionW),
          zIndex: 1
        }}
      >
        <CompanionSprite info={info} rowOverride={walk.row} />
      </div>
      {/* Hearts puff off the companion; its celebrate ("yay"/jump) pose is driven by
          burstVibeHearts's router. */}
      <CompanionHeartField companionH={companionH} companionW={companionW} />
      {/* Plugin lines (ctx.companion.say) float above the companion, upright (outside the
          mirrored sprite wrapper) and grown toward the window's middle so a companion
          parked at an edge never pushes its bubble off-screen. Core status
          lines stay in the pop-out overlay; in-window the app is the surface. */}
      <div
        style={{
          bottom: '100%',
          marginBottom: 4,
          pointerEvents: 'none',
          position: 'absolute',
          zIndex: 2,
          ...(position.x + companionW / 2 < (window.innerWidth || 800) / 2 ? { left: 0 } : { right: 0 })
        }}
      >
        <CompanionBubble showStatus={false} />
      </div>
    </div>
  )
}
