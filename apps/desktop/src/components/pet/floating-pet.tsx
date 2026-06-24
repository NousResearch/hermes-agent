import { useStore } from '@nanostores/react'
import { useCallback, useEffect, useRef, useState } from 'react'

import { useGatewayRequest } from '@/app/gateway/hooks/use-gateway-request'
import { persistString, storedString } from '@/lib/storage'
import { $petInfo, clearPetUnread, type PetInfo, petProfile, setPetInfo } from '@/store/pet'
import { resetPetGallery } from '@/store/pet-gallery'
import { $petOverlayActive, initPetOverlayBridge, popOutPet, restorePetOverlay } from '@/store/pet-overlay'
import { $activeGatewayProfile, normalizeProfileKey } from '@/store/profile'
import { $gatewayState } from '@/store/session'
import { isSecondaryWindow } from '@/store/windows'
import { useTheme } from '@/themes/context'

import { PetSprite } from './pet-sprite'

// v2: positions are now top/left anchored (v1 stored bottom-anchored values,
// which dragged inverted). Bumping the key discards stale v1 coordinates.
const POSITION_KEY = 'hermes.desktop.pet-position.v2'

interface Point {
  x: number
  y: number
}

function clampToViewport({ x, y }: Point): Point {
  const maxX = Math.max(0, (window.innerWidth || 800) - 80)
  const maxY = Math.max(0, (window.innerHeight || 600) - 80)

  return { x: Math.min(Math.max(0, x), maxX), y: Math.min(Math.max(0, y), maxY) }
}

// The sprite art faces left by default, so mirror it when the pet's center sits
// on the left half of the window — it always faces inward, toward the content.
function facing(leftX: number, petW: number): string {
  return leftX + petW / 2 < (window.innerWidth || 800) / 2 ? 'scaleX(-1)' : 'none'
}

function loadPosition(): Point {
  try {
    const raw = storedString(POSITION_KEY)

    if (raw) {
      const parsed = JSON.parse(raw) as Point

      if (typeof parsed.x === 'number' && typeof parsed.y === 'number') {
        return clampToViewport(parsed)
      }
    }
  } catch {
    // fall through to default
  }

  // Default: lower-left corner (top/left anchored).
  return clampToViewport({ x: 24, y: (window.innerHeight || 600) - 220 })
}

/**
 * In-window floating petdex mascot. Always-on-top within the app, draggable,
 * and reactive to agent activity via `$petState`. Fetches the active pet via
 * the shared `pet.info` RPC; renders nothing until a pet is installed +
 * enabled.
 *
 * Adopting a pet is fully in-app: type `/pet boba` in the composer. That
 * writes `display.pet.*` from the slash worker, so we keep polling `pet.info`
 * while no pet is active and the mascot pops in within a few seconds — no
 * reload, no CLI. Once a pet is live we stop polling.
 *
 * Promotion to a separate frameless OS-level window is a follow-up — the
 * sprite + state logic here is reused as-is, only the host changes.
 */
const PET_POLL_MS = 3000

export function FloatingPet() {
  const { requestGateway } = useGatewayRequest()
  const { resolvedMode } = useTheme()
  const gatewayState = useStore($gatewayState)
  const info = useStore($petInfo)
  const overlayActive = useStore($petOverlayActive)

  const [position, setPosition] = useState<Point>(loadPosition)
  const containerRef = useRef<HTMLDivElement | null>(null)
  // The facing mirror lives on the sprite wrapper, not the container, so the
  // speech bubble (a container child) never renders flipped/backwards.
  const spriteWrapRef = useRef<HTMLDivElement | null>(null)
  const petW = (info.frameW ?? 192) * (info.scale ?? 0.33)
  // Soft contact shadow, sized off the pet so every scale/species grounds the
  // same way (cf. lairp's per-actor feet ellipse). Lighter on light backgrounds.
  const shadowW = Math.round(petW * 0.55)
  const shadowH = Math.max(3, Math.round(shadowW * 0.28))
  const shadowAlpha = resolvedMode === 'light' ? 0.2 : 0.55
  // Live drag offset (pointer → element top-left). Drag updates the DOM
  // directly to avoid a React re-render (and canvas reflow) per pointermove —
  // state is only committed on release.
  const dragRef = useRef<{ dx: number; dy: number; x: number; y: number } | null>(null)

  // Fetch pet.info on connect, then keep polling while no pet is active so an
  // in-app `/pet <slug>` shows up live. Stops polling once a pet is enabled.
  const active = info.enabled && Boolean(info.spritesheetBase64)
  useEffect(() => {
    if (gatewayState !== 'open' || active) {
      return
    }

    let cancelled = false

    const pull = async () => {
      try {
        const next = await requestGateway<PetInfo>('pet.info', { profile: petProfile() })

        if (!cancelled && next) {
          setPetInfo(next)
        }
      } catch {
        // cosmetic feature — never surface gateway errors
      }
    }

    void pull()
    const timer = window.setInterval(() => void pull(), PET_POLL_MS)

    return () => {
      cancelled = true
      window.clearInterval(timer)
    }
  }, [gatewayState, active, requestGateway])

  // Pets are per-profile. When the active profile changes, drop the previous
  // profile's mascot + gallery cache so the poll above refetches the new
  // profile's pet (its config + pets dir resolve per-profile on the backend).
  const profileRef = useRef(normalizeProfileKey($activeGatewayProfile.get()))
  useEffect(
    () =>
      $activeGatewayProfile.subscribe(next => {
        const key = normalizeProfileKey(next)

        if (key === profileRef.current) {
          return
        }

        profileRef.current = key
        setPetInfo({ enabled: false })
        resetPetGallery()
      }),
    []
  )

  // Wire the overlay control channel once, only in the primary window — the
  // pop-out overlay belongs to it (main.cjs positions it against the main
  // window and routes control messages back to it).
  useEffect(() => {
    if (isSecondaryWindow()) {
      return
    }

    return initPetOverlayBridge()
  }, [])

  // Returning to the app (by any route, not just the mail icon) clears the pet's
  // "new message" hint — you've seen it now.
  useEffect(() => {
    if (isSecondaryWindow()) {
      return
    }

    const onFocus = () => clearPetUnread()
    window.addEventListener('focus', onFocus)

    return () => window.removeEventListener('focus', onFocus)
  }, [])

  // Restore a popped-out pet on boot, once the pet has loaded (so we never spawn
  // an empty overlay window). Primary window only; runs at most once.
  const restoredRef = useRef(false)
  useEffect(() => {
    if (isSecondaryWindow() || restoredRef.current || !active) {
      return
    }

    restoredRef.current = true
    restorePetOverlay()
  }, [active])

  // A window resize must never strand the pet off-screen — re-clamp the
  // committed position (and persist it) whenever the viewport shrinks.
  useEffect(() => {
    const onResize = () =>
      setPosition(prev => {
        const next = clampToViewport(prev)

        if (next.x === prev.x && next.y === prev.y) {
          return prev
        }

        persistString(POSITION_KEY, JSON.stringify(next))

        return next
      })

    window.addEventListener('resize', onResize)

    return () => window.removeEventListener('resize', onResize)
  }, [])

  const onPointerDown = useCallback((e: React.PointerEvent) => {
    const el = containerRef.current

    if (!el) {
      return
    }

    const rect = el.getBoundingClientRect()

    // Shift-click pops the pet out into a free-floating desktop overlay (it can
    // leave the window and stays visible while Hermes is minimized) instead of
    // starting an in-window drag. Primary window only — the overlay is anchored
    // to it.
    if (e.shiftKey && !isSecondaryWindow()) {
      popOutPet({ height: rect.height, width: rect.width, x: rect.left, y: rect.top })

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

      const next = clampToViewport({ x: e.clientX - drag.dx, y: e.clientY - drag.dy })
      drag.x = next.x
      drag.y = next.y
      // Mutate the DOM directly — no setState, so no re-render while dragging. The
      // mirror follows the pointer across the midline for the same reason; it
      // rides the sprite wrapper so the bubble stays upright.
      el.style.left = `${next.x}px`
      el.style.top = `${next.y}px`

      if (spriteWrapRef.current) {
        spriteWrapRef.current.style.transform = facing(next.x, petW)
      }
    },
    [petW]
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
      <div ref={spriteWrapRef} style={{ lineHeight: 0, position: 'relative', transform: facing(position.x, petW), zIndex: 1 }}>
        <PetSprite info={info} />
      </div>
    </div>
  )
}
