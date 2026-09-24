import { type CSSProperties, useCallback, useRef } from 'react'

import { ContribBoundary, ContribRender } from '@/contrib/react/boundary'
import { useContributions } from '@/contrib/react/use-contributions'

/** Below this much pointer travel, a header press counts as a click, not a
 *  drag (the pet overlay's slop gate). */
const CLICK_SLOP_PX = 3
/** Resize grip size (px) in the bottom-right corner. */
const RESIZE_GRIP_PX = 16
/** Window minimums, mirrored from plugin-overlay-geometry.ts. */
const MIN_WIDTH = 120
const MIN_HEIGHT = 80

interface DragState {
  startX: number
  startY: number
  offX: number
  offY: number
  moved: boolean
}

interface ResizeState {
  startX: number
  startY: number
  width: number
  height: number
  moved: boolean
}

/**
 * The plugin-overlay window's only view: a transparent, draggable, resizable
 * frame hosting ONE plugin contribution (`area: 'pluginOverlay'`), in one of
 * TWO postures owned by main:
 *
 *   mascot — a small non-activating sprite pinned to the Hermes app window.
 *            The whole window is the drag handle; drag reports go to main,
 *            which clamps the window into the main window's rect (the
 *            sprite never leaves the app). No header, no resize grip.
 *   card   — the interactive Q&A surface. Full opaque card, header drag,
 *            corner resize grip, – shrinks back to the mascot (no ✕ close:
 *            the overlay's only exit is shrinking to the sprite).
 *
 * The window's geometry authority is THIS renderer (same contract as the
 * pet overlay): the header drags the window, the corner grip resizes it.
 * Live movement flows through `setBounds` (transient — main snaps the
 * window); the END of a drag/resize flows through `reportBounds` (durable —
 * main snaps AND persists under its own hosted-plugin latch).
 *
 * The contribution renders inside a ContribBoundary so a crashing plugin
 * degrades to an error card, never a white window.
 */
export function PluginOverlayApp({
  pluginId,
  mode,
  ready
}: {
  pluginId: null | string
  mode: null | 'mascot' | 'card'
  ready: boolean
}) {
  const dragRef = useRef<DragState | null>(null)
  const resizeRef = useRef<ResizeState | null>(null)

  const bridge = window.hermesDesktop?.pluginOverlay
  const contributions = useContributions('pluginOverlay')
  const contribution = contributions.find(c => c.source === `plugin:${pluginId}`) ?? contributions[0]

  const isMascot = mode === 'mascot'

  const sendBounds = useCallback(
    (bounds: { x: number; y: number; width: number; height: number }, durable: boolean) => {
      if (durable) {
        bridge?.reportBounds(bounds)
      } else {
        bridge?.setBounds(bounds)
      }
    },
    [bridge]
  )

  // Header drag — the pet overlay's drag pattern, translated to a header bar.
  const onHeaderPointerDown = (e: React.PointerEvent) => {
    if (e.button !== 0 || !bridge) return

    ;(e.target as Element).setPointerCapture?.(e.pointerId)
    dragRef.current = {
      startX: e.screenX,
      startY: e.screenY,
      offX: e.screenX - window.screenX,
      offY: e.screenY - window.screenY,
      moved: false
    }
  }

  const onHeaderPointerMove = (e: React.PointerEvent) => {
    const drag = dragRef.current

    if (!drag) return

    if (Math.hypot(e.screenX - drag.startX, e.screenY - drag.startY) > CLICK_SLOP_PX) {
      drag.moved = true
    }

    sendBounds(
      {
        x: e.screenX - drag.offX,
        y: e.screenY - drag.offY,
        width: window.outerWidth,
        height: window.outerHeight
      },
      false
    )
  }

  const onHeaderPointerUp = (e: React.PointerEvent) => {
    const drag = dragRef.current

    dragRef.current = null
    ;(e.target as Element).releasePointerCapture?.(e.pointerId)

    if (drag?.moved) {
      sendBounds(
        {
          x: e.screenX - drag.offX,
          y: e.screenY - drag.offY,
          width: window.outerWidth,
          height: window.outerHeight
        },
        true
      )
    } else if (isMascot) {
      // A press without drag slop is a click — expand to the card. Pointer
      // events are the reliable signal here: the window is focusable:false
      // in mascot posture, where a synthetic browser `click` never fires.
      bridge?.setMode('card')
    }
  }

  // Corner resize — same live/durable split.
  const onResizePointerDown = (e: React.PointerEvent) => {
    if (e.button !== 0 || !bridge) return

    ;(e.target as Element).setPointerCapture?.(e.pointerId)
    resizeRef.current = {
      startX: e.screenX,
      startY: e.screenY,
      width: window.outerWidth,
      height: window.outerHeight,
      moved: false
    }
  }

  const onResizePointerMove = (e: React.PointerEvent) => {
    const resize = resizeRef.current

    if (!resize) return

    if (Math.hypot(e.screenX - resize.startX, e.screenY - resize.startY) > CLICK_SLOP_PX) {
      resize.moved = true
    }

    sendBounds(
      {
        x: window.screenX,
        y: window.screenY,
        width: Math.max(MIN_WIDTH, resize.width + (e.screenX - resize.startX)),
        height: Math.max(MIN_HEIGHT, resize.height + (e.screenY - resize.startY))
      },
      false
    )
  }

  const onResizePointerUp = (e: React.PointerEvent) => {
    const resize = resizeRef.current

    resizeRef.current = null
    ;(e.target as Element).releasePointerCapture?.(e.pointerId)

    if (resize?.moved) {
      sendBounds(
        {
          x: window.screenX,
          y: window.screenY,
          width: Math.max(MIN_WIDTH, resize.width + (e.screenX - resize.startX)),
          height: Math.max(MIN_HEIGHT, resize.height + (e.screenY - resize.startY))
        },
        true
      )
    }
  }

  const backToMascot = () => {
    bridge?.setMode('mascot')
  }

  // ── Mascot posture: the whole window IS the sprite. Drag moves the
  // window (main clamps into the app rect); a click (below drag slop)
  // expands to the card. No header, no grip — nothing but the sprite.
  if (isMascot) {
    return (
      <div
        onPointerDown={onHeaderPointerDown}
        onPointerMove={onHeaderPointerMove}
        onPointerUp={onHeaderPointerUp}
        style={{
          alignItems: 'center',
          background: 'transparent',
          cursor: 'grab',
          display: 'flex',
          height: '100vh',
          justifyContent: 'center',
          touchAction: 'none',
          userSelect: 'none',
          width: '100vw'
        }}
        title="Ask Dash"
      >
        {!ready ? (
          <div style={{ color: 'var(--ui-text-secondary)', fontSize: 12, padding: 16 }}>…</div>
        ) : contribution?.render ? (
          <ContribBoundary id={contribution.id} variant="pane">
            <ContribRender render={contribution.render} />
          </ContribBoundary>
        ) : null}
      </div>
    )
  }

  // ── Card posture: full opaque card with the header drag handle.
  return (
    <div
      style={{
        // The card paints its own full surface — Electron cannot flip
        // `transparent` at runtime, so the content just fills the window
        // and the desktop shows nowhere. SQUARE corners: without a
        // compositor, rounded corners would leave black alpha pixels.
        background: 'var(--ui-bg-elevated)',
        display: 'flex',
        flexDirection: 'column',
        height: '100vh',
        overflow: 'hidden',
        width: '100vw',
        userSelect: 'none'
      }}
    >
      {/* Header — the drag handle. */}
      <div
        onPointerDown={onHeaderPointerDown}
        onPointerMove={onHeaderPointerMove}
        onPointerUp={onHeaderPointerUp}
        style={{
          alignItems: 'center',
          cursor: 'grab',
          display: 'flex',
          flexShrink: 0,
          gap: 8,
          padding: '8px 10px',
          touchAction: 'none',
          background: 'transparent',
          borderBottom: '1px solid var(--ui-stroke-secondary)',
          color: 'var(--ui-text-secondary)',
          fontSize: 12
        }}
      >
        <span style={{ flex: 1, fontWeight: 600, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
          {pluginId ?? 'Plugin overlay'}
        </span>
        <button
          aria-label="Shrink to mascot"
          onClick={backToMascot}
          onPointerDown={e => e.stopPropagation()}
          onPointerUp={e => e.stopPropagation()}
          style={{
            alignItems: 'center',
            background: 'transparent',
            border: '1px solid var(--ui-stroke-secondary)',
            borderRadius: 6,
            color: 'var(--ui-text-secondary)',
            cursor: 'pointer',
            display: 'flex',
            height: 22,
            justifyContent: 'center',
            padding: 0,
            width: 22
          }}
          title="Shrink to the pencil"
          type="button"
        >
          –
        </button>
      </div>

      {/* Hosted contribution. */}
      <div
        style={{
          flex: 1,
          minHeight: 0,
          overflow: 'auto',
          userSelect: 'text',
          WebkitUserSelect: 'text',
          // The card surface: opaque for readability (the card posture is
          // the "non-transparent window" by design).
          background: 'var(--ui-bg-elevated)'
        }}
      >
        {!ready ? (
          <div style={{ color: 'var(--ui-text-secondary)', fontSize: 12, padding: 16 }}>Connecting…</div>
        ) : !pluginId ? (
          <div style={{ color: 'var(--ui-text-secondary)', fontSize: 12, padding: 16 }}>No plugin hosted.</div>
        ) : contribution?.render ? (
          <ContribBoundary id={contribution.id} variant="pane">
            <ContribRender render={contribution.render} />
          </ContribBoundary>
        ) : (
          <div style={{ color: 'var(--ui-text-secondary)', fontSize: 12, padding: 16 }}>
            Plugin "{pluginId}" registers no pluginOverlay contribution.
          </div>
        )}
      </div>

      {/* Corner resize grip. */}
      <div
        aria-label="Resize"
        onPointerDown={onResizePointerDown}
        onPointerMove={onResizePointerMove}
        onPointerUp={onResizePointerUp}
        role="separator"
        style={
          {
            bottom: 0,
            cursor: 'nwse-resize',
            height: RESIZE_GRIP_PX,
            position: 'absolute',
            right: 0,
            touchAction: 'none',
            width: RESIZE_GRIP_PX
          } as CSSProperties
        }
      />
    </div>
  )
}
