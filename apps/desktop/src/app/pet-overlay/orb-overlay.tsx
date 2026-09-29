import './orb-overlay.css'

import { useEffect, useRef, useState } from 'react'

import { JarvisCore } from '@/app/jarvis/core'
import { desktopOrbCopy } from '@/app/jarvis/desktop-orb-copy'
import {
  clampOrbScale,
  clampToArea,
  ORB_SCALE_MAX,
  ORB_SCALE_MIN,
  ORB_SCALE_STEP,
  orbWindowSize,
  type Rect,
  resizeAroundCenter,
  scaleFromDrag,
  snapToEdges
} from '@/app/jarvis/desktop-orb-geometry'
import { type DesktopOrbState, moveOrb, readOrbScale, writeOrbScale } from '@/app/jarvis/desktop-orb-state'
import { Button } from '@/components/ui/button'
import { TRANSLATIONS } from '@/i18n'
import { ExternalLink, Mic, Minus, Plus, Square, X } from '@/lib/icons'
import { playUiSound } from '@/lib/ui-sound'

/** The usable desktop rectangle of the display the orb is on (DIPs, like window bounds). */
function workArea(): Rect {
  const screen = window.screen as Screen & { availLeft?: number; availTop?: number }

  return { height: screen.availHeight, width: screen.availWidth, x: screen.availLeft ?? 0, y: screen.availTop ?? 0 }
}

const currentBounds = (): Rect => ({
  height: window.innerHeight,
  width: window.innerWidth,
  x: window.screenX,
  y: window.screenY
})

export function OrbOverlay({ state }: { state: DesktopOrbState }) {
  const copy = desktopOrbCopy[state.locale]
  const phaseCopy = TRANSLATIONS[state.locale].jarvisShell.dashboard.core.task
  const api = window.hermesDesktop?.petOverlay
  const orbRef = useRef<HTMLDivElement>(null)
  const [scale, setScale] = useState(readOrbScale)
  const scaleRef = useRef(scale)
  const persistTimer = useRef<ReturnType<typeof setTimeout>>(undefined)
  const grip = useRef<null | { centre: { x: number; y: number }; distance: number; scale: number }>(null)

  /**
   * Resize the orb around its own centre and keep it reachable. The window
   * follows at once (direct manipulation); the saved size and position settle
   * shortly after the last change so a wheel spin writes once, not per tick.
   */
  const changeScale = (requested: number, { quiet = false }: { quiet?: boolean } = {}) => {
    const next = clampOrbScale(requested)

    if (next === scaleRef.current) {
      return
    }

    if (!quiet) {
      playUiSound(next > scaleRef.current ? 'grow' : 'shrink')
    }

    scaleRef.current = next
    setScale(next)
    const bounds = clampToArea(resizeAroundCenter(currentBounds(), orbWindowSize(next)), workArea())
    api?.setBounds(bounds)
    clearTimeout(persistTimer.current)

    persistTimer.current = setTimeout(() => {
      writeOrbScale(next)
      api?.control({ type: 'bounds', bounds })
    }, 250)
  }

  useEffect(() => () => clearTimeout(persistTimer.current), [])

  const drag = useRef<null | {
    origin: { x: number; y: number }
    start: { x: number; y: number }
    x: number
    y: number
  }>(null)

  const status = !state.connected
    ? copy.offline
    : state.voice === 'error' || state.task === 'failed'
      ? copy.error
      : state.voice === 'speaking'
        ? copy.speaking
        : state.voice === 'listening'
          ? copy.listening
          : phaseCopy[state.task] || copy.idle

  useEffect(() => {
    let ignored = false

    const setIgnore = (next: boolean) => {
      if (next !== ignored) {
        ignored = next
        api?.setIgnoreMouse(next)
      }
    }

    const move = (event: MouseEvent) => {
      if (drag.current) {return}
      const target = document.elementFromPoint(event.clientX, event.clientY)
      const control = target?.closest('[data-orb-controls]')
      const rect = orbRef.current?.getBoundingClientRect()

      const inOrb =
        rect &&
        Math.hypot(event.clientX - rect.left - rect.width / 2, event.clientY - rect.top - rect.height / 2) <=
          rect.width * 0.44

      setIgnore(!control && !inOrb)
    }

    api?.setIgnoreMouse(false)
    window.addEventListener('mousemove', move)

    return () => {
      window.removeEventListener('mousemove', move)
      api?.setIgnoreMouse(false)
    }
  }, [api])

  return (
    <section
      aria-label={copy.show}
      className="desktop-orb"
      data-voice={state.voice}
      style={{ '--orb-scale': scale } as React.CSSProperties}
    >
      <div className="desktop-orb__stage">
      <div
        aria-label={copy.drag}
        className="desktop-orb__sphere"
        onDoubleClick={() => api?.control({ type: 'open-app' })}
        onLostPointerCapture={() => {
          drag.current = null
        }}
        onPointerDown={event => {
          if (event.button !== 0) {return}
          const origin = { x: window.screenX, y: window.screenY }
          drag.current = { origin, start: { x: event.screenX, y: event.screenY }, ...origin }
          event.currentTarget.setPointerCapture(event.pointerId)
          api?.setIgnoreMouse(false)
        }}
        onPointerMove={event => {
          if (!drag.current) {return}
          const next = moveOrb(drag.current.origin, drag.current.start, { x: event.screenX, y: event.screenY })
          Object.assign(drag.current, next)
          api?.setBounds({ ...next, width: window.innerWidth, height: window.innerHeight })
        }}
        onPointerUp={event => {
          if (!drag.current) {return}
          // Let go near a screen edge and the orb settles against it.
          const dropped = clampToArea({ ...currentBounds(), x: drag.current.x, y: drag.current.y }, workArea())
          const { bounds, snapped } = snapToEdges(dropped, workArea())

          if (snapped || bounds.x !== drag.current.x || bounds.y !== drag.current.y) {
            api?.setBounds(bounds)
          }

          if (snapped) {
            playUiSound('snap')
          }

          api?.control({ type: 'bounds', bounds })
          drag.current = null
          event.currentTarget.releasePointerCapture(event.pointerId)
        }}
        onWheel={event => changeScale(scaleRef.current - Math.sign(event.deltaY) * ORB_SCALE_STEP / 2)}
        ref={orbRef}
      >
        <JarvisCore
          live
          taskPhase={state.task}
          variant="hero"
          voice={state.voice === 'idle' && state.active ? 'listening' : state.voice}
        />
      </div>
        <button
          aria-label={copy.resize}
          className="desktop-orb__grip"
          data-orb-controls=""
          onPointerCancel={() => {
            grip.current = null
          }}
          onPointerDown={event => {
            const rect = orbRef.current?.getBoundingClientRect()

            if (event.button !== 0 || !rect) {return}
            const centre = { x: rect.left + rect.width / 2, y: rect.top + rect.height / 2 }

            grip.current = {
              centre,
              distance: Math.hypot(event.clientX - centre.x, event.clientY - centre.y),
              scale: scaleRef.current
            }
            event.currentTarget.setPointerCapture(event.pointerId)
          }}
          onPointerMove={event => {
            const g = grip.current

            if (!g) {return}
            // The window's centre stays put while resizing, so the sphere's centre in the
            // window shifts only by the growth; measure from the pointer's screen position.
            const distance = Math.hypot(event.clientX - g.centre.x, event.clientY - g.centre.y)
            changeScale(scaleFromDrag(g.scale, g.distance, distance), { quiet: true })
          }}
          onPointerUp={event => {
            grip.current = null
            event.currentTarget.releasePointerCapture(event.pointerId)
          }}
          type="button"
        />
      </div>
      <div className="desktop-orb__caption" role="status">
        <span aria-hidden="true" className="desktop-orb__dot" />
        {status}
      </div>
      <div className="desktop-orb__controls" data-orb-controls="">
        <Button
          aria-label={copy.open}
          onClick={() => {
            playUiSound('open')
            api?.control({ type: 'open-app' })
          }}
          size="icon-sm"
          variant="ghost"
        >
          <ExternalLink />
        </Button>
        <Button
          aria-label={copy.smaller}
          disabled={scale <= ORB_SCALE_MIN}
          onClick={() => changeScale(scale - ORB_SCALE_STEP)}
          size="icon-sm"
          variant="ghost"
        >
          <Minus />
        </Button>
        <Button
          aria-label={state.active ? copy.stop : copy.start}
          aria-pressed={state.active}
          className="desktop-orb__mic"
          disabled={!state.connected && !state.active}
          onClick={() => {
            playUiSound(state.active ? 'close' : 'open')
            api?.control({ type: 'orb-toggle-voice' })
          }}
          size="icon-lg"
          variant="default"
        >
          {state.active ? <Square /> : <Mic />}
        </Button>
        <Button
          aria-label={copy.larger}
          disabled={scale >= ORB_SCALE_MAX}
          onClick={() => changeScale(scale + ORB_SCALE_STEP)}
          size="icon-sm"
          variant="ghost"
        >
          <Plus />
        </Button>
        <Button
          aria-label={copy.hide}
          onClick={() => {
            playUiSound('close')
            api?.control({ type: 'pop-in' })
          }}
          size="icon-sm"
          variant="ghost"
        >
          <X />
        </Button>
      </div>
    </section>
  )
}
