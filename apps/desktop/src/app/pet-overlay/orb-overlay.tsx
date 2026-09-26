import './orb-overlay.css'

import { useEffect, useRef } from 'react'

import { JarvisCore } from '@/app/jarvis/core'
import { desktopOrbCopy } from '@/app/jarvis/desktop-orb-copy'
import { type DesktopOrbState, moveOrb } from '@/app/jarvis/desktop-orb-state'
import { Button } from '@/components/ui/button'
import { ExternalLink, Mic, Square, X } from '@/lib/icons'

export function OrbOverlay({ state }: { state: DesktopOrbState }) {
  const copy = desktopOrbCopy[state.locale]
  const api = window.hermesDesktop?.petOverlay
  const orbRef = useRef<HTMLDivElement>(null)

  const drag = useRef<null | {
    origin: { x: number; y: number }
    start: { x: number; y: number }
    x: number
    y: number
  }>(null)

  const status = !state.connected
    ? copy.offline
    : state.voice === 'error'
      ? copy.error
      : state.voice === 'speaking'
        ? copy.speaking
        : state.active
          ? copy.listening
          : state.task === 'running' || state.task === 'planning'
            ? copy.working
            : copy.idle

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
    <section aria-label={copy.show} className="desktop-orb" data-voice={state.voice}>
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
          api?.control({
            type: 'bounds',
            bounds: { x: drag.current.x, y: drag.current.y, width: window.innerWidth, height: window.innerHeight }
          })
          drag.current = null
          event.currentTarget.releasePointerCapture(event.pointerId)
        }}
        ref={orbRef}
      >
        <JarvisCore
          live
          taskPhase={state.task}
          variant="hero"
          voice={state.voice === 'idle' && state.active ? 'listening' : state.voice}
        />
      </div>
      <div className="desktop-orb__caption" role="status">
        {status}
      </div>
      <div className="desktop-orb__controls" data-orb-controls="">
        <Button
          aria-label={copy.open}
          onClick={() => api?.control({ type: 'open-app' })}
          size="icon"
          variant="secondary"
        >
          <ExternalLink />
        </Button>
        <Button
          aria-label={state.active ? copy.stop : copy.start}
          aria-pressed={state.active}
          disabled={!state.connected && !state.active}
          onClick={() => api?.control({ type: 'orb-toggle-voice' })}
          size="icon-lg"
          variant="default"
        >
          {state.active ? <Square /> : <Mic />}
        </Button>
        <Button aria-label={copy.hide} onClick={() => api?.control({ type: 'pop-in' })} size="icon" variant="secondary">
          <X />
        </Button>
      </div>
    </section>
  )
}
