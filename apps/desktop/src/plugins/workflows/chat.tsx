/**
 * Hermes, on the canvas.
 *
 * This is the app's own chat — `SessionChat` from the SDK is the same
 * `ChatView` the workspace pane renders, so the transcript, the tool cards,
 * the streaming and thinking indicators, attachments, voice and the composer
 * are not approximations of the real thing, they ARE it. The plugin's only job
 * is to say which conversation and to give it a shape that suits a canvas.
 *
 * There was a hand-rolled version of this: bespoke turn rows, a hand-driven
 * event stream, a spinner made of three dots. It cost a lot and looked like a
 * copy, which is what it was. Everything it did is now upstream, and the parts
 * worth keeping — the dock the composer sits in, the transport fused to its
 * top — are chrome around the real component rather than a rebuild of it.
 *
 * Layout is CSS, from `[data-canvas-chat]` down, the way HUD mode restyles
 * this same tree through `[data-hud-shell]`. No variant props, nothing forked:
 * when the chat gains a feature, this gains it too.
 */

import { Codicon, SessionChat, useValue } from '@hermes/plugin-sdk'
import { useCallback, useEffect, useRef, useState } from 'react'

import { $workflows } from './documents'
import { ensureCanvasSession, replaceCanvasSession } from './session'

/**
 * Measure the furniture the band is positioned against.
 *
 * The chat is built to fill a pane: its scroll container is `min-height: 100%`,
 * so in a dock it either fills everything it is given or collapses to nothing,
 * and neither is a band. So the band is an absolutely-positioned box, and it
 * needs to know how tall the bar and transport under it are to sit on their
 * top edge.
 *
 * Its HEIGHT is not measured. That was HUD's trick — size the band to the tight
 * bbox of the rows — and in a dock it means the transcript's ceiling moves with
 * whatever is in it, so a short exchange gets a tiny box. The stylesheet owns
 * one height now; the only thing measured here is whether there are turns at
 * all, because a session with none should show no sheet.
 */
function useBandMetrics(root: React.RefObject<HTMLDivElement | null>) {
  useEffect(() => {
    const el = root.current

    if (!el) {
      return
    }

    let viewport: HTMLElement | null = null
    const ro = new ResizeObserver(() => measure())

    const measure = () => {
      const found = el.querySelector<HTMLElement>('[data-slot="aui_thread-viewport"]')

      if (found !== viewport) {
        viewport = found

        if (found) {
          ro.observe(found)

          if (found.firstElementChild) {
            ro.observe(found.firstElementChild)
          }
        }
      }

      // Turns only. The content box always has furniture — titlebar pad,
      // composer clearance, the empty-state grid (h-full + py-8) — and
      // counting those is how a brand-new session grew a blank sheet the
      // size of the band the moment the composer took focus.
      const turns = Array.from(
        viewport?.querySelectorAll<HTMLElement>(
          '[data-slot="aui_user-message-root"], [data-slot="aui_assistant-message-root"]'
        ) ?? []
      )

      el.toggleAttribute(
        'data-canvas-thread',
        turns.some(row => row.getBoundingClientRect().height > 0)
      )

      const bar = el.querySelector<HTMLElement>('[data-slot="composer-dock"]')

      if (bar) {
        ro.observe(bar)
        el.style.setProperty('--canvas-bar-height', `${Math.round(bar.getBoundingClientRect().height)}px`)
      }

      // The transport rides between the band and the composer, so the band has
      // to clear it as well. It lives outside this element (it's the dock card
      // the composer is fused to), hence the reach upward.
      const transport = el.parentElement?.querySelector<HTMLElement>('.canvas-dock-transport')

      if (transport) {
        ro.observe(transport)
        el.style.setProperty('--canvas-transport-height', `${Math.round(transport.getBoundingClientRect().height)}px`)
      }
    }

    // The chat surface mounts async, so poll until the viewport exists and let
    // the observer take it from there.
    measure()
    const probe = window.setInterval(measure, 500)

    return () => {
      window.clearInterval(probe)
      ro.disconnect()
    }
  }, [root])
}

export function CanvasChat({ autofocus, workflowId }: { autofocus?: boolean; workflowId: string }) {
  const bound = useValue($workflows).find(d => d.id === workflowId)?.sessionId ?? ''
  const [session, setSession] = useState(bound)
  const [error, setError] = useState('')
  const root = useRef<HTMLDivElement>(null)

  useBandMetrics(root)

  const replacing = useRef(false)
  const remint = useCallback(() => {
    if (replacing.current) {
      return
    }

    replacing.current = true
    setSession('')
    void replaceCanvasSession(workflowId)
      .then(id => setSession(id))
      .catch((err: unknown) => setError(err instanceof Error ? err.message : String(err)))
      .finally(() => {
        replacing.current = false
      })
  }, [workflowId])

  useEffect(() => {
    if (bound) {
      setSession(bound)
      setError('')

      return
    }

    let live = true

    ensureCanvasSession(workflowId)
      .then(id => live && setSession(id))
      .catch((err: unknown) => live && setError(err instanceof Error ? err.message : String(err)))

    return () => {
      live = false
    }
  }, [bound, workflowId])

  useEffect(() => {
    if (!autofocus || !session) {
      return
    }

    const node = root.current
    let tries = 0

    const tick = window.setInterval(() => {
      const input = node?.querySelector<HTMLElement>('[data-slot="composer-rich-input"]')

      if (input) {
        input.focus()
        window.clearInterval(tick)
      } else if (++tries > 40) {
        window.clearInterval(tick)
      }
    }, 50)

    return () => window.clearInterval(tick)
  }, [autofocus, session])

  if (error) {
    return (
      <div className="canvas-chat-idle">
        <Codicon name="warning" />
        {error}
      </div>
    )
  }

  return (
    <div className="canvas-chat" data-canvas-chat="" ref={root}>
      {/* A session that vanished under us and an explicit `/new` want the same
          thing: mint a fresh one and stay on the canvas. */}
      {session ? <SessionChat onMissing={remint} onNew={remint} storedSessionId={session} /> : null}
    </div>
  )
}
