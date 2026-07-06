import { useStore } from '@nanostores/react'
import { useEffect, useRef, useState } from 'react'

import { cn } from '@/lib/utils'
import { $desktopBoot } from '@/store/boot'
import { $gatewayState } from '@/store/session'

// Static, always-legible prefix; only TAIL ever scrambles. Splitting them at
// the render level means no timer logic (even a stale HMR one) can ever
// scramble "CONN".
const PREFIX = 'CONN'
const TAIL = 'ECTING'
// Even-weight mono ascii so cycling glyphs don't jump width (matches the
// nousnet-web download-button decode effect).
const SCRAMBLE_CHARS = '/\\|-_=+<>~:*'
const TICK_MS = 45

// Exit choreography (ms): text fades down + out, hold, then the overlay fades.
const TEXT_OUT_MS = 360
const POST_TEXT_HOLD_MS = 300
const OVERLAY_OUT_MS = 520
// Preview-only: how long to "connect" for, and the pause before replaying.
const PREVIEW_CONNECT_MS = 2600
const PREVIEW_REPLAY_MS = 1100

type Phase = 'live' | 'text-out' | 'overlay-out' | 'gone'

// Dev affordance: a warm Cmd+R reconnects almost instantly, so the overlay
// only flashes. Load with `?connecting=1` to force a looping preview.
function forcedPreview(): boolean {
  if (!import.meta.env.DEV || typeof window === 'undefined') {
    return false
  }

  try {
    return new URLSearchParams(window.location.search).get('connecting') === '1'
  } catch {
    return false
  }
}

function scrambledTail(resolvedCount: number): string {
  return Array.from(TAIL, (ch, i) =>
    i < resolvedCount ? ch : SCRAMBLE_CHARS[(Math.random() * SCRAMBLE_CHARS.length) | 0]
  ).join('')
}

export function GatewayConnectingOverlay() {
  const gatewayState = useStore($gatewayState)
  const boot = useStore($desktopBoot)
  const [previewing] = useState(forcedPreview)
  const [tail, setTail] = useState(TAIL)
  const [phase, setPhase] = useState<Phase>('live')

  // The full-screen connecting overlay is for initial boot only. After a
  // healthy boot, flaky networks / sleep-wake can drop the socket and flip the
  // gateway state back to closed/error while the app reconnects. Do not cover
  // the chat then — users should still be able to type drafts, open settings,
  // and recover instead of staring at a modal CONNECTING screen.
  const initialBootActive = boot.visible || boot.running || boot.progress < 100
  const connecting = gatewayState !== 'open' && !boot.error && initialBootActive
  // Latches once we've actually shown the overlay, so the brief frame where
  // gatewayState flips to "open" (connecting -> false) before the exit phase
  // kicks in doesn't unmount us and cause a flash.
  const shownRef = useRef(false)

  if (previewing || connecting) {
    shownRef.current = true
  }

  // Decode loop — only while live (freeze the resolved word during the exit).
  useEffect(() => {
    if (phase !== 'live' || (!previewing && !connecting)) {
      return
    }

    let resolved = 0
    let hold = 0

    const id = window.setInterval(() => {
      if (resolved >= TAIL.length) {
        hold += 1

        if (hold > 16) {
          resolved = 0
          hold = 0
        }

        setTail(TAIL)

        return
      }

      resolved += 0.5
      setTail(scrambledTail(Math.floor(resolved)))
    }, TICK_MS)

    return () => window.clearInterval(id)
  }, [phase, previewing, connecting])

  // Kick off the exit when connected: real connect, or a faked timer in preview.
  useEffect(() => {
    if (phase !== 'live') {
      return
    }

    if (previewing) {
      const id = window.setTimeout(() => {
        setTail(TAIL)
        setPhase('text-out')
      }, PREVIEW_CONNECT_MS)

      return () => window.clearTimeout(id)
    }

    if (gatewayState === 'open' && shownRef.current) {
      setTail(TAIL)
      setPhase('text-out')
    }
  }, [phase, previewing, gatewayState])

  // Advance the exit choreography: text-out -> overlay-out -> gone.
  useEffect(() => {
    if (phase === 'text-out') {
      const id = window.setTimeout(() => setPhase('overlay-out'), TEXT_OUT_MS + POST_TEXT_HOLD_MS)

      return () => window.clearTimeout(id)
    }

    if (phase === 'overlay-out') {
      const id = window.setTimeout(() => setPhase('gone'), OVERLAY_OUT_MS)

      return () => window.clearTimeout(id)
    }

    // Preview replays so we can keep watching the transition.
    if (phase === 'gone' && previewing) {
      const id = window.setTimeout(() => {
        setTail(TAIL)
        setPhase('live')
      }, PREVIEW_REPLAY_MS)

      return () => window.clearTimeout(id)
    }
  }, [phase, previewing])

  // Boot failed — BootFailureOverlay owns the screen; don't linger behind it.
  if (boot.error && !previewing) {
    return null
  }

  // Real connect: once the fade finishes, get out of the way for good.
  if (phase === 'gone' && !previewing) {
    return null
  }

  // Never showed (e.g. gateway already up on a warm reload) — stay out.
  if (!previewing && !connecting && !shownRef.current) {
    return null
  }

  const leaving = phase !== 'live'
  const overlayHidden = phase === 'overlay-out' || phase === 'gone'

  return (
    <div
      className={cn(
        'fixed inset-0 z-[1200] grid place-items-center bg-(--ui-chat-surface-background) transition-opacity duration-500 ease-out',
        overlayHidden ? 'pointer-events-none opacity-0' : 'opacity-100'
      )}
    >
      <style>{'@keyframes gco-cursor { 0%, 49% { opacity: 1 } 50%, 100% { opacity: 0 } }'}</style>
      <span
        className={cn(
          'inline-flex items-center pl-[0.4em] font-mono text-[0.64rem] font-semibold uppercase tracking-[0.4em] tabular-nums text-(--theme-primary) transition duration-300 ease-out',
          leaving ? 'translate-y-2 opacity-0 saturate-0' : 'translate-y-0 opacity-100 saturate-100'
        )}
      >
        {PREFIX}
        {tail}
        <span
          aria-hidden="true"
          className="dither ml-0.5 inline-block size-2 shrink-0 -translate-y-px rounded-[1px]"
          style={{ animation: 'gco-cursor 1s step-end infinite' }}
        />
      </span>
    </div>
  )
}
