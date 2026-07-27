import { useStore } from '@nanostores/react'
import { useEffect, useRef, useState } from 'react'

import { DecodeText } from '@/components/ui/decode-text'
import { cn } from '@/lib/utils'
import { $desktopBoot } from '@/store/boot'
import { $gatewaySwitching } from '@/store/gateway-switch'
import { $gatewayState } from '@/store/session'

// Decode mechanics live in the shared <DecodeText> primitive
// (components/ui/decode-text.tsx). "CONN" stays legible via prefix={4}.
const TEXT = 'CONNECTING'

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

function prefersReducedMotion(): boolean {
  return typeof window !== 'undefined' && Boolean(window.matchMedia?.('(prefers-reduced-motion: reduce)').matches)
}

export function GatewayConnectingOverlay() {
  const gatewayState = useStore($gatewayState)
  const boot = useStore($desktopBoot)
  const gatewaySwitching = useStore($gatewaySwitching)
  const [previewing] = useState(forcedPreview)
  const reduce = prefersReducedMotion()
  // Under reduced motion, skip the multi-phase exit choreography (text-out →
  // hold → overlay fade) and jump straight to gone so the overlay unmounts
  // the instant the gateway opens. E2E screenshots rely on this to avoid
  // catching the overlay mid-fade.
  const [phase, setPhase] = useState<Phase>('live')
  // Once cold boot has completed once, never resurrect the fullscreen overlay
  // — soft gateway switches keep the shell and reskeleton the sidebar instead.
  const coldBootDoneRef = useRef(false)

  if (!boot.running && boot.progress >= 100 && !boot.error) {
    coldBootDoneRef.current = true
  }

  // The full-screen connecting overlay is for initial boot only. After a
  // healthy boot, flaky networks / sleep-wake can drop the socket and flip the
  // gateway state back to closed/error while the app reconnects. Do not cover
  // the chat then — users should still be able to type drafts, open settings,
  // and recover instead of staring at a modal CONNECTING screen.
  const initialBootActive = boot.visible || boot.running || boot.progress < 100

  const connecting =
    !coldBootDoneRef.current && !gatewaySwitching && gatewayState !== 'open' && !boot.error && initialBootActive

  // Latches once we've actually shown the overlay, so the brief frame where
  // gatewayState flips to "open" (connecting -> false) before the exit phase
  // kicks in doesn't unmount us and cause a flash.
  const shownRef = useRef(false)

  if (previewing || connecting) {
    shownRef.current = true
  }

  // Kick off the exit when connected: real connect, or a faked timer in preview.
  useEffect(() => {
    if (phase !== 'live') {
      return
    }

    if (previewing) {
      const id = window.setTimeout(() => setPhase('text-out'), PREVIEW_CONNECT_MS)

      return () => window.clearTimeout(id)
    }

    if (gatewayState === 'open' && shownRef.current) {
      // Under reduced motion, skip the multi-phase exit choreography
      // (text-out → hold → overlay fade) and jump straight to gone so the
      // overlay unmounts the instant the gateway opens. E2E screenshots
      // rely on this to avoid catching the overlay mid-fade.
      setPhase(reduce ? 'gone' : 'text-out')
    }
  }, [phase, previewing, gatewayState, reduce])

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
      const id = window.setTimeout(() => setPhase('live'), PREVIEW_REPLAY_MS)

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
      <DecodeText
        active={phase === 'live' && (previewing || connecting)}
        className={cn(
          'pl-[0.4em] text-(--theme-primary) transition duration-300 ease-out',
          leaving ? 'translate-y-2 opacity-0 saturate-0' : 'translate-y-0 opacity-100 saturate-100'
        )}
        cursor
        prefix={4}
        text={TEXT}
      />
    </div>
  )
}
