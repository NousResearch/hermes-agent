import { useEffect, useRef, useState } from 'react'

import { GlyphSpinner } from '@/components/ui/glyph-spinner'
import { useI18n } from '@/i18n'
import { cn } from '@/lib/utils'

// Shown over the conversation while the live gateway swaps to another profile's
// backend (lazily spawned). Keeps the last profile name through the fade-out so
// the label doesn't blank. Purely visual — pointer-events-none.
export function ChatSwapOverlay({ botMode = false, profile }: { botMode?: boolean; profile: string | null }) {
  const { t } = useI18n()
  const [label, setLabel] = useState<null | string>(profile)
  const [covering, setCovering] = useState(Boolean(profile))
  const [statusVisible, setStatusVisible] = useState(!botMode && Boolean(profile))
  const overlayRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (profile) {
      setLabel(profile)
      setCovering(true)

      if (!botMode) {
        setStatusVisible(true)

        return
      }

      // Warm Bot Chat switches normally paint in well under this threshold.
      // Keep the opaque ownership guard immediate, but do not flash a modal
      // "Waking up" card for a transition that completes almost at once. A
      // genuinely cold wake still gets status once the delay elapses.
      setStatusVisible(false)
      const statusTimer = window.setTimeout(() => setStatusVisible(true), 300)

      return () => window.clearTimeout(statusTimer)
    }

    setStatusVisible(false)

    if (!covering) {
      return
    }

    // The session stores can settle before React commits, the tab becomes
    // visible and a large transcript finishes its keep-alive catch-up. Keep a
    // short paint-frame floor, then require two quiet frames after the last DOM
    // mutation in this chat surface. This covers the raw-session flash without
    // putting the old fixed 500ms sleep back on every warm switch.
    let frame: number | undefined
    let frames = 0
    let quietFrames = 0
    const surface = overlayRef.current?.closest('[data-chat-surface]')
    const observer = surface
      ? new MutationObserver(() => {
          quietFrames = 0
        })
      : null

    observer?.observe(surface!, { attributes: true, characterData: true, childList: true, subtree: true })

    const settleTimer = window.setTimeout(() => {
      const settle = () => {
        frames += 1
        quietFrames += 1

        if (frames >= 6 && quietFrames >= 2) {
          setCovering(false)

          return
        }

        frame = window.requestAnimationFrame(settle)
      }

      frame = window.requestAnimationFrame(settle)
    }, 0)

    return () => {
      window.clearTimeout(settleTimer)
      observer?.disconnect()

      if (frame !== undefined) {
        window.cancelAnimationFrame(frame)
      }
    }
  }, [botMode, covering, profile])

  const coverVisible = Boolean(profile) || (botMode && covering)

  return (
    <div
      aria-hidden
      className={cn(
        'pointer-events-none absolute inset-0 z-50 flex items-center justify-center',
        botMode ? 'bg-(--ui-chat-surface-background)' : '',
        coverVisible ? 'opacity-100' : 'opacity-0 transition-opacity duration-150 ease-out'
      )}
      data-glass-opaque={botMode ? '' : undefined}
      data-slot="chat-swap-overlay"
      ref={overlayRef}
    >
      <div
        className={cn(
          'flex items-center gap-2 bg-[color-mix(in_srgb,var(--dt-card)_92%,transparent)] px-4 py-2 font-mono text-[0.8125rem] text-foreground shadow-composer transition-opacity duration-100',
          statusVisible ? 'opacity-100' : 'opacity-0'
        )}
        data-slot="chat-swap-status"
      >
        {/* Was a local 80ms setInterval + setState braille ticker — the same
            mechanism class (per-tick DOM mutation scheduling style recalc)
            that GlyphSpinner was rewritten to remove. `braille` is exactly the
            frame set and 80ms cadence this used. `justify-start` keeps the
            glyph left-aligned in its w-3 box the way the bare span was, and
            `paused` restores the old "no ticking once the swap is done"
            behaviour while the overlay fades out still mounted. */}
        <GlyphSpinner className="w-3 justify-start text-(--ui-accent)" paused={!profile} spinner="braille" />
        {t.composer.wakingProfile(botMode ? t.composer.botChat : (label ?? ''))}
      </div>
    </div>
  )
}

// Subtle corner badge for a PAINT-FIRST wake (#89843): the stored transcript
// is already on screen and usable, but the active-profile gate hasn't caught
// up yet (shared-remote serves every profile through the primary socket).
// Deliberately quiet — a pill in the corner, not an overlay — because the
// content is real; only the background profile sync is still settling.
export function ChatSyncBadge({ profile }: { profile: string | null }) {
  const { t } = useI18n()

  if (!profile) {
    return null
  }

  return (
    <div
      aria-live="polite"
      className="pointer-events-none absolute right-3 top-2 z-30 flex items-center gap-1.5 rounded-full border border-border/50 bg-[color-mix(in_srgb,var(--dt-card)_92%,transparent)] px-2 py-0.5 font-mono text-[0.6875rem] text-muted-foreground shadow-composer"
    >
      <GlyphSpinner className="w-3 justify-start text-(--ui-accent)" spinner="braille" />
      {t.desktop.hydrationSyncing(profile)}
    </div>
  )
}
