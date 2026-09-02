import { useEffect, useState } from 'react'

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

  useEffect(() => {
    if (profile) {
      setLabel(profile)
      setCovering(true)

      return
    }

    if (!covering) {
      return
    }

    // The message store can settle before React commits and scroll-restores a
    // giant transcript. Preserve one fully painted cover, then begin the fade;
    // otherwise the browser exposes the transcript's intermediate layout.
    let secondFrame: number | undefined

    const firstFrame = window.requestAnimationFrame(() => {
      secondFrame = window.requestAnimationFrame(() => setCovering(false))
    })

    return () => {
      window.cancelAnimationFrame(firstFrame)

      if (secondFrame !== undefined) {
        window.cancelAnimationFrame(secondFrame)
      }
    }
  }, [covering, profile])

  const coverVisible = Boolean(profile) || (botMode && covering)

  return (
    <div
      aria-hidden
      className={cn(
        'pointer-events-none absolute inset-0 z-50 flex items-center justify-center transition-opacity duration-150 ease-out',
        botMode && coverVisible ? 'bg-(--ui-chat-surface-background)' : '',
        coverVisible ? 'opacity-100' : 'opacity-0'
      )}
      data-glass-opaque={botMode && coverVisible ? '' : undefined}
      data-slot="chat-swap-overlay"
    >
      <div className="flex items-center gap-2 bg-[color-mix(in_srgb,var(--dt-card)_92%,transparent)] px-4 py-2 font-mono text-[0.8125rem] text-foreground shadow-composer">
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
