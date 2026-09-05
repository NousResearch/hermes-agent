import { useEffect, useState } from 'react'

import { GlyphSpinner } from '@/components/ui/glyph-spinner'
import { useI18n } from '@/i18n'
import { cn } from '@/lib/utils'

// Shown over the conversation while the live gateway swaps to another profile's
// backend (lazily spawned). Keeps the last profile name through the fade-out so
// the label doesn't blank. Purely visual — pointer-events-none.
export function ChatSwapOverlay({ profile }: { profile: string | null }) {
  const { t } = useI18n()
  const [label, setLabel] = useState<null | string>(profile)

  useEffect(() => {
    if (profile) {
      setLabel(profile)
    }
  }, [profile])

  return (
    <div
      aria-hidden
      className={cn(
        'pointer-events-none absolute inset-0 z-50 flex items-center justify-center transition-opacity duration-150 ease-out',
        profile ? 'opacity-100' : 'opacity-0'
      )}
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
        {t.composer.wakingProfile(label ?? '')}
      </div>
    </div>
  )
}
