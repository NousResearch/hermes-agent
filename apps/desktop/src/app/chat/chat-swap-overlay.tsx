import { useEffect, useState } from 'react'

import { cn } from '@/lib/utils'

// Braille spinner frames — reads as a tiny ASCII loader in monospace.
const FRAMES = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']

// Shown over the conversation while the live gateway swaps to another profile's
// backend (lazily spawned). Keeps the last profile name through the fade-out so
// the label doesn't blank. Purely visual — pointer-events-none.
export function ChatSwapOverlay({ profile }: { profile: string | null }) {
  const [frame, setFrame] = useState(0)
  const [label, setLabel] = useState<null | string>(profile)

  useEffect(() => {
    if (profile) {
      setLabel(profile)
    }
  }, [profile])

  useEffect(() => {
    if (!profile) {
      return
    }

    const id = window.setInterval(() => setFrame(value => (value + 1) % FRAMES.length), 80)

    return () => window.clearInterval(id)
  }, [profile])

  return (
    <div
      aria-hidden
      className={cn(
        'pointer-events-none absolute inset-0 z-50 flex items-center justify-center transition-opacity duration-150 ease-out',
        profile ? 'opacity-100' : 'opacity-0'
      )}
    >
      <div className="flex items-center gap-2 rounded-full border border-[color-mix(in_srgb,var(--dt-composer-ring)_45%,transparent)] bg-[color-mix(in_srgb,var(--dt-card)_92%,transparent)] px-4 py-2 font-mono text-[0.8125rem] text-foreground shadow-composer">
        <span className="w-3 text-(--ui-accent)">{FRAMES[frame]}</span>
        Waking up {label}…
      </div>
    </div>
  )
}
