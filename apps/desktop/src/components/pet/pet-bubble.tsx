import { useStore } from '@nanostores/react'
import { useEffect, useState } from 'react'

import { AlertCircle, Clock, type IconComponent } from '@/lib/icons'
import { $petActivity, $petState, type PetState } from '@/store/pet'

/**
 * Speech bubble + status glyph for the popped-out pet overlay — the
 * "notification" half of the mascot. It externalizes what the agent is doing
 * (Codex-style) so a glance at the desktop pet replaces switching back to the
 * window. The in-window pet doesn't show it (the app itself is the surface);
 * only the overlay renders it.
 *
 * Text is derived purely from the same `$petState` / `$petActivity` the sprite
 * already reacts to, so it never drifts from the animation. The bubble is shown
 * only when there's something worth saying (working / reviewing / a transient
 * done/error beat / waiting on the user) and is hidden at plain idle.
 */

type Tone = 'error' | 'wait'

interface Spec {
  lines: string[]
  glyph?: IconComponent
  tone?: Tone
}

// Phrasings per mood, picked at random (no immediate repeat) for a bit of life.
// Keep them short — the bubble is tiny and never wraps.
const SPECS: Partial<Record<PetState, Spec>> = {
  run: {
    lines: [
      'working…',
      'on it…',
      'crunching…',
      'tinkering…',
      'cooking…',
      'in the weeds…',
      'wiring it up…',
      'making moves…',
      'heads down…',
      'hammering away…'
    ]
  },
  review: {
    lines: [
      'thinking…',
      'reading…',
      'reviewing…',
      'pondering…',
      'connecting dots…',
      'sizing it up…',
      'tracing it…',
      'mulling…',
      'scheming…',
      'hmm…'
    ]
  },
  failed: {
    glyph: AlertCircle,
    lines: ['hit a snag', 'welp', 'that broke', 'oof', 'snagged'],
    tone: 'error'
  },
  waiting: {
    glyph: Clock,
    lines: ['your turn', 'all yours', 'over to you', 'ball’s in your court', 'awaiting orders'],
    tone: 'wait'
  }
}

const TONE_COLOR: Record<Tone, string> = {
  error: 'var(--ui-red)',
  wait: 'var(--ui-yellow)'
}

// Random pick that avoids repeating the line we're already showing.
function pick(lines: string[], prev: string): string {
  if (lines.length <= 1) {
    return lines[0] ?? ''
  }

  let next = prev

  while (next === prev) {
    next = lines[Math.floor(Math.random() * lines.length)]
  }

  return next
}

export function PetBubble() {
  const state = useStore($petState)
  const activity = useStore($petActivity)
  const [line, setLine] = useState('')

  // Finish beats are carried by the sprite/mail icon; idle only speaks up when
  // it's actually the user's turn. Everything else maps to a mood spec.
  const specKey: null | PetState =
    state in SPECS ? state : state === 'idle' && activity.awaitingInput ? 'waiting' : null

  const rotating = specKey === 'run' || specKey === 'review'

  // Pick a fresh line on every mood change, then keep rotating (random, no
  // repeat) only while the agent is actively working/thinking.
  useEffect(() => {
    const spec = specKey ? SPECS[specKey] : null

    if (!spec) {
      setLine('')

      return
    }

    setLine(prev => pick(spec.lines, prev))

    if (!rotating || spec.lines.length <= 1) {
      return
    }

    const id = window.setInterval(() => setLine(prev => pick(spec.lines, prev)), 2600)

    return () => window.clearInterval(id)
  }, [specKey, rotating])

  const spec = specKey ? SPECS[specKey] : null

  if (!spec) {
    return null
  }

  const Glyph = spec.glyph
  const text = line || spec.lines[0]
  const hasText = Boolean(text)

  return (
    <div
      style={{
        alignItems: 'center',
        // Solid, theme-driven surface (the prior --ui-bg-card mixes in
        // `transparent`, so the bubble was see-through).
        background: 'var(--ui-bg-elevated)',
        border: '1px solid var(--ui-stroke-secondary)',
        borderRadius: hasText ? 10 : 999,
        boxShadow: '0 4px 14px rgba(0,0,0,0.22)',
        color: 'var(--foreground)',
        display: 'inline-flex',
        fontSize: 11,
        fontWeight: 500,
        gap: hasText ? 5 : 0,
        lineHeight: 1,
        // Glyph-only bubbles collapse to a tight, symmetric badge.
        padding: hasText ? '5px 8px' : 5,
        pointerEvents: 'none',
        whiteSpace: 'nowrap'
      }}
    >
      {Glyph && (
        <span style={{ display: 'inline-flex' }}>
          <Glyph style={{ color: spec.tone ? TONE_COLOR[spec.tone] : 'currentColor', height: 13, width: 13 }} />
        </span>
      )}
      {text}
    </div>
  )
}
