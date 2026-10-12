import { useStore } from '@nanostores/react'
import { useEffect, useState } from 'react'

import { AlertCircle, Clock, type IconComponent } from '@/lib/icons'
import { $petActivity, $petState, type PetState } from '@/store/pet'
import { $petPluginMessage, type PetMessageTone } from '@/store/pet-plugin-messages'

/**
 * Speech bubble + status glyph for the pet — the "notification" half of the
 * mascot. It externalizes what the agent is doing (Codex-style) so a glance at
 * the desktop pet replaces switching back to the window. The core status lines
 * show only in the popped-out overlay (`showStatus`); in-window the app itself
 * is the surface, so the in-window pet renders the bubble for plugin lines only.
 *
 * Plugin lines (`ctx.pet.say`, see store/pet-plugin-messages) share the bubble:
 * the newest live line shows with its plugin's name as a small label, unless
 * the agent is in an error or waiting-on-you state — those keep the bubble.
 *
 * Text is derived purely from the same `$petState` / `$petActivity` the sprite
 * already reacts to, so it never drifts from the animation. The bubble is shown
 * only when there's something worth saying (working / reviewing / a transient
 * done/error beat / waiting on the user) and is hidden at plain idle.
 */

type Tone = PetMessageTone

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
  info: 'currentColor',
  wait: 'var(--ui-yellow)'
}

const TONE_GLYPH: Partial<Record<Tone, IconComponent>> = { error: AlertCircle, wait: Clock }

// Core states that outrank a plugin line: the pet is flagging trouble or the
// turn is paused on the user, and a plugin must not talk over that.
const PRIORITY_TONES: ReadonlySet<Tone | undefined> = new Set(['error', 'wait'])

const BUBBLE_SURFACE = {
  // Solid, theme-driven surface (the prior --ui-bg-card mixes in
  // `transparent`, so the bubble was see-through).
  background: 'var(--ui-bg-elevated)',
  border: '1px solid var(--ui-stroke-secondary)',
  boxShadow: '0 4px 14px rgba(0,0,0,0.22)',
  color: 'var(--foreground)',
  fontSize: 11,
  fontWeight: 500,
  pointerEvents: 'none'
} as const

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

export interface PetBubbleProps {
  /** Render the core status lines (working…, your turn). The overlay passes
   *  true; the in-window pet leaves them to the app and shows plugin lines only. */
  showStatus?: boolean
}

export function PetBubble({ showStatus = true }: PetBubbleProps = {}) {
  const state = useStore($petState)
  const activity = useStore($petActivity)
  const pluginLine = useStore($petPluginMessage)
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

  if (pluginLine && !PRIORITY_TONES.has(spec?.tone)) {
    const Glyph = TONE_GLYPH[pluginLine.tone]

    return (
      <div
        data-pet-plugin={pluginLine.pluginId}
        data-slot="pet-plugin-bubble"
        style={{
          ...BUBBLE_SURFACE,
          borderRadius: 10,
          display: 'inline-flex',
          flexDirection: 'column',
          gap: 3,
          lineHeight: 1.3,
          maxWidth: 220,
          padding: '5px 8px',
          width: 'max-content'
        }}
      >
        <span
          style={{
            color: 'var(--ui-text-secondary, var(--muted-foreground))',
            fontSize: 9,
            fontWeight: 600,
            letterSpacing: 0.2,
            lineHeight: 1,
            overflow: 'hidden',
            textOverflow: 'ellipsis',
            whiteSpace: 'nowrap'
          }}
        >
          {pluginLine.pluginName}
        </span>
        <span style={{ alignItems: 'flex-start', display: 'inline-flex', gap: 5, overflowWrap: 'anywhere' }}>
          {Glyph && (
            <span style={{ display: 'inline-flex', flex: 'none', paddingTop: 1 }}>
              <Glyph style={{ color: TONE_COLOR[pluginLine.tone], height: 12, width: 12 }} />
            </span>
          )}
          {pluginLine.text}
        </span>
      </div>
    )
  }

  if (!spec || !showStatus) {
    return null
  }

  const Glyph = spec.glyph
  const text = line || spec.lines[0]
  const hasText = Boolean(text)

  return (
    <div
      style={{
        ...BUBBLE_SURFACE,
        alignItems: 'center',
        borderRadius: hasText ? 10 : 999,
        display: 'inline-flex',
        gap: hasText ? 5 : 0,
        lineHeight: 1,
        // Glyph-only bubbles collapse to a tight, symmetric badge.
        padding: hasText ? '5px 8px' : 5,
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
