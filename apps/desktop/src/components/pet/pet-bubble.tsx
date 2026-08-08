import { useStore } from '@nanostores/react'
import { useEffect, useState } from 'react'

import { useI18n } from '@/i18n'
import { AlertCircle, Clock, type IconComponent } from '@/lib/icons'
import { $petActivity, $petState, type PetState } from '@/store/pet'
import { $petOverlayApproval } from '@/store/pet-overlay'

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

function petSpecs(copy: ReturnType<typeof useI18n>['t']['petBubble']): Partial<Record<PetState, Spec>> {
  return {
    run: {
      lines: copy.runLines
    },
    review: {
      lines: copy.reviewLines
    },
    failed: {
      glyph: AlertCircle,
      lines: copy.failedLines,
      tone: 'error'
    },
    waiting: {
      glyph: Clock,
      lines: copy.waitingLines,
      tone: 'wait'
    }
  }
}

const TONE_COLOR: Record<Tone, string> = {
  error: 'var(--ui-red)',
  wait: 'var(--ui-yellow)'
}

export function summarizePetApproval(command: string, description: string, fallback: string): string {
  return command.trim() || description.trim() || fallback
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
  const { t } = useI18n()
  const copy = t.petBubble
  const specs = petSpecs(copy)
  const state = useStore($petState)
  const activity = useStore($petActivity)
  const approval = useStore($petOverlayApproval)
  const [line, setLine] = useState('')
  const [submitting, setSubmitting] = useState<'deny' | 'once' | null>(null)

  // Finish beats are carried by the sprite/mail icon; idle only speaks up when
  // it's actually the user's turn. Everything else maps to a mood spec.
  const specKey: null | PetState =
    state in specs ? state : state === 'idle' && activity.awaitingInput ? 'waiting' : null

  const rotating = specKey === 'run' || specKey === 'review'

  // Pick a fresh line on every mood change, then keep rotating (random, no
  // repeat) only while the agent is actively working/thinking.
  useEffect(() => {
    const spec = specKey ? specs[specKey] : null

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

  const respond = (choice: 'deny' | 'once') => {
    if (!approval || submitting) {
      return
    }

    setSubmitting(choice)
    window.hermesDesktop?.petOverlay?.control({
      choice,
      sessionId: approval.sessionId,
      type: 'approval'
    })
  }

  // A new or cleared approval resets any in-flight submit state, so a follow-up
  // request never inherits a stale disabled button.
  useEffect(() => {
    setSubmitting(null)
  }, [approval])

  // The store only clears the approval after the gateway confirms success and
  // deliberately keeps the prompt on failure. Release the buttons again after a
  // short window so a failed respond leaves the overlay actionable instead of
  // permanently disabled.
  useEffect(() => {
    if (!submitting) {
      return
    }

    const id = window.setTimeout(() => setSubmitting(null), 5000)

    return () => window.clearTimeout(id)
  }, [submitting])

  if (approval) {
    return (
      <div
        style={{
          background: 'var(--ui-bg-elevated)',
          border: '1px solid var(--ui-stroke-secondary)',
          borderRadius: 10,
          boxShadow: '0 4px 14px rgba(0,0,0,0.22)',
          color: 'var(--foreground)',
          display: 'flex',
          flexDirection: 'column',
          fontSize: 11,
          gap: 6,
          maxWidth: 250,
          padding: '7px 9px',
          pointerEvents: 'auto'
        }}
      >
        <strong>{copy.approvalTitle}</strong>
        <pre
          style={{
            background: 'var(--ui-chat-surface-background)',
            border: '1px solid var(--ui-stroke-tertiary)',
            borderRadius: 6,
            fontFamily: 'var(--font-mono)',
            lineHeight: 1.35,
            margin: 0,
            maxHeight: 120,
            overflow: 'auto',
            padding: '5px 6px',
            whiteSpace: 'pre-wrap',
            wordBreak: 'break-word'
          }}
        >
          {summarizePetApproval(approval.command, approval.description, copy.approvalFallback)}
        </pre>
        <span style={{ display: 'flex', gap: 6 }}>
          <button disabled={Boolean(submitting)} onClick={() => respond('once')} type="button">
            {submitting === 'once' ? copy.processing : copy.approveOnce}
          </button>
          <button disabled={Boolean(submitting)} onClick={() => respond('deny')} type="button">
            {submitting === 'deny' ? copy.processing : copy.deny}
          </button>
        </span>
      </div>
    )
  }

  const spec = specKey ? specs[specKey] : null

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
