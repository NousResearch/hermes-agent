import './core.css'

import { type CSSProperties, useEffect, useId, useMemo, useRef, useState } from 'react'

import { useI18n } from '@/i18n'
import { cn } from '@/lib/utils'
import { useTheme } from '@/themes/context'

import { clampAudioLevel, jarvisAudioVars, useJarvisAudioBinding } from './audio-level'
import { type PlasmaSurface, plasmaTone } from './plasma'
import { PlasmaCanvas } from './plasma-canvas'
import type { JarvisTaskPhase, JarvisVoiceState } from './types'
import { VoiceAura } from './voice-aura'

export interface JarvisCoreProps {
  voice: JarvisVoiceState
  taskPhase: JarvisTaskPhase
  /** A measured level in 0…1. Ignored unless the voice state is an audio one. */
  audioLevel?: number
  className?: string
  compact?: boolean
  /**
   * Bind to the live microphone store instead of the `audioLevel` prop. The
   * binding writes CSS variables straight to the DOM, so the pulse tracks the
   * real meter without re-rendering this tree — or the chat around it — once
   * per animation frame.
   */
  live?: boolean
  /** Override the painted surface when a host owns its background. */
  surface?: PlasmaSurface
  /**
   * `hero` is the home screen's centrepiece: a larger, denser orb floating
   * over the dashboard backdrop. `default` is the compact status orb.
   */
  variant?: 'default' | 'hero'
}

/** How strongly each task phase drives the core's non-audio motion. */
const TASK_SIGNAL: Record<JarvisTaskPhase, number> = {
  approval: 0.28,
  cancelled: 0.14,
  cancelling: 0.58,
  failed: 0.2,
  idle: 0,
  planning: 0.46,
  running: 0.72,
  verified: 0.38
}

function formatNumber(value: number): string {
  return Number(value.toFixed(3)).toString()
}

function formatPx(value: number): string {
  return `${formatNumber(value)}px`
}

function safeSvgIdPrefix(id: string): string {
  const suffix = id.replace(/[^a-zA-Z0-9_-]/g, '')

  return `jarvis-core-${suffix || 'svg'}`
}

function reducedMotionPreference() {
  return window.matchMedia?.('(prefers-reduced-motion: reduce)').matches ?? false
}

function useReducedMotion() {
  const [reduced, setReduced] = useState(reducedMotionPreference)

  useEffect(() => {
    const media = window.matchMedia?.('(prefers-reduced-motion: reduce)')

    if (!media) {
      return undefined
    }

    const update = () => setReduced(media.matches)

    if (media.addEventListener) {
      media.addEventListener('change', update)
    } else {
      media.addListener?.(update)
    }

    return () => {
      if (media.removeEventListener) {
        media.removeEventListener('change', update)
      } else {
        media.removeListener?.(update)
      }
    }
  }, [])

  return reduced
}

export function JarvisCore({
  audioLevel,
  className,
  compact = false,
  live = false,
  surface,
  taskPhase,
  variant = 'default',
  voice
}: JarvisCoreProps) {
  const { t } = useI18n()
  const copy = t.jarvisShell.dashboard.core
  const reactId = useId()
  const rootRef = useRef<HTMLDivElement>(null)
  const reducedMotion = useReducedMotion()
  // What is actually painted behind the orb (a "dark" skin can keep a light surface).
  const { renderedMode } = useTheme()
  const audioActive = voice === 'listening' || voice === 'speaking'
  // A task running with the mic closed must not fake a microphone reading:
  // only an audio voice state may drive amplitude.
  const reactiveAudioLevel = audioActive ? clampAudioLevel(audioLevel) : 0
  const taskSignal = TASK_SIGNAL[taskPhase]
  const hero = variant === 'hero'
  const size = hero ? (compact ? 356 : 550) : compact ? 184 : 364
  // Until the canvas proves it can paint, the SVG liquid stays as the fallback.
  const [plasmaReady, setPlasmaReady] = useState(false)
  const taskCopy = copy.task[taskPhase]
  const label = taskCopy ? copy.both(copy.voice[voice], taskCopy) : copy.voiceOnly(copy.voice[voice])

  const svgIds = useMemo(() => {
    const prefix = safeSvgIdPrefix(reactId)

    return {
      glass: `${prefix}-glass`,
      liquid: `${prefix}-liquid`,
      soft: `${prefix}-soft`
    }
  }, [reactId])

  // When `live`, the binding owns the audio variables from mount onward; the
  // inline style below still renders a correct resting first frame.
  useJarvisAudioBinding(rootRef, { active: audioActive, enabled: live, taskSignal })

  const style = {
    ...jarvisAudioVars(reactiveAudioLevel, taskSignal),
    // The hero takes its size from the home screen (which fits it to the
    // column and to short windows), falling back to the same cap here.
    '--jarvis-core-size': hero ? `var(--jarvis-hero-size, min(${size}px, 42vh))` : `${size}px`,
    '--jarvis-state-size': formatPx(Math.max(12, size * 0.055))
  } as CSSProperties

  return (
    <div
      aria-label={label}
      className={cn('jarvis-core', className)}
      data-audio-active={audioActive ? 'true' : 'false'}
      data-compact={compact ? 'true' : 'false'}
      data-motion={reducedMotion ? 'reduced' : 'full'}
      data-plasma={plasmaReady ? 'on' : 'off'}
      data-task={taskPhase}
      data-testid="jarvis-core"
      data-variant={variant}
      data-voice={voice}
      ref={rootRef}
      role="status"
      style={style}
    >
      <span className="jarvis-core__sr">{label}</span>
      <span aria-hidden="true" className="jarvis-core__stage">
        {/* Ripples: emitted only while the mic or the speaker is open, and
            scaled by the measured level, so silence really does look silent. */}
        <span className="jarvis-core__pulse jarvis-core__pulse--1" />
        <span className="jarvis-core__pulse jarvis-core__pulse--2" />
        <span className="jarvis-core__pulse jarvis-core__pulse--3" />
        <span className="jarvis-core__halo jarvis-core__halo--primary" />
        <span className="jarvis-core__halo jarvis-core__halo--accent" />
        <svg className="jarvis-core__orb" focusable="false" viewBox="0 0 200 200">
          <defs>
            <radialGradient cx="34%" cy="26%" id={svgIds.glass} r="76%">
              <stop offset="0%" stopColor="#FFFFFF" stopOpacity="0.5" />
              <stop offset="36%" stopColor="#8ED8FF" stopOpacity="0.2" />
              <stop offset="70%" stopColor="#0C1622" stopOpacity="0.56" />
              <stop offset="100%" stopColor="#04060A" stopOpacity="0.9" />
            </radialGradient>
            <linearGradient id={svgIds.liquid} x1="32" x2="168" y1="150" y2="54">
              <stop offset="0%" stopColor="#29E68C" stopOpacity="0.82" />
              <stop offset="46%" stopColor="#00B7FF" stopOpacity="0.8" />
              <stop offset="100%" stopColor="#7C5CFF" stopOpacity="0.72" />
            </linearGradient>
            <filter colorInterpolationFilters="sRGB" id={svgIds.soft}>
              <feGaussianBlur stdDeviation="4" />
            </filter>
          </defs>
          <circle className="jarvis-core__glass" cx="100" cy="100" fill={`url(#${svgIds.glass})`} r="76" />
          <path
            className="jarvis-core__liquid"
            d="M38 121 C58 96 76 112 95 90 C116 66 140 72 164 90 C158 133 132 157 98 158 C70 159 50 146 38 121 Z"
            fill={`url(#${svgIds.liquid})`}
            filter={`url(#${svgIds.soft})`}
          />
          <path
            className="jarvis-core__current"
            d="M54 84 C80 57 118 56 146 83 M46 116 C76 139 119 143 154 116"
            fill="none"
          />
          <circle className="jarvis-core__inner-ring" cx="100" cy="100" r="55" />
          <circle className="jarvis-core__level-ring" cx="100" cy="100" r="72" />
          <path className="jarvis-core__shine" d="M62 58 C76 42 107 35 129 48" fill="none" />
        </svg>
      </span>
      {/* A sibling of the stage, not a child, so the halo layers stay inside
          the stage's clip while the plasma paints over the whole core. */}
      <PlasmaCanvas
        audioActive={audioActive}
        audioLevel={reactiveAudioLevel}
        hero={hero}
        live={live}
        onReady={setPlasmaReady}
        reducedMotion={reducedMotion}
        signal={taskSignal}
        surface={surface ?? renderedMode}
        tone={plasmaTone(voice, taskPhase)}
      />
      {/* The voice ring: only a live core owns the microphone binding. */}
      {live ? <VoiceAura active={audioActive} reducedMotion={reducedMotion} /> : null}
    </div>
  )
}
