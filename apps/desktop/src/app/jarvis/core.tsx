import { useEffect, useId, useMemo, useState, type CSSProperties } from 'react'

import type { JarvisTaskPhase, JarvisVoiceState } from './types'

import './core.css'

export interface JarvisCoreProps {
  voice: JarvisVoiceState
  taskPhase: JarvisTaskPhase
  audioLevel?: number
  compact?: boolean
}

const VOICE_COPY: Record<JarvisVoiceState, string> = {
  idle: 'czeka',
  listening: 'słucha',
  speaking: 'mówi',
  error: 'ma problem z głosem'
}

const TASK_COPY: Record<JarvisTaskPhase, string> = {
  idle: '',
  planning: 'planuje zadanie',
  running: 'wykonuje zadanie',
  approval: 'czeka na zatwierdzenie',
  cancelling: 'zatrzymuje zadanie',
  cancelled: 'zatrzymał zadanie',
  failed: 'zgłasza błąd zadania',
  verified: 'zweryfikował rezultat'
}

const TASK_SIGNAL: Record<JarvisTaskPhase, number> = {
  idle: 0,
  planning: 0.46,
  running: 0.72,
  approval: 0.28,
  cancelling: 0.58,
  cancelled: 0.14,
  failed: 0.2,
  verified: 0.38
}

function clampAudioLevel(value: number | undefined): number {
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    return 0
  }

  return Math.min(1, Math.max(0, value))
}

function formatNumber(value: number): string {
  return Number(value.toFixed(3)).toString()
}

function formatPx(value: number): string {
  return `${formatNumber(value)}px`
}

function ariaLabelFor(voice: JarvisVoiceState, taskPhase: JarvisTaskPhase): string {
  const voiceCopy = VOICE_COPY[voice]
  const taskCopy = TASK_COPY[taskPhase]

  return taskCopy ? `Jarvis ${voiceCopy} i ${taskCopy}` : `Jarvis ${voiceCopy}`
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

export function JarvisCore({ audioLevel, compact = false, taskPhase, voice }: JarvisCoreProps) {
  const reactId = useId()
  const reducedMotion = useReducedMotion()
  const audioActive = voice === 'listening' || voice === 'speaking'
  const reactiveAudioLevel = audioActive ? clampAudioLevel(audioLevel) : 0
  const taskSignal = TASK_SIGNAL[taskPhase]
  const size = compact ? 148 : 244
  const label = ariaLabelFor(voice, taskPhase)
  const svgIds = useMemo(() => {
    const prefix = safeSvgIdPrefix(reactId)
    return {
      glass: `${prefix}-glass`,
      liquid: `${prefix}-liquid`,
      soft: `${prefix}-soft`
    }
  }, [reactId])

  const energyScale = 1 + reactiveAudioLevel * 0.36
  const liquidShift = (0.5 - reactiveAudioLevel) * 6

  const style = {
    '--jarvis-audio-level': formatNumber(reactiveAudioLevel),
    '--jarvis-audio-scale-max': formatNumber(1 + reactiveAudioLevel * 0.46),
    '--jarvis-audio-scale-min': formatNumber(1 + reactiveAudioLevel * 0.18),
    '--jarvis-breathe-scale-max': formatNumber(energyScale * 1.08),
    '--jarvis-breathe-scale-min': formatNumber(energyScale * 0.96),
    '--jarvis-counter-scale-max': formatNumber(energyScale * 1.02),
    '--jarvis-counter-scale-min': formatNumber(energyScale * 0.88),
    '--jarvis-core-size': `${size}px`,
    '--jarvis-current-opacity': formatNumber(0.32 + taskSignal * 0.52),
    '--jarvis-energy-scale': formatNumber(energyScale),
    '--jarvis-halo-opacity': formatNumber(0.34 + reactiveAudioLevel * 0.38 + taskSignal * 0.16),
    '--jarvis-liquid-opacity': formatNumber(0.58 + reactiveAudioLevel * 0.26),
    '--jarvis-liquid-shift': formatPx(liquidShift),
    '--jarvis-liquid-shift-crest': formatPx(liquidShift - 4),
    '--jarvis-state-size': formatPx(Math.max(12, size * 0.055)),
    '--jarvis-task-signal': formatNumber(taskSignal)
  } as CSSProperties

  return (
    <div
      aria-label={label}
      className="jarvis-core"
      data-audio-active={audioActive ? 'true' : 'false'}
      data-compact={compact ? 'true' : 'false'}
      data-motion={reducedMotion ? 'reduced' : 'full'}
      data-task={taskPhase}
      data-testid="jarvis-core"
      data-voice={voice}
      role="status"
      style={style}
    >
      <span className="jarvis-core__sr">{label}</span>
      <span aria-hidden="true" className="jarvis-core__stage">
        <span className="jarvis-core__halo jarvis-core__halo--cyan" />
        <span className="jarvis-core__halo jarvis-core__halo--green" />
        <svg className="jarvis-core__orb" focusable="false" viewBox="0 0 200 200">
          <defs>
            <radialGradient cx="34%" cy="26%" id={svgIds.glass} r="76%">
              <stop offset="0%" stopColor="#FFFFFF" stopOpacity="0.62" />
              <stop offset="36%" stopColor="#A9F8FF" stopOpacity="0.22" />
              <stop offset="70%" stopColor="#0B2028" stopOpacity="0.54" />
              <stop offset="100%" stopColor="#030506" stopOpacity="0.88" />
            </radialGradient>
            <linearGradient id={svgIds.liquid} x1="32" x2="168" y1="150" y2="54">
              <stop offset="0%" stopColor="#7CFF1E" stopOpacity="0.9" />
              <stop offset="42%" stopColor="#00E7FF" stopOpacity="0.78" />
              <stop offset="100%" stopColor="#7C5CFF" stopOpacity="0.68" />
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
          <circle className="jarvis-core__outer-ring" cx="100" cy="100" r="88" />
          <path className="jarvis-core__shine" d="M62 58 C76 42 107 35 129 48" fill="none" />
        </svg>
        <span className="jarvis-core__state jarvis-core__state--voice" />
        <span className="jarvis-core__state jarvis-core__state--task" />
      </span>
    </div>
  )
}
