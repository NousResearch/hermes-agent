import { useStore } from '@nanostores/react'
import { useEffect, useRef } from 'react'

import { Button } from '@/components/ui/button'
import { Loader2, Mic, Volume2, VolumeX } from '@/lib/icons'
import { cn } from '@/lib/utils'
import { stopVoicePlayback } from '@/lib/voice-playback'
import { $voicePlayback } from '@/store/voice-playback'

import type { VoiceActivityState } from './types'

type BrowserAudioContext = typeof AudioContext

interface ElementAnalyser {
  analyser: AnalyserNode
}

const elementAnalysers = new WeakMap<HTMLAudioElement, ElementAnalyser>()
let playbackAudioContext: AudioContext | null = null

function getPlaybackAudioContext(): AudioContext | null {
  if (playbackAudioContext && playbackAudioContext.state !== 'closed') {
    return playbackAudioContext
  }

  const audioWindow = window as Window & { webkitAudioContext?: BrowserAudioContext }
  const AudioContextCtor = window.AudioContext || audioWindow.webkitAudioContext

  if (!AudioContextCtor) {
    return null
  }

  playbackAudioContext = new AudioContextCtor()

  return playbackAudioContext
}

function formatElapsed(seconds: number) {
  const safeSeconds = Math.max(0, Math.floor(seconds))
  const minutes = Math.floor(safeSeconds / 60)
  const remainingSeconds = safeSeconds % 60

  return `${minutes}:${remainingSeconds.toString().padStart(2, '0')}`
}

function VoiceLevelBars({ level, active }: { active: boolean; level: number }) {
  const normalized = Math.max(0, Math.min(level, 1))
  const bars = [0.5, 0.78, 1, 0.78, 0.5]

  return (
    <div aria-hidden="true" className="flex h-4 items-center gap-0.5">
      {bars.map((weight, index) => {
        const height = active ? 0.25 + Math.min(0.68, normalized * weight) : 0.25

        return (
          <span
            className={cn(
              'w-0.5 rounded-full bg-current transition-[height,opacity] duration-100 ease-out',
              active ? 'opacity-80' : 'animate-pulse opacity-45'
            )}
            key={index}
            style={{ height: `${height * 100}%` }}
          />
        )
      })}
    </div>
  )
}

function getElementAnalyser(audioElement: HTMLAudioElement): ElementAnalyser | null {
  let entry = elementAnalysers.get(audioElement)

  if (!entry) {
    const context = getPlaybackAudioContext()

    if (!context) {
      return null
    }

    const source = context.createMediaElementSource(audioElement)
    const analyser = context.createAnalyser()

    analyser.fftSize = 512
    analyser.smoothingTimeConstant = 0.65
    source.connect(analyser)
    analyser.connect(context.destination)
    entry = { analyser }
    elementAnalysers.set(audioElement, entry)
  }

  void playbackAudioContext?.resume()

  return entry
}

const WAVE_W = 88
const WAVE_H = 16
const BAR_W = 2
const BAR_GAP = 5
const STEP = BAR_W + BAR_GAP
const BARS = Math.floor((WAVE_W + BAR_GAP) / STEP)
const X0 = Math.round((WAVE_W - (BARS * STEP - BAR_GAP)) / 2)

function PlaybackWaveform({ audioElement }: { audioElement: HTMLAudioElement | null }) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null)

  useEffect(() => {
    const canvas = canvasRef.current

    if (!canvas || !audioElement) {
      return
    }

    const entry = getElementAnalyser(audioElement)
    const ctx = canvas.getContext('2d')

    if (!entry || !ctx) {
      return
    }

    const dpr = Math.max(1, window.devicePixelRatio || 1)
    const { analyser } = entry
    const buf = new Uint8Array(analyser.frequencyBinCount)
    const hi = Math.floor(buf.length * 0.9)

    canvas.width = Math.round(WAVE_W * dpr)
    canvas.height = Math.round(WAVE_H * dpr)
    canvas.style.width = `${WAVE_W}px`
    canvas.style.height = `${WAVE_H}px`
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
    ctx.imageSmoothingEnabled = false
    ctx.fillStyle = getComputedStyle(canvas).color

    let raf = 0

    const tick = () => {
      analyser.getByteFrequencyData(buf)
      ctx.clearRect(0, 0, WAVE_W, WAVE_H)

      for (let i = 0; i < BARS; i++) {
        const a = Math.floor((i / BARS) * hi)
        const b = Math.floor(((i + 1) / BARS) * hi)
        let peak = 0

        for (let j = a; j < b; j++) {
          peak = Math.max(peak, buf[j] ?? 0)
        }

        const amp = Math.sqrt(peak / 255)
        const bh = Math.max(3, Math.round((0.18 + amp * 0.82) * WAVE_H))
        ctx.fillRect(X0 + i * STEP, Math.round((WAVE_H - bh) / 2), BAR_W, bh)
      }

      raf = requestAnimationFrame(tick)
    }

    tick()

    return () => cancelAnimationFrame(raf)
  }, [audioElement])

  return <canvas aria-hidden="true" className="block h-4 w-[88px]" ref={canvasRef} />
}

export function VoiceActivity({ state }: { state: VoiceActivityState }) {
  if (state.status === 'idle') {
    return null
  }

  const recording = state.status === 'recording'
  const title = recording ? 'Dictating' : 'Transcribing'

  return (
    <div
      aria-live="polite"
      className={cn(
        'flex h-8 items-center gap-2 rounded-xl border border-border/55 bg-muted/55 px-2.5 text-xs text-muted-foreground',
        'shadow-[inset_0_1px_0_rgba(255,255,255,0.35)] backdrop-blur-sm'
      )}
      role="status"
    >
      <div
        className={cn(
          'flex size-5 shrink-0 items-center justify-center rounded-full',
          recording ? 'bg-primary/15 text-primary' : 'bg-primary/10 text-primary'
        )}
      >
        {recording ? <Mic size={12} /> : <Loader2 className="animate-spin" size={12} />}
      </div>

      <div className="flex min-w-0 flex-1 items-center gap-2">
        <span className="truncate font-medium text-foreground/85">{title}</span>
        <span className="font-mono text-[0.6875rem] text-muted-foreground/85">
          {formatElapsed(state.elapsedSeconds)}
        </span>
      </div>

      <VoiceLevelBars active={recording} level={state.level} />
    </div>
  )
}

export function VoicePlaybackActivity() {
  const playback = useStore($voicePlayback)

  if (playback.status === 'idle') {
    return null
  }

  const preparing = playback.status === 'preparing'

  const title = preparing
    ? 'Preparing audio'
    : playback.source === 'voice-conversation'
      ? 'Speaking response'
      : 'Reading aloud'

  return (
    <div
      aria-live="polite"
      className={cn(
        'flex h-8 items-center gap-2 rounded-xl border border-primary/20 bg-primary/10 px-2.5 text-xs text-primary',
        'shadow-[inset_0_1px_0_rgba(255,255,255,0.35)] backdrop-blur-sm'
      )}
      role="status"
    >
      <div className="flex size-5 shrink-0 items-center justify-center rounded-full bg-primary/15 text-primary">
        {preparing ? <Loader2 className="animate-spin" size={12} /> : <Volume2 size={12} />}
      </div>

      <div className="flex min-w-0 flex-1 items-center gap-2">
        <span className="truncate font-medium text-foreground/85">{title}</span>
        {!preparing && <PlaybackWaveform audioElement={playback.audioElement} />}
      </div>

      <Button
        className="h-6 shrink-0 gap-1 rounded-full px-2 text-[0.6875rem]"
        onClick={stopVoicePlayback}
        size="sm"
        type="button"
        variant="ghost"
      >
        <VolumeX size={12} />
        Stop
      </Button>
    </div>
  )
}
