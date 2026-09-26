import { useRef, useState } from 'react'

import { Button } from '@/components/ui/button'
import { useI18n } from '@/i18n'
import { Loader2, Mic, MicOff, Square, VolumeX } from '@/lib/icons'
import { cn } from '@/lib/utils'

import { useMicLevelVar } from './audio-level'
import { DesktopOrbToggle } from './desktop-orb-toggle'
import { VoiceWaveform } from './voice-waveform'

type VoiceAction = () => Promise<void> | void

export interface VoiceControlsProps {
  cancelTask: VoiceAction
  disabled?: boolean
  error?: null | string
  listening: boolean
  loading?: boolean
  /** True when the conversation is running with the microphone muted. */
  muted?: boolean
  speaking: boolean
  startListening: VoiceAction
  stopListening: VoiceAction
  stopPlayback: VoiceAction
  taskRunning: boolean
  /** Omitted when the active conversation cannot be muted. */
  toggleMute?: VoiceAction
}

type PendingAction = 'cancelTask' | 'startListening' | 'stopListening' | 'stopPlayback' | 'toggleMute' | null

export function VoiceControls({
  cancelTask,
  disabled = false,
  error = null,
  listening,
  loading = false,
  muted = false,
  speaking,
  startListening,
  stopListening,
  stopPlayback,
  taskRunning,
  toggleMute
}: VoiceControlsProps) {
  const { t } = useI18n()
  const copy = t.jarvisShell.dashboard.voiceControls
  const [pendingAction, setPendingAction] = useState<PendingAction>(null)
  const meterRef = useRef<HTMLDivElement>(null)
  const busy = loading || pendingAction !== null

  // The meter reads the real recorder through CSS variables: a live level must
  // not re-render the dashboard on every animation frame.
  useMicLevelVar(meterRef, listening && !muted)

  const run = async (action: Exclude<PendingAction, null>, handler: VoiceAction) => {
    if (disabled || busy) {
      return
    }

    setPendingAction(action)

    try {
      await handler()
    } finally {
      setPendingAction(null)
    }
  }

  const listenLabel = listening ? copy.stopListening : copy.startListening
  const listenAction = listening ? stopListening : startListening
  const listenPending = pendingAction === 'startListening' || pendingAction === 'stopListening'
  const muteLabel = muted ? copy.unmute : copy.mute

  return (
    <section
      aria-label={copy.label}
      className="flex min-w-0 flex-wrap items-center gap-2 rounded-xl border border-(--ui-stroke-tertiary) bg-(--ui-bg-secondary)/50 px-2 py-2 backdrop-blur"
      data-testid="jarvis-voice-controls"
    >
      <div
        aria-label={copy.micLevel}
        aria-valuemax={100}
        aria-valuemin={0}
        aria-valuenow={0}
        className="flex min-h-11 items-center gap-1 px-1 text-(--ui-text-secondary)"
        data-testid="jarvis-mic-meter"
        ref={meterRef}
        role="meter"
        style={{ '--jarvis-audio-level': '0' } as React.CSSProperties}
      >
        {/* A scrolling history of measured levels; flat whenever the mic is
            closed or muted. */}
        <VoiceWaveform active={listening && !muted} className="h-8 w-36 sm:w-44" />
      </div>
      <Button
        aria-label={listenLabel}
        aria-pressed={listening}
        className="min-h-11"
        disabled={disabled || busy}
        onClick={() => void run(listening ? 'stopListening' : 'startListening', listenAction)}
        type="button"
        variant={listening ? 'secondary' : 'default'}
      >
        {listenPending ? <Loader2 className="animate-spin" /> : listening ? <Square /> : <Mic />}
        {listenLabel}
      </Button>
      {toggleMute && (
        <Button
          aria-label={muteLabel}
          aria-pressed={muted}
          className="min-h-11"
          disabled={disabled || busy || !listening}
          onClick={() => void run('toggleMute', toggleMute)}
          type="button"
          variant="secondary"
        >
          {pendingAction === 'toggleMute' ? <Loader2 className="animate-spin" /> : muted ? <MicOff /> : <Mic />}
          {muteLabel}
        </Button>
      )}
      <Button
        aria-label={copy.stopSpeaking}
        className="min-h-11"
        disabled={disabled || busy || !speaking}
        onClick={() => void run('stopPlayback', stopPlayback)}
        type="button"
        variant="secondary"
      >
        {pendingAction === 'stopPlayback' ? <Loader2 className="animate-spin" /> : <VolumeX />}
        {copy.stopSpeaking}
      </Button>
      <Button
        aria-label={copy.cancelTask}
        className={cn('min-h-11', taskRunning && 'text-destructive')}
        disabled={disabled || busy || !taskRunning}
        onClick={() => void run('cancelTask', cancelTask)}
        type="button"
        variant="outline"
      >
        {pendingAction === 'cancelTask' ? <Loader2 className="animate-spin" /> : <Square />}
        {copy.cancelTask}
      </Button>
      <DesktopOrbToggle />
      {error ? (
        <p className="basis-full text-xs text-destructive" role="alert">
          {error}
        </p>
      ) : null}
    </section>
  )
}
