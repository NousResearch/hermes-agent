import { useRef, useState } from 'react'

import { Button } from '@/components/ui/button'
import { useI18n } from '@/i18n'
import { Loader2, Mic, MicOff, Square, VolumeX } from '@/lib/icons'
import { cn } from '@/lib/utils'

import { useMicLevelVar } from './audio-level'
import { VoiceWaveform } from './voice-waveform'

type VoiceAction = () => Promise<void> | void

export interface VoiceControlsProps {
  cancelTask: VoiceAction
  disabled?: boolean
  error?: null | string
  listening: boolean
  loading?: boolean
  /** True when the conversation is running with its microphone muted. */
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

/**
 * The conversation's voice dock: the meter first, then the four controls a live
 * conversation actually needs — the microphone, its mute, the playback stop and
 * the task stop. Nothing else belongs here.
 *
 * Only one control is a microphone: the mute button wears the slashed icon in
 * both states, so an idle dock never shows two identical microphones side by
 * side (the duplicate this row used to have).
 */
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

  // The meter reads the real recorder through CSS variables. Live input must
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
  const iconButton = 'jarvis-icon-btn size-10 min-h-10 min-w-10 rounded-full'

  return (
    <section
      aria-label={copy.label}
      className="jarvis-voice-dock mx-auto flex w-fit max-w-full items-center border border-(--ui-stroke-tertiary) px-2 py-1.5 backdrop-blur-2xl"
      data-testid="jarvis-voice-controls"
    >
      <div
        aria-label={copy.micLevel}
        aria-valuemax={100}
        aria-valuemin={0}
        aria-valuenow={0}
        className="jarvis-voice-dock__meter flex items-center overflow-hidden px-1 text-(--ui-text-secondary)"
        data-testid="jarvis-mic-meter"
        ref={meterRef}
        role="meter"
        style={{ '--jarvis-audio-level': '0' } as React.CSSProperties}
      >
        <VoiceWaveform active={listening && !muted} className="h-7 w-full" />
      </div>

      <Button
        aria-label={listenLabel}
        aria-pressed={listening}
        className={cn(iconButton, listening && 'jarvis-icon-btn--live')}
        disabled={disabled || busy}
        onClick={() => void run(listening ? 'stopListening' : 'startListening', listenAction)}
        size="icon"
        title={listenLabel}
        type="button"
      >
        {listenPending ? <Loader2 className="animate-spin" /> : listening ? <Square /> : <Mic />}
      </Button>

      {toggleMute && (
        <Button
          aria-label={muteLabel}
          aria-pressed={muted}
          className={cn(iconButton, muted && 'jarvis-icon-btn--live')}
          disabled={disabled || busy || !listening}
          onClick={() => void run('toggleMute', toggleMute)}
          size="icon"
          title={muteLabel}
          type="button"
        >
          {pendingAction === 'toggleMute' ? <Loader2 className="animate-spin" /> : <MicOff />}
        </Button>
      )}

      <Button
        aria-label={copy.stopSpeaking}
        className={iconButton}
        disabled={disabled || busy || !speaking}
        onClick={() => void run('stopPlayback', stopPlayback)}
        size="icon"
        title={copy.stopSpeaking}
        type="button"
      >
        {pendingAction === 'stopPlayback' ? <Loader2 className="animate-spin" /> : <VolumeX />}
      </Button>

      <Button
        aria-label={copy.cancelTask}
        className={cn(iconButton, taskRunning && 'jarvis-icon-btn--danger')}
        disabled={disabled || busy || !taskRunning}
        onClick={() => void run('cancelTask', cancelTask)}
        size="icon"
        title={copy.cancelTask}
        type="button"
      >
        {pendingAction === 'cancelTask' ? <Loader2 className="animate-spin" /> : <Square />}
      </Button>

      {error ? (
        <p className="sr-only" role="alert">
          {error}
        </p>
      ) : null}
    </section>
  )
}
