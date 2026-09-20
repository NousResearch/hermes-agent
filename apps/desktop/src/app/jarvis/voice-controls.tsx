import { useState } from 'react'

import { Button } from '@/components/ui/button'
import { useI18n } from '@/i18n'
import { Loader2, Mic, Square, VolumeX } from '@/lib/icons'
import { cn } from '@/lib/utils'

type VoiceAction = () => Promise<void> | void

export interface VoiceControlsProps {
  audioLevel: number
  cancelTask: VoiceAction
  disabled?: boolean
  error?: string | null
  listening: boolean
  loading?: boolean
  speaking: boolean
  startListening: VoiceAction
  stopListening: VoiceAction
  stopPlayback: VoiceAction
  taskRunning: boolean
}

type PendingAction = 'cancelTask' | 'startListening' | 'stopListening' | 'stopPlayback' | null

export function VoiceControls({
  audioLevel,
  cancelTask,
  disabled = false,
  error = null,
  listening,
  loading = false,
  speaking,
  startListening,
  stopListening,
  stopPlayback,
  taskRunning
}: VoiceControlsProps) {
  const { t } = useI18n()
  const copy = t.jarvisShell.dashboard.voiceControls
  const [pendingAction, setPendingAction] = useState<PendingAction>(null)
  const busy = loading || pendingAction !== null
  const safeLevel = Number.isFinite(audioLevel) ? Math.max(0, Math.min(audioLevel, 1)) : 0

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

  return (
    <section
      aria-label={copy.label}
      className="flex min-w-0 flex-wrap items-center gap-2 rounded-[4px] bg-(--ui-bg-secondary)/60 px-2 py-2"
      data-testid="jarvis-voice-controls"
    >
      <div
        aria-label={copy.micLevel}
        aria-valuemax={100}
        aria-valuemin={0}
        aria-valuenow={Math.round(safeLevel * 100)}
        className="flex min-h-11 items-center gap-1 px-1 text-(--ui-text-secondary)"
        role="meter"
      >
        {[0.4, 0.65, 1, 0.65, 0.4].map((weight, index) => (
          <span
            aria-hidden="true"
            className="w-1 rounded-full bg-current transition-[height] duration-75"
            key={index}
            style={{ height: `${(0.25 + (listening ? safeLevel * weight * 0.75 : 0)) * 1.5}rem` }}
          />
        ))}
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
      {error ? (
        <p className="basis-full text-xs text-destructive" role="alert">
          {error}
        </p>
      ) : null}
    </section>
  )
}
