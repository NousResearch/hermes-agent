import { AlertCircle, CheckCircle2, Loader2, Mic, Power, Volume2, Wrench } from '@/lib/icons'
import { cn } from '@/lib/utils'

import type { JarvisTaskPhase, JarvisUiState, JarvisVoiceState } from './types'

interface StatusStripProps {
  className?: string
  connected: boolean
  copy: {
    connection: {
      connected: string
      disconnected: string
    }
    task: Record<JarvisTaskPhase, string>
    toolIdle: string
    voice: Record<JarvisVoiceState, string>
  }
  state: JarvisUiState
}

function statusTone(taskPhase: JarvisTaskPhase): string {
  if (taskPhase === 'failed') {
    return 'text-(--ui-text-primary)'
  }

  if (taskPhase === 'verified') {
    return 'text-(--ui-accent)'
  }

  return 'text-(--ui-text-secondary)'
}

export function JarvisStatusStrip({ className, connected, copy, state }: StatusStripProps) {
  const taskLabel = copy.task[state.task.phase]
  const connectionLabel = connected ? copy.connection.connected : copy.connection.disconnected
  const toolLabel = state.activeTool?.label ?? copy.toolIdle
  const TaskIcon = state.task.phase === 'verified' ? CheckCircle2 : state.task.phase === 'failed' ? AlertCircle : Loader2
  const VoiceIcon = state.voice === 'speaking' ? Volume2 : Mic

  return (
    <section aria-label={copy.task[state.task.phase]} className={cn('flex flex-wrap gap-2', className)}>
      <span className="inline-flex min-h-11 items-center gap-2 rounded-md bg-(--ui-bg-quaternary) px-3 text-sm text-(--ui-text-secondary)">
        <Power className="size-4 text-(--ui-text-tertiary)" />
        {connectionLabel}
      </span>
      <span
        className={cn(
          'inline-flex min-h-11 items-center gap-2 rounded-md bg-(--ui-bg-quaternary) px-3 text-sm',
          statusTone(state.task.phase)
        )}
      >
        <TaskIcon className="size-4" />
        {taskLabel}
      </span>
      <span className="inline-flex min-h-11 items-center gap-2 rounded-md bg-(--ui-bg-quaternary) px-3 text-sm text-(--ui-text-secondary)">
        <Wrench className="size-4 text-(--ui-text-tertiary)" />
        {toolLabel}
      </span>
      <span className="inline-flex min-h-11 items-center gap-2 rounded-md bg-(--ui-bg-quaternary) px-3 text-sm text-(--ui-text-secondary)">
        <VoiceIcon className="size-4 text-(--ui-text-tertiary)" />
        {copy.voice[state.voice]}
      </span>
    </section>
  )
}
