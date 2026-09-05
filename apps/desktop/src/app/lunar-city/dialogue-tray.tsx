import { useState } from 'react'

import { Button } from '@/components/ui/button'

import type { WorldActionIntent, WorldActionResult } from './world-actions'
import type { WorldCondition, WorldEvent } from './world-events'

export interface DialogueSubject {
  condition?: WorldCondition
  conditions?: readonly WorldCondition[]
  detail?: string
  event?: WorldEvent
  events?: readonly WorldEvent[]
  title: string
}

interface DialogueTrayProps {
  onAction?: (intent: WorldActionIntent) => Promise<WorldActionResult>
  onClose?: () => void
  subject: DialogueSubject
}

function actionFor(subject: DialogueSubject, kind: WorldActionIntent['kind']): WorldActionIntent | null {
  const taskId = subject.event?.sourceRef?.taskId ?? subject.condition?.sourceRef?.taskId

  if (kind === 'inspect' || kind === 'inspect_blocker' || kind === 'show_source') {
    return {
      kind,
      target: {
        board: subject.event?.sourceRef?.board ?? subject.condition?.sourceRef?.board,
        eventId: subject.event?.id,
        prId: subject.event?.sourceRef?.prId,
        taskId
      }
    }
  }

  if (kind === 'recover_task' && taskId) {
    return { kind, patch: { status: 'todo' }, taskId }
  }

  return null
}

export function DialogueTray({ onAction, onClose, subject }: DialogueTrayProps) {
  const event = subject.event
  const condition = subject.condition

  const actions = new Set<WorldActionIntent['kind']>([
    ...(event?.actionKinds ?? []),
    ...(condition?.kind === 'task.blocked' || condition?.kind === 'worker.stale' ? ['inspect_blocker' as const] : [])
  ])

  const [actionMessage, setActionMessage] = useState<string | null>(null)

  const runAction = async (kind: WorldActionIntent['kind']) => {
    const intent = actionFor(subject, kind)

    if (!intent || !onAction) {
      setActionMessage('This action is not available from the current Hermes state.')

      return
    }

    const result = await onAction(intent)
    setActionMessage(result.ok ? 'Hermes accepted the action.' : result.message)
  }

  return (
    <aside
      aria-label={`Dialogue: ${subject.title}`}
      className="rounded-2xl border border-violet-300/30 bg-slate-950/95 p-4 shadow-xl"
    >
      <div className="flex items-start justify-between gap-3">
        <div>
          <p className="text-[0.65rem] font-semibold uppercase tracking-[0.18em] text-violet-300">In-world dialogue</p>
          <h2 className="mt-1 text-lg font-semibold text-white">{subject.title}</h2>
        </div>
        {onClose && (
          <Button aria-label="Close dialogue" onClick={onClose} size="icon-xs" variant="ghost">
            ×
          </Button>
        )}
      </div>

      {(event || condition) && (
        <dl className="mt-4 grid grid-cols-[auto_1fr] gap-x-3 gap-y-1 text-xs">
          <dt className="text-slate-400">Source</dt>
          <dd className="font-mono text-slate-200">{event?.source ?? condition?.source}</dd>
          <dt className="text-slate-400">State</dt>
          <dd className="text-slate-200">{event?.kind ?? condition?.kind}</dd>
          <dt className="text-slate-400">Severity</dt>
          <dd className="text-slate-200">{event?.severity ?? condition?.severity}</dd>
          {(event?.sourceRef?.taskId || condition?.sourceRef?.taskId) && (
            <>
              <dt className="text-slate-400">Task</dt>
              <dd className="font-mono text-slate-200">{event?.sourceRef?.taskId ?? condition?.sourceRef?.taskId}</dd>
            </>
          )}
        </dl>
      )}

      <p className="mt-4 whitespace-pre-wrap text-sm leading-6 text-slate-200">
        {event?.detail ?? condition?.detail ?? subject.detail ?? 'No additional source detail is available.'}
      </p>

      {(typeof event?.facts.role === 'string' || typeof condition?.facts.assignee === 'string') && (
        <p className="mt-2 text-xs text-slate-400">
          {event?.facts.role
            ? `Role: ${String(event.facts.role)}`
            : `Assigned to: ${String(condition?.facts.assignee)}`}
        </p>
      )}

      {subject.events && subject.events.length > 0 && (
        <div className="mt-4 space-y-2 border-t border-white/10 pt-3">
          {subject.events.slice(-8).map(item => (
            <button
              className="block w-full rounded-md border border-white/10 bg-white/5 px-2 py-1.5 text-left text-xs text-slate-200 hover:bg-white/10"
              key={item.id}
              onClick={() => void runAction('inspect')}
              type="button"
            >
              <span className="font-mono text-violet-200">{item.kind}</span> — {item.title}
            </button>
          ))}
        </div>
      )}

      {actions.size > 0 && (
        <div className="mt-4 flex flex-wrap gap-2 border-t border-white/10 pt-3">
          {actions.has('inspect_blocker') && (
            <Button onClick={() => void runAction('inspect_blocker')} size="sm" variant="secondary">
              Inspect blocker
            </Button>
          )}
          {actions.has('inspect') && (
            <Button onClick={() => void runAction('inspect')} size="sm" variant="ghost">
              Inspect
            </Button>
          )}
          {(condition?.kind === 'task.blocked' || event?.kind === 'task.blocked') && (
            <Button onClick={() => void runAction('recover_task')} size="sm">
              Extinguish / recover
            </Button>
          )}
        </div>
      )}

      {actionMessage && (
        <p className="mt-3 text-xs text-slate-300" role="status">
          {actionMessage}
        </p>
      )}
    </aside>
  )
}
