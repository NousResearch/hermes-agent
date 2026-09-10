import { type FormEvent, useState } from 'react'

import { Button } from '@/components/ui/button'

import type { DialogueSubject } from './dialogue-tray'
import type { WorldActionResult, WorldActionRunner } from './world-actions'
import type { WorldCondition, WorldEvent } from './world-events'

export interface DispatcherCubeContext {
  actionRunner: WorldActionRunner
  conditions: readonly WorldCondition[]
  events: readonly WorldEvent[]
}

interface DispatcherCubeProps {
  context: DispatcherCubeContext
  onSubjectSelected?: (subject: DialogueSubject) => void
}

function resultText(result: WorldActionResult): string {
  return result.ok ? 'Hermes accepted the action.' : result.message
}

export function DispatcherCube({ context, onSubjectSelected }: DispatcherCubeProps) {
  const [mode, setMode] = useState<'report' | 'task' | 'session' | null>(null)
  const [title, setTitle] = useState('')
  const [body, setBody] = useState('')
  const [feedback, setFeedback] = useState<string | null>(null)
  const [pending, setPending] = useState(false)

  const run = async (action: Parameters<WorldActionRunner['run']>[0]) => {
    setPending(true)
    const result = await context.actionRunner.run(action)
    setFeedback(resultText(result))
    setPending(false)
  }

  const submitTask = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault()

    if (!title.trim()) {
      setFeedback('A task title is required.')

      return
    }

    await run({
      body: { ...(body.trim() ? { body: body.trim() } : {}), title: title.trim() },
      kind: 'create_task'
    })
    setTitle('')
    setBody('')
  }

  const showSubject = (subject: DialogueSubject) => {
    onSubjectSelected?.(subject)
  }

  return (
    <section
      aria-label="Dispatcher companion"
      className="rounded-2xl border border-cyan-400/30 bg-slate-950/90 p-4 shadow-xl"
    >
      <div className="flex items-start justify-between gap-3">
        <div>
          <p className="text-[0.65rem] font-semibold uppercase tracking-[0.18em] text-cyan-300">Dispatcher cube</p>
          <h2 className="mt-1 text-lg font-semibold text-white">Command center</h2>
          <p className="mt-1 text-xs text-slate-300">Talk to Hermes through the city.</p>
        </div>
        <div
          aria-hidden
          className="grid size-10 place-items-center rounded-lg border border-cyan-300/50 bg-cyan-300/10 text-xl text-cyan-200"
        >
          ◈
        </div>
      </div>

      <div className="mt-4 grid grid-cols-2 gap-2">
        <Button onClick={() => setMode(mode === 'task' ? null : 'task')} size="sm" variant="secondary">
          New task
        </Button>
        <Button onClick={() => setMode(mode === 'session' ? null : 'session')} size="sm" variant="secondary">
          New session
        </Button>
        <Button onClick={() => setMode(mode === 'report' ? null : 'report')} size="sm" variant="secondary">
          What needs my attention?
        </Button>
        <Button
          onClick={() =>
            showSubject({
              title: 'Recent Hermes events',
              events: context.events.slice(-8),
              conditions: context.conditions
            })
          }
          size="sm"
          variant="ghost"
        >
          Recent activity
        </Button>
      </div>

      {mode === 'task' && (
        <form
          aria-label="Create Kanban task"
          className="mt-4 space-y-2 rounded-xl border border-white/10 bg-white/5 p-3"
          onSubmit={submitTask}
        >
          <label className="block text-xs text-slate-300" htmlFor="dispatcher-task-title">
            Title
          </label>
          <input
            className="w-full rounded-md border border-white/15 bg-black/20 px-2 py-1.5 text-sm text-white outline-none focus:border-cyan-300"
            id="dispatcher-task-title"
            onChange={event => setTitle(event.target.value)}
            value={title}
          />
          <label className="block text-xs text-slate-300" htmlFor="dispatcher-task-body">
            Brief
          </label>
          <textarea
            className="min-h-16 w-full rounded-md border border-white/15 bg-black/20 px-2 py-1.5 text-sm text-white outline-none focus:border-cyan-300"
            id="dispatcher-task-body"
            onChange={event => setBody(event.target.value)}
            value={body}
          />
          <Button disabled={pending} size="sm" type="submit">
            {pending ? 'Sending to Hermes…' : 'Dispatch task'}
          </Button>
        </form>
      )}

      {mode === 'session' && (
        <div className="mt-4 rounded-xl border border-white/10 bg-white/5 p-3 text-sm text-slate-200">
          <p>Start a fresh Hermes conversation in the current desktop context?</p>
          <Button
            className="mt-3"
            disabled={pending}
            onClick={() => void run({ kind: 'create_session', params: { source: 'lunar-city' } })}
            size="sm"
          >
            {pending ? 'Starting…' : 'Start session'}
          </Button>
        </div>
      )}

      {mode === 'report' && (
        <div className="mt-4 space-y-2 rounded-xl border border-white/10 bg-white/5 p-3 text-sm text-slate-200">
          <p>{context.conditions.length} active world conditions</p>
          <p>{context.events.length} recent source events</p>
          <Button
            onClick={() =>
              showSubject({
                title: 'Situation report',
                events: context.events,
                conditions: context.conditions
              })
            }
            size="sm"
            variant="ghost"
          >
            Open report
          </Button>
        </div>
      )}

      {feedback && (
        <p
          className="mt-3 rounded-md border border-white/10 bg-white/5 px-2 py-1.5 text-xs text-slate-200"
          role="status"
        >
          {feedback}
        </p>
      )}
    </section>
  )
}

export type { DialogueSubject }
