import { Button, Input, Textarea, useMutation, useQuery, useQueryClient, useValue } from '@hermes/plugin-sdk'
import { useState } from 'react'

import { $boardSlug, archiveSessionMirror, deleteSessionMirror, fetchSessionMirrors, promoteSessionMirror, useKanbanScope } from './api'
import { useKanban } from './ui'

export function KanbanSessionsView() {
  const k = useKanban()
  const scope = useKanbanScope()
  const slug = useValue($boardSlug)
  const qc = useQueryClient()
  const key = ['kanban', 'sessions', scope, slug] as const
  const { data, isLoading } = useQuery({ queryKey: key, queryFn: fetchSessionMirrors, refetchInterval: 60_000 })
  const [draft, setDraft] = useState<Record<number, { title: string; body: string }>>({})
  const refresh = () => void qc.invalidateQueries({ queryKey: key })
  const archive = useMutation({ mutationFn: archiveSessionMirror, onSuccess: refresh })
  const remove = useMutation({ mutationFn: deleteSessionMirror, onSuccess: refresh })
  const promote = useMutation({ mutationFn: ({ id, title, body }: { id: number; title: string; body: string }) => promoteSessionMirror(id, title, body), onSuccess: refresh })

  return <section aria-label={k.sessions} className="flex-1 overflow-y-auto p-4">
    <h2 className="mb-2 text-sm font-semibold">{k.sessions}</h2>
    <p className="mb-4 inline-flex rounded-full bg-(--ui-bg-quaternary) px-2 py-1 text-xs text-(--ui-text-tertiary)">{k.sessionsReadOnly}</p>
    {isLoading ? <p>…</p> : !data?.mirrors.length ? <p className="text-xs text-(--ui-text-tertiary)">{k.noSessions}</p> :
      <ul className="space-y-3">{data.mirrors.map(m => {
        const value = draft[m.id] ?? { title: '', body: '' }
        const update = (patch: Partial<typeof value>) => setDraft(prev => ({ ...prev, [m.id]: { ...value, ...patch } }))

        return <li className="rounded-md border border-(--ui-stroke-tertiary) p-3" key={m.id}>
          <div className="flex flex-wrap items-center gap-2"><strong>{m.title}</strong><span className="rounded bg-(--ui-bg-quaternary) px-1.5 text-xs">Session · {m.status}</span><span className="text-xs text-(--ui-text-tertiary)">{m.platform} · {m.profile} · {m.chat_id}</span></div>
          <div className="mt-2 text-xs text-(--ui-text-tertiary)">Session {m.session_id}{m.thread_id ? ` · Thread ${m.thread_id}` : ''}{m.promoted_task_id ? ` · Task ${m.promoted_task_id}` : ''}</div>
          <div className="mt-1 flex flex-wrap gap-x-4 gap-y-1 text-xs text-(--ui-text-tertiary)">
            <time>{k.sessionReceived}: {new Date(m.received_at * 1000).toLocaleString()}</time>
            {m.started_at != null && <time>{k.sessionStarted}: {new Date(m.started_at * 1000).toLocaleString()}</time>}
            {m.completed_at != null && <time>{k.sessionCompleted}: {new Date(m.completed_at * 1000).toLocaleString()}</time>}
          </div>
          {!m.promoted_task_id && <div className="mt-3 grid gap-2"><Input aria-label={k.sessionTitle} onChange={e => update({ title: e.target.value })} placeholder={k.sessionTitle} value={value.title}/><Textarea aria-label={k.sessionBody} onChange={e => update({ body: e.target.value })} placeholder={k.sessionBody} value={value.body}/><Button disabled={!value.title.trim() || promote.isPending} onClick={() => promote.mutate({ id: m.id, ...value })}>{k.promoteSession}</Button></div>}
          <div className="mt-2 flex gap-2"><Button onClick={() => archive.mutate(m.id)} size="sm" variant="outline">{k.archiveSession}</Button><Button onClick={() => remove.mutate(m.id)} size="sm" variant="destructive">{k.deleteSession}</Button></div>
        </li>
      })}</ul>}
  </section>
}
