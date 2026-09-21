import { useCallback, useEffect, useState } from 'react'

import { useGatewayRequest } from '@/app/gateway/hooks/use-gateway-request'
import { AlertCircle, Archive, CheckCircle2, MessageCircle, Play, Plus, RefreshCw, Users } from '@/lib/icons'
import { cn } from '@/lib/utils'

// ── Control Room snapshot wire types (mirror of the shared Python contract) ──

export interface CrAttentionItem {
  kind: string
  id: string
  severity: number
  title: string
  detail?: string
  profile?: string
}

export interface CrCounts {
  needs_you: number
  agents_active: number
  tasks_running: number
  messages_unread: number
  system_severity: number
}

export interface CrCapabilities {
  approvals: boolean
  peer_messages: boolean
  kanban_actions: boolean
  process_control: boolean
  delegation_control: boolean
}

export interface CrSnapshot {
  version: number
  profile: string
  generated_at: string
  attention: CrAttentionItem[]
  counts: CrCounts
  agents: { id: string; kind: string; name: string; status: string; detail?: string; available_actions?: string[] }[]
  tasks: { id: string; title: string; state: string; owner?: string; available_actions?: string[] }[]
  messages: { id: string; kind: string; title: string; state: string; sender?: string; available_actions?: string[] }[]
  system: { state: string; severity: number; detail?: string }
  capabilities: CrCapabilities
}

const SEVERITY_GLYPH = ['!!', '!', '~', '·'] as const

/**
 * Control Room home for the desktop command center (CR-402, CR-406).
 * Reads the shared gateway snapshot contract — no private-store query. Renders
 * the attention-first home: Needs You / Agents / Tasks / Messages / System.
 * Unavailable providers surface as typed unavailable rows, never fake zeros.
 */
export function ControlRoomHome() {
  const { requestGateway } = useGatewayRequest()
  const [snapshot, setSnapshot] = useState<CrSnapshot | null>(null)
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(true)
  const [section, setSection] = useState<'home' | 'needs-you' | 'agents' | 'tasks' | 'messages' | 'system'>('home')

  const refresh = useCallback(() => {
    setLoading(true)
    requestGateway<CrSnapshot>('control.room.snapshot', { profile: 'default' })
      .then(snap => {
        if (snap) {
          setSnapshot(snap)
          setError('')
        } else {
          setError('snapshot unavailable')
        }
      })
      .catch((e: unknown) => setError(e instanceof Error ? e.message : String(e)))
      .finally(() => setLoading(false))
  }, [requestGateway])

  useEffect(() => {
    refresh()
  }, [refresh])

  if (error && !snapshot) {
    return (
      <div className="flex h-full min-h-48 flex-col items-center justify-center gap-2 px-6 text-center">
        <AlertCircle className="size-6 text-(--ui-red)" />
        <div className="text-sm text-(--ui-text-tertiary)">Control Room unavailable: {error}</div>
        <button className="text-xs text-(--ui-accent) hover:underline" onClick={refresh} type="button">
          retry
        </button>
      </div>
    )
  }

  if (loading && !snapshot) {
    return (
      <div className="flex h-full min-h-48 items-center justify-center px-6">
        <div className="text-sm text-(--ui-text-tertiary)">loading Control Room…</div>
      </div>
    )
  }

  if (!snapshot) {
    return (
      <div className="flex h-full min-h-48 items-center justify-center px-6">
        <div className="text-sm text-(--ui-text-tertiary)">Control Room unavailable</div>
      </div>
    )
  }

  const back = () => setSection('home')

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <div className="mb-3 flex items-center justify-between gap-2">
        <div className="text-xs font-medium uppercase tracking-wider text-(--ui-text-tertiary)">
          {section === 'home' ? `Control Room · ${snapshot.profile}` : `Control Room › ${section}`}
        </div>
        <button
          aria-label="refresh"
          className="rounded-md p-1 text-(--ui-text-tertiary) hover:bg-(--chrome-action-hover) hover:text-foreground"
          onClick={refresh}
          type="button"
        >
          <RefreshCw className={cn('size-3.5', loading && 'animate-spin')} />
        </button>
      </div>

      {section === 'home' ? (
        <HomeView onOpen={setSection} snapshot={snapshot} />
      ) : (
        <SectionView onBack={back} section={section} snapshot={snapshot} />
      )}
    </div>
  )
}

// ── Home ──────────────────────────────────────────────────────────────

function HomeView({
  snapshot,
  onOpen
}: {
  snapshot: CrSnapshot
  onOpen: (s: 'needs-you' | 'agents' | 'tasks' | 'messages' | 'system') => void
}) {
  const c = snapshot.counts

  const rows = [
    { key: 'needs-you' as const, icon: AlertCircle, label: 'Needs You', value: `${c.needs_you}`, detail: 'approvals · blocked · stalled' },
    { key: 'agents' as const, icon: Users, label: 'Agents', value: `${c.agents_active} active`, detail: 'foreground · delegations · processes' },
    { key: 'tasks' as const, icon: Archive, label: 'Tasks', value: `${c.tasks_running} running`, detail: 'kanban by attention/running/ready' },
    { key: 'messages' as const, icon: MessageCircle, label: 'Messages', value: `${c.messages_unread} unread`, detail: 'peer inbox · requests' },
    { key: 'system' as const, icon: CheckCircle2, label: 'System', value: snapshot.system?.state ?? 'unknown', detail: snapshot.system?.detail ?? '' }
  ]

  return (
    <div className="min-h-0 flex-1 overflow-y-auto">
      <div className="grid gap-2">
        {rows.map(row => {
          const Icon = row.icon

          return (
            <button
              className="group flex items-center gap-3 rounded-lg border border-(--ui-stroke-secondary) px-3 py-2.5 text-left hover:bg-(--chrome-action-hover)"
              key={row.key}
              onClick={() => onOpen(row.key)}
              type="button"
            >
              <Icon className="size-4 shrink-0 text-(--ui-text-tertiary)" />
              <span className="min-w-0 flex-1">
                <span className="block truncate text-sm font-medium text-foreground">{row.label}</span>
                <span className="block truncate text-xs text-(--ui-text-tertiary)">{row.detail}</span>
              </span>
              <span className="shrink-0 text-sm font-semibold text-(--ui-accent)">{row.value}</span>
            </button>
          )
        })}
      </div>

      <div className="mt-3 flex flex-wrap gap-2">
        <NewActionButton icon={Plus} label="+ New Task" onClick={() => {}} />
        <NewActionButton icon={MessageCircle} label="+ New Message" onClick={() => {}} />
        <NewActionButton icon={Play} label="+ New Agent Run" onClick={() => {}} />
      </div>
    </div>
  )
}

function NewActionButton({
  icon: Icon,
  label,
  onClick
}: {
  icon: typeof Plus
  label: string
  onClick: () => void
}) {
  return (
    <button
      className="inline-flex items-center gap-1.5 rounded-md border border-(--ui-stroke-secondary) px-2.5 py-1.5 text-xs font-medium text-foreground hover:bg-(--chrome-action-hover)"
      onClick={onClick}
      type="button"
    >
      <Icon className="size-3.5" />
      {label}
    </button>
  )
}

// ── Sections ──────────────────────────────────────────────────────────

function SectionView({
  snapshot,
  section,
  onBack
}: {
  snapshot: CrSnapshot
  section: 'needs-you' | 'agents' | 'tasks' | 'messages' | 'system'
  onBack: () => void
}) {
  return (
    <div className="min-h-0 flex-1 overflow-y-auto">
      {section === 'needs-you' && (
        <>
          {snapshot.attention.filter(a => a.severity <= 1).length === 0 ? (
            <EmptyNote text="Nothing needs you right now." />
          ) : (
            snapshot.attention
              .filter(a => a.severity <= 1)
              .map(item => (
                <Row
                  glyph={SEVERITY_GLYPH[item.severity] ?? '·'}
                  key={`${item.kind}:${item.id}`}
                  sub={item.detail}
                  text={item.title}
                />
              ))
          )}
          <UnavailableHint text="Approvals not wired in this runtime — shown unavailable." visible={!snapshot.capabilities.approvals} />
        </>
      )}

      {section === 'agents' && (
        <>
          {snapshot.agents.length === 0 ? (
            <EmptyNote text="No active agents." />
          ) : (
            snapshot.agents.map(a => (
              <Row key={`${a.kind}:${a.id}`} sub={a.detail} text={`[${a.kind}] ${a.name} · ${a.status}`} />
            ))
          )}
          {!snapshot.capabilities.process_control && snapshot.agents.length > 0 && (
            <UnavailableHint text="Process/delegation control not wired in this runtime." visible />
          )}
        </>
      )}

      {section === 'tasks' && (
        <>
          {snapshot.tasks.length === 0 ? (
            <EmptyNote text="No tasks." />
          ) : (
            snapshot.tasks.map(task => (
              <Row key={task.id} sub={`${task.state}${task.owner ? ` · ${task.owner}` : ''}`} text={`${task.id} · ${task.title}`} />
            ))
          )}
          {!snapshot.capabilities.kanban_actions && snapshot.tasks.length > 0 && (
            <UnavailableHint text="Kanban write route not wired in this runtime." visible />
          )}
        </>
      )}

      {section === 'messages' && (
        <>
          {snapshot.messages.length === 0 ? (
            <EmptyNote text="No held or queued peer messages." />
          ) : (
            snapshot.messages.map(m => (
              <Row key={`${m.kind}:${m.id}`} sub={m.sender ? `from ${m.sender}` : ''} text={`[${m.state}] ${m.title}`} />
            ))
          )}
          {!snapshot.capabilities.peer_messages && snapshot.messages.length > 0 && (
            <UnavailableHint text="Hermes Peer plugin not wired in this runtime." visible />
          )}
        </>
      )}

      {section === 'system' && (
        <>
          <Row sub={snapshot.system?.detail} text={`state: ${snapshot.system?.state ?? 'unknown'}`} />
        </>
      )}

      <button className="mt-3 text-xs text-(--ui-accent) hover:underline" onClick={onBack} type="button">
        ← back
      </button>
    </div>
  )
}

function Row({ glyph, sub, text }: { glyph?: string; sub?: string; text: string }) {
  return (
    <div className="flex items-start gap-2 rounded-md px-2 py-1.5 hover:bg-(--chrome-action-hover)">
      {glyph && <span className="w-5 shrink-0 text-center text-sm text-(--ui-text-tertiary)">{glyph}</span>}
      <span className="min-w-0 flex-1">
        <span className="block truncate text-sm text-foreground">{text}</span>
        {sub && <span className="block truncate text-xs text-(--ui-text-tertiary)">{sub}</span>}
      </span>
    </div>
  )
}

function EmptyNote({ text }: { text: string }) {
  return <div className="px-2 py-3 text-sm text-(--ui-text-tertiary)">{text}</div>
}

function UnavailableHint({ text, visible }: { text: string; visible: boolean }) {
  if (!visible) {return null}

  return <div className="mt-2 px-2 text-xs text-(--ui-text-tertiary)">{text}</div>
}
