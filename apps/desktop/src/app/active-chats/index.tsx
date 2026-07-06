import { useStore } from '@nanostores/react'
import type * as React from 'react'
import { useEffect, useMemo, useState } from 'react'

import { PageLoader } from '@/components/page-loader'
import { Button } from '@/components/ui/button'
import { Textarea } from '@/components/ui/textarea'
import { getSessionMessages, listAllProfileSessions } from '@/hermes'
import { useI18n } from '@/i18n'
import { type ChatMessage, chatMessageText, toChatMessages } from '@/lib/chat-messages'
import { sessionTitle } from '@/lib/chat-runtime'
import { Clock, FolderOpen, Loader2, MessageSquareText, Send, Users, Zap } from '@/lib/icons'
import { cn } from '@/lib/utils'
import { notify, notifyError } from '@/store/notifications'
import {
  $attentionSessionIds,
  $messagingSessions,
  $sessions,
  $sessionsLoading,
  $workingSessionIds
} from '@/store/session'
import type { SessionInfo } from '@/types/hermes'

import { PageSearchShell } from '../page-search-shell'

interface ActiveChatsViewProps extends React.ComponentProps<'section'> {
  onOpenSession: (sessionId: string) => void
  onRefreshSessions: () => Promise<void>
  onSendReply: (session: SessionInfo, text: string) => Promise<boolean>
}

type ChatState = 'idle' | 'recent' | 'reply' | 'running' | 'waiting'
type GroupMode = 'profile' | 'status' | 'workspace'

interface ActiveGroup {
  id: string
  label: string
  mode: GroupMode
  sessions: SessionInfo[]
}

const PREVIEW_LIMIT = 10
const REPLY_SCAN_LIMIT = 100

function sessionKey(session: SessionInfo): string {
  return `${session.profile || 'default'}:${session.id}`
}

function sessionTimestamp(session: SessionInfo): number {
  return (session.last_active || session.started_at || 0) * 1000
}

function chatState(
  session: SessionInfo,
  workingIds: string[],
  attentionIds: string[],
  replyNeededKeys: ReadonlySet<string>
): ChatState {
  if (attentionIds.includes(session.id)) {
    return 'waiting'
  }

  if (workingIds.includes(session.id)) {
    return 'running'
  }

  return replyNeededKeys.has(sessionKey(session)) ? 'reply' : 'idle'
}

function stateRank(state: ChatState): number {
  if (state === 'waiting') {
    return 0
  }

  if (state === 'running') {
    return 1
  }

  if (state === 'reply') {
    return 2
  }

  if (state === 'recent') {
    return 3
  }

  return 4
}

function activeState(state: ChatState): state is 'reply' | 'running' | 'waiting' {
  return state === 'reply' || state === 'running' || state === 'waiting'
}

function transcriptNeedsReply(messages: ChatMessage[]): boolean {
  const lastVisible = messages.filter(message => !message.hidden).at(-1)

  return lastVisible?.role === 'user'
}

function workspaceLabel(session: SessionInfo): string {
  if (!session.cwd) {
    return 'No workspace'
  }

  const parts = session.cwd.split('/').filter(Boolean)

  return parts.at(-1) || session.cwd
}

function formatAge(timestampMs: number): string {
  const delta = Math.max(0, Date.now() - timestampMs)
  const minute = 60_000
  const hour = 60 * minute

  if (delta < minute) {
    return 'now'
  }

  if (delta < hour) {
    return `${Math.floor(delta / minute)}m`
  }

  return `${Math.floor(delta / hour)}h`
}

export function ActiveChatsView({
  className,
  onOpenSession,
  onRefreshSessions,
  onSendReply,
  ...props
}: ActiveChatsViewProps) {
  const { t } = useI18n()
  const copy = t.activeChats
  const sessions = useStore($sessions)
  const messagingSessions = useStore($messagingSessions)
  const workingIds = useStore($workingSessionIds)
  const attentionIds = useStore($attentionSessionIds)
  const sessionsLoading = useStore($sessionsLoading)
  const [scannedSessions, setScannedSessions] = useState<SessionInfo[]>([])
  const [replyNeededKeys, setReplyNeededKeys] = useState<Set<string>>(() => new Set())
  const [query, setQuery] = useState('')
  const [groupMode, setGroupMode] = useState<GroupMode>('status')
  const [selectedKey, setSelectedKey] = useState('')
  const [messages, setMessages] = useState<ChatMessage[] | null>(null)
  const [messagesLoading, setMessagesLoading] = useState(false)
  const [messagesError, setMessagesError] = useState('')
  const [draftBySession, setDraftBySession] = useState<Record<string, string>>({})
  const [sendingKey, setSendingKey] = useState('')

  const candidateRows = useMemo(() => {
    const byKey = new Map<string, SessionInfo>()

    for (const session of [...scannedSessions, ...sessions, ...messagingSessions]) {
      byKey.set(sessionKey(session), session)
    }

    return [...byKey.values()].sort((a, b) => sessionTimestamp(b) - sessionTimestamp(a))
  }, [messagingSessions, scannedSessions, sessions])

  const activeRows = useMemo(() => {
    const byKey = new Map<string, SessionInfo>()

    for (const session of candidateRows) {
      if (activeState(chatState(session, workingIds, attentionIds, replyNeededKeys))) {
        byKey.set(sessionKey(session), session)
      }
    }

    return [...byKey.values()].sort((a, b) => {
      const stateDelta =
        stateRank(chatState(a, workingIds, attentionIds, replyNeededKeys)) -
        stateRank(chatState(b, workingIds, attentionIds, replyNeededKeys))

      return stateDelta || sessionTimestamp(b) - sessionTimestamp(a)
    })
  }, [attentionIds, candidateRows, replyNeededKeys, workingIds])

  const visibleRows = useMemo(() => {
    const needle = query.trim().toLowerCase()

    if (!needle) {
      return activeRows
    }

    return activeRows.filter(session =>
      [sessionTitle(session), session.preview, session.cwd, session.profile, session.source, session.id]
        .filter(Boolean)
        .some(value => String(value).toLowerCase().includes(needle))
    )
  }, [activeRows, query])

  const groupedRows = useMemo(() => {
    const groups = new Map<string, ActiveGroup>()

    for (const session of visibleRows) {
      const state = chatState(session, workingIds, attentionIds, replyNeededKeys)

      const id =
        groupMode === 'status'
          ? state
          : groupMode === 'profile'
            ? session.profile || 'default'
            : workspaceLabel(session)

      const label =
        groupMode === 'status'
          ? copy[state]
          : groupMode === 'profile'
            ? session.profile || 'default'
            : workspaceLabel(session)

      const group = groups.get(id) ?? { id, label, mode: groupMode, sessions: [] }
      group.sessions.push(session)
      groups.set(id, group)
    }

    return [...groups.values()].sort((a, b) => {
      if (groupMode === 'status') {
        return stateRank(a.id as ChatState) - stateRank(b.id as ChatState)
      }

      return a.label.localeCompare(b.label)
    })
  }, [attentionIds, copy, groupMode, replyNeededKeys, visibleRows, workingIds])

  const selected = useMemo(() => {
    if (!visibleRows.length) {
      return null
    }

    return visibleRows.find(session => sessionKey(session) === selectedKey) ?? visibleRows[0]
  }, [selectedKey, visibleRows])

  const selectedSessionKey = selected ? sessionKey(selected) : ''
  const visibleMessages = useMemo(() => (messages ?? []).filter(message => !message.hidden).slice(-PREVIEW_LIMIT), [messages])
  const draft = selectedSessionKey ? (draftBySession[selectedSessionKey] ?? '') : ''

  const waitingCount = activeRows.filter(
    session => chatState(session, workingIds, attentionIds, replyNeededKeys) === 'waiting'
  ).length

  const runningCount = activeRows.filter(
    session => chatState(session, workingIds, attentionIds, replyNeededKeys) === 'running'
  ).length

  useEffect(() => {
    let cancelled = false

    listAllProfileSessions(REPLY_SCAN_LIMIT, 1, 'exclude', 'recent', 'all', { excludeSources: ['cron'] })
      .then(result => {
        if (!cancelled) {
          setScannedSessions(result.sessions)
        }
      })
      .catch(() => undefined)

    return () => {
      cancelled = true
    }
  }, [])

  useEffect(() => {
    let cancelled = false
    const candidates = candidateRows.slice(0, REPLY_SCAN_LIMIT)

    void Promise.all(
      candidates.map(async session => {
        if (attentionIds.includes(session.id) || workingIds.includes(session.id)) {
          return [sessionKey(session), false] as const
        }

        try {
          const result = await getSessionMessages(session.id, session.profile)

          return [sessionKey(session), transcriptNeedsReply(toChatMessages(result.messages))] as const
        } catch {
          return [sessionKey(session), false] as const
        }
      })
    ).then(results => {
      if (cancelled) {
        return
      }

      setReplyNeededKeys(new Set(results.filter(([, needsReply]) => needsReply).map(([key]) => key)))
    })

    return () => {
      cancelled = true
    }
  }, [attentionIds, candidateRows, workingIds])

  useEffect(() => {
    if (selected && selectedSessionKey !== selectedKey) {
      setSelectedKey(selectedSessionKey)
    }
  }, [selected, selectedKey, selectedSessionKey])

  useEffect(() => {
    if (!selected) {
      setMessages(null)
      setMessagesError('')
      setMessagesLoading(false)

      return
    }

    let cancelled = false

    setMessagesLoading(true)
    setMessagesError('')
    getSessionMessages(selected.id, selected.profile)
      .then(result => {
        if (!cancelled) {
          setMessages(toChatMessages(result.messages))
        }
      })
      .catch(err => {
        if (!cancelled) {
          setMessages(null)
          setMessagesError(err instanceof Error ? err.message : String(err))
        }
      })
      .finally(() => {
        if (!cancelled) {
          setMessagesLoading(false)
        }
      })

    return () => {
      cancelled = true
    }
  }, [selected])

  async function submitReply() {
    if (!selected || !draft.trim() || sendingKey) {
      return
    }

    const key = sessionKey(selected)

    setSendingKey(key)

    try {
      const sent = await onSendReply(selected, draft)

      if (sent) {
        setDraftBySession(current => ({ ...current, [key]: '' }))
        notify({ kind: 'success', title: copy.sent, message: sessionTitle(selected) })
      } else {
        notify({ kind: 'error', title: copy.sendFailed, message: sessionTitle(selected) })
      }
    } catch (err) {
      notifyError(err, copy.sendFailed)
    } finally {
      setSendingKey('')
    }
  }

  return (
    <PageSearchShell
      {...props}
      className={className}
      filters={
        <div className="flex min-w-0 items-center gap-2 text-[length:var(--conversation-caption-font-size)] leading-(--conversation-caption-line-height) text-(--ui-text-tertiary)">
          <Zap className="size-3.5 shrink-0" />
          <span className="truncate">{copy.subtitle}</span>
        </div>
      }
      onSearchChange={setQuery}
      searchPlaceholder={copy.search}
      searchValue={query}
      tabs={[{ id: 'active', label: copy.title }]}
    >
      {sessionsLoading && activeRows.length === 0 ? (
        <PageLoader label={t.common.loading} />
      ) : visibleRows.length === 0 ? (
        <EmptyState description={copy.emptyDesc} title={copy.emptyTitle} />
      ) : (
        <div className="grid h-full min-h-0 grid-cols-1 lg:grid-cols-[21rem_minmax(0,1fr)]">
          <aside className="flex min-h-0 flex-col border-r border-(--ui-stroke-tertiary)">
            <div className="shrink-0 border-b border-(--ui-stroke-tertiary) px-3 py-3">
              <div className="grid grid-cols-2 gap-2">
                <QueueMetric icon={<Clock className="size-3.5" />} label={copy.waiting} value={waitingCount} />
                <QueueMetric icon={<Zap className="size-3.5" />} label={copy.running} value={runningCount} />
              </div>
              <div className="mt-3 grid grid-cols-3 rounded-md border border-(--ui-stroke-tertiary) bg-(--ui-surface-muted) p-0.5">
                <GroupButton active={groupMode === 'status'} icon={<Zap className="size-3.5" />} label={copy.groupStatus} onClick={() => setGroupMode('status')} />
                <GroupButton active={groupMode === 'profile'} icon={<Users className="size-3.5" />} label={copy.groupProfile} onClick={() => setGroupMode('profile')} />
                <GroupButton active={groupMode === 'workspace'} icon={<FolderOpen className="size-3.5" />} label={copy.groupWorkspace} onClick={() => setGroupMode('workspace')} />
              </div>
            </div>
            <div className="min-h-0 flex-1 overflow-y-auto p-2">
              <div className="space-y-3">
                {groupedRows.map(group => (
                  <section key={`${group.mode}:${group.id}`}>
                    <div className="mb-1.5 flex items-center justify-between gap-2 px-1">
                      <div className="flex min-w-0 items-center gap-1.5">
                        <GroupIcon mode={group.mode} />
                        <h3 className="truncate text-[0.6875rem] font-semibold uppercase tracking-[0.08em] text-(--ui-text-tertiary)">
                          {group.label}
                        </h3>
                      </div>
                      <span className="text-[0.6875rem] tabular-nums text-(--ui-text-quaternary)">{group.sessions.length}</span>
                    </div>
                    <ul className="space-y-1">
                      {group.sessions.map(session => {
                        const state = chatState(session, workingIds, attentionIds, replyNeededKeys)
                        const active = sessionKey(session) === selectedSessionKey

                        return (
                          <li key={sessionKey(session)}>
                            <ChatRow
                              active={active}
                              onSelect={() => setSelectedKey(sessionKey(session))}
                              session={session}
                              state={state}
                            />
                          </li>
                        )
                      })}
                    </ul>
                  </section>
                ))}
              </div>
            </div>
          </aside>
          <main className="flex min-h-0 min-w-0 flex-col overflow-hidden">
            {selected ? (
              <>
                <header className="flex shrink-0 items-start justify-between gap-3 border-b border-(--ui-stroke-tertiary) px-4 py-3">
                  <div className="min-w-0">
                    <div className="flex min-w-0 flex-wrap items-center gap-2">
                      <h3 className="min-w-0 truncate text-[0.9375rem] font-semibold tracking-tight">
                        {sessionTitle(selected)}
                      </h3>
                      <StatePill state={chatState(selected, workingIds, attentionIds, replyNeededKeys)} />
                    </div>
                    <p className="mt-1 truncate text-[length:var(--conversation-caption-font-size)] leading-(--conversation-caption-line-height) text-(--ui-text-tertiary)">
                      {selected.profile ? copy.profile(selected.profile) : selected.cwd || selected.id}
                    </p>
                  </div>
                  <div className="flex shrink-0 items-center gap-1">
                    <Button onClick={() => void onRefreshSessions()} size="sm" variant="ghost">
                      {t.common.refresh}
                    </Button>
                    <Button onClick={() => onOpenSession(selected.id)} size="sm" variant="outline">
                      {copy.open}
                    </Button>
                  </div>
                </header>
                <div className="min-h-0 flex-1 overflow-y-auto px-4 py-3">
                  {messagesLoading ? (
                    <InlineStatus icon={<Loader2 className="size-3.5 animate-spin" />} text={copy.loadingTranscript} />
                  ) : messagesError ? (
                    <InlineStatus text={`${copy.failedTranscript}: ${messagesError}`} />
                  ) : visibleMessages.length === 0 ? (
                    <InlineStatus text={copy.noPreview} />
                  ) : (
                    <div className="mx-auto max-w-3xl space-y-2">
                      {visibleMessages.map((message, index) => (
                        <TranscriptRow key={message.id || `${message.role}:${index}`} message={message} />
                      ))}
                    </div>
                  )}
                </div>
                <footer className="shrink-0 border-t border-(--ui-stroke-tertiary) px-4 py-3">
                  <div className="mx-auto grid max-w-3xl gap-2">
                    <Textarea
                      className="min-h-20 resize-none"
                      disabled={Boolean(sendingKey)}
                      onChange={event =>
                        setDraftBySession(current => ({ ...current, [selectedSessionKey]: event.target.value }))
                      }
                      onKeyDown={event => {
                        if ((event.metaKey || event.ctrlKey) && event.key === 'Enter') {
                          event.preventDefault()
                          void submitReply()
                        }
                      }}
                      placeholder={copy.replyPlaceholder}
                      value={draft}
                    />
                    <div className="flex items-center justify-between gap-2">
                      <span className="text-[length:var(--conversation-caption-font-size)] leading-(--conversation-caption-line-height) text-(--ui-text-tertiary)">
                        {copy.messages(selected.message_count)}
                      </span>
                      <Button disabled={!draft.trim() || Boolean(sendingKey)} onClick={() => void submitReply()} size="sm">
                        {sendingKey ? <Loader2 className="size-3.5 animate-spin" /> : <Send className="size-3.5" />}
                        {copy.answer}
                      </Button>
                    </div>
                  </div>
                </footer>
              </>
            ) : null}
          </main>
        </div>
      )}
    </PageSearchShell>
  )
}

function ChatRow({
  active,
  onSelect,
  session,
  state
}: {
  active: boolean
  onSelect: () => void
  session: SessionInfo
  state: ChatState
}) {
  return (
    <button
      className={cn(
        'grid w-full gap-1 rounded-md border border-transparent px-2.5 py-2 text-left transition-colors hover:bg-(--ui-row-hover-background)',
        active && 'border-(--ui-stroke-tertiary) bg-(--ui-row-active-background)'
      )}
      onClick={onSelect}
      type="button"
    >
      <span className="flex min-w-0 items-center gap-2">
        <StateDot state={state} />
        <span className="min-w-0 flex-1 truncate text-[0.8125rem] text-foreground">{sessionTitle(session)}</span>
        <span className="shrink-0 text-[0.625rem] text-(--ui-text-tertiary)">{formatAge(sessionTimestamp(session))}</span>
      </span>
      <span className="truncate pl-4 text-[length:var(--conversation-caption-font-size)] leading-(--conversation-caption-line-height) text-(--ui-text-tertiary)">
        {session.preview || workspaceLabel(session) || session.source || session.id}
      </span>
    </button>
  )
}

function QueueMetric({ icon, label, value }: { icon: React.ReactNode; label: string; value: number }) {
  return (
    <div className="grid gap-1 rounded-md border border-(--ui-stroke-tertiary) bg-(--ui-bg-elevated) px-2.5 py-2">
      <div className="flex items-center justify-between gap-2 text-(--ui-text-tertiary)">
        {icon}
        <span className="text-[0.6875rem] uppercase tracking-[0.08em]">{label}</span>
      </div>
      <div className="text-xl font-semibold leading-none tabular-nums text-foreground">{value}</div>
    </div>
  )
}

function GroupButton({
  active,
  icon,
  label,
  onClick
}: {
  active: boolean
  icon: React.ReactNode
  label: string
  onClick: () => void
}) {
  return (
    <button
      className={cn(
        'flex h-7 min-w-0 items-center justify-center gap-1 rounded-[4px] px-1.5 text-[0.6875rem] text-(--ui-text-tertiary) transition-colors hover:text-foreground',
        active && 'bg-(--ui-bg-elevated) text-foreground shadow-sm'
      )}
      onClick={onClick}
      type="button"
    >
      {icon}
      <span className="truncate">{label}</span>
    </button>
  )
}

function GroupIcon({ mode }: { mode: GroupMode }) {
  if (mode === 'profile') {
    return <Users className="size-3 text-(--ui-text-quaternary)" />
  }

  if (mode === 'workspace') {
    return <FolderOpen className="size-3 text-(--ui-text-quaternary)" />
  }

  return <Zap className="size-3 text-(--ui-text-quaternary)" />
}

function StatePill({ state }: { state: ChatState }) {
  const { t } = useI18n()
  const copy = t.activeChats

  return (
    <span className="inline-flex h-5 shrink-0 items-center gap-1 rounded-full border border-(--ui-stroke-tertiary) px-2 text-[0.6875rem] text-(--ui-text-secondary)">
      <StateDot state={state} />
      {copy[state]}
    </span>
  )
}

function StateDot({ state }: { state: ChatState }) {
  return (
    <span
      className={cn(
        'size-1.5 shrink-0 rounded-full',
        state === 'waiting' && 'bg-amber-400',
        state === 'running' && 'bg-cyan-400',
        state === 'recent' && 'bg-emerald-400',
        state === 'idle' && 'bg-(--ui-text-quaternary)'
      )}
    />
  )
}

function TranscriptRow({ message }: { message: ChatMessage }) {
  const text = chatMessageText(message).trim() || message.error || ''

  if (!text) {
    return null
  }

  return (
    <article className={cn('grid gap-1 rounded-md px-3 py-2', message.role === 'user' ? 'bg-(--ui-row-active-background)' : 'bg-(--ui-surface-muted)')}>
      <div className="text-[0.6875rem] uppercase tracking-[0.08em] text-(--ui-text-tertiary)">{message.role}</div>
      <p className="whitespace-pre-wrap break-words text-[0.8125rem] leading-5 text-(--ui-text-secondary)">{text}</p>
    </article>
  )
}

function EmptyState({ description, title }: { description: string; title: string }) {
  return (
    <div className="grid h-full place-items-center px-6">
      <div className="max-w-sm text-center">
        <div className="mx-auto mb-3 grid size-9 place-items-center rounded-md border border-(--ui-stroke-tertiary)">
          <MessageSquareText className="size-4 text-(--ui-text-tertiary)" />
        </div>
        <h3 className="text-sm font-semibold text-foreground">{title}</h3>
        <p className="mt-1 text-[0.8125rem] leading-5 text-(--ui-text-tertiary)">{description}</p>
      </div>
    </div>
  )
}

function InlineStatus({ icon, text }: { icon?: React.ReactNode; text: string }) {
  return (
    <div className="flex h-full items-center justify-center gap-2 text-[0.8125rem] text-(--ui-text-tertiary)">
      {icon}
      {text}
    </div>
  )
}
