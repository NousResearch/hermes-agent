import { useStore } from '@nanostores/react'
import { useCallback, useEffect, useRef, useState } from 'react'
import { Streamdown } from 'streamdown'

import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { Textarea } from '@/components/ui/textarea'
import { Tip } from '@/components/ui/tooltip'
import type { IxChatConversationMeta, IxChatDisplayItem, IxChatRendererEvent } from '@/global'
import { openExternalLink } from '@/lib/external-link'
import { cn } from '@/lib/utils'
import { notifyError } from '@/store/notifications'

import { $ixPendingSkill } from './copilot-store'
import skillsData from './data/skills.json'
import { LoginPane } from './login-pane'
import type { IxSkillItem } from './types'

/**
 * The native IX Agency copilot: LiteLLM streaming + the admin-mcp tool loop
 * run in the MAIN process (electron/ix-chat.ts); this pane only renders
 * display events and hosts the two channels the model can never reach —
 * the Confirm/Cancel buttons of the write gate and the native OTP sign-in.
 *
 * Everything is gated behind the portal OTP login (probed main-side against
 * the persist:ix-agency-portal session): signed out ⇒ the NATIVE login form
 * IS the pane (no webview — main drives /api/auth/otp/* directly).
 */

const SKILLS: IxSkillItem[] = Array.isArray(skillsData.items) ? (skillsData.items as IxSkillItem[]) : []

const MAX_ACTIVE_SKILLS = 3

const AUTH_POLL_MS = 20_000

interface VizArtifact {
  type: string
  url: string
}

/** Portal-style inline viz: tool results shaped {viz:true,type,url}. */
function parseVizArtifact(result: string | undefined): null | VizArtifact {
  if (!result || result.length > 200_000) {
    return null
  }

  try {
    const parsed = JSON.parse(result)

    if (parsed && parsed.viz === true && typeof parsed.url === 'string' && /^https?:\/\//.test(parsed.url)) {
      return { type: String(parsed.type ?? 'link'), url: parsed.url }
    }
  } catch {
    // not JSON — no artifact
  }

  return null
}

function VizArtifactView({ artifact }: { artifact: VizArtifact }) {
  if (artifact.type === 'image') {
    return (
      <button className="block max-w-md cursor-zoom-in" onClick={() => openExternalLink(artifact.url)} type="button">
        <img
          alt="Rendered visualization"
          className="rounded-md border border-(--ui-border-primary)"
          src={artifact.url}
        />
      </button>
    )
  }

  if (artifact.type === 'video') {
    return <video className="max-w-md rounded-md border border-(--ui-border-primary)" controls src={artifact.url} />
  }

  return (
    <Button onClick={() => openExternalLink(artifact.url)} size="xs" variant="outline">
      <Codicon name="link-external" size="0.75rem" />
      Open {artifact.type || 'artifact'}
    </Button>
  )
}

const TOOL_STATUS_META: Record<string, { icon: string; tone: string; label: string }> = {
  approved: { icon: 'check-all', tone: 'text-emerald-500', label: 'executed (user-approved write)' },
  error: { icon: 'error', tone: 'text-red-500', label: 'failed' },
  gated: { icon: 'shield', tone: 'text-amber-500', label: 'awaiting confirmation' },
  ok: { icon: 'check', tone: 'text-emerald-500', label: 'ok' },
  running: { icon: 'loading~spin', tone: 'text-muted-foreground', label: 'running' }
}

/** Collapsed tool-call card (portal pattern): name + status, expandable. */
function ToolCard({ item }: { item: IxChatDisplayItem & { running?: boolean } }) {
  const status = item.running ? 'running' : (item.status ?? 'ok')
  const meta = TOOL_STATUS_META[status] ?? TOOL_STATUS_META.ok
  const viz = parseVizArtifact(item.result)

  return (
    <div className="space-y-2">
      <details className="rounded-md border border-(--ui-border-primary) bg-(--ui-bg-quinary) text-xs">
        <summary className="flex cursor-pointer items-center gap-2 px-2.5 py-1.5 select-none">
          <Codicon className={meta.tone} name={meta.icon} size="0.8125rem" />
          <code className="min-w-0 flex-1 truncate font-mono text-[0.7rem]">{item.name}</code>
          <span className="shrink-0 text-[0.65rem] text-muted-foreground/70">{meta.label}</span>
        </summary>
        <div className="space-y-2 border-t border-(--ui-border-primary) px-2.5 py-2">
          {item.argsSummary && (
            <div className="break-all">
              <span className="font-medium text-muted-foreground">args </span>
              <code className="font-mono text-[0.68rem]">{item.argsSummary}</code>
            </div>
          )}
          {item.result && (
            <pre className="max-h-64 overflow-auto rounded bg-(--ui-bg-quaternary) p-2 font-mono text-[0.68rem] whitespace-pre-wrap">
              {item.result.slice(0, 6000)}
              {item.result.length > 6000 ? '\n…(truncated in view)' : ''}
            </pre>
          )}
        </div>
      </details>
      {viz && <VizArtifactView artifact={viz} />}
    </div>
  )
}

/** The write gate's Confirm/Cancel card — the ONLY approval channel. */
function ConfirmCard({
  item,
  onDecision
}: {
  item: IxChatDisplayItem
  onDecision: (nonce: string, approve: boolean) => void
}) {
  const state = item.state ?? 'pending'

  return (
    <div
      className={cn(
        'rounded-md border px-3 py-2.5 text-xs',
        state === 'pending'
          ? 'border-amber-500/40 bg-amber-500/10'
          : 'border-(--ui-border-primary) bg-(--ui-bg-quinary)'
      )}
    >
      <div className="flex items-center gap-2">
        <Codicon className="text-amber-500" name="shield" size="0.875rem" />
        <span className="font-medium">Write action requires your confirmation</span>
      </div>
      <div className="mt-1.5 break-all">
        <code className="font-mono text-[0.7rem]">{item.name}</code>
        {item.argsSummary && (
          <code className="ml-1 font-mono text-[0.68rem] text-muted-foreground">{item.argsSummary}</code>
        )}
      </div>
      {state === 'pending' && item.nonce ? (
        <div className="mt-2 flex gap-2">
          <Button onClick={() => onDecision(item.nonce as string, true)} size="xs">
            Confirm
          </Button>
          <Button onClick={() => onDecision(item.nonce as string, false)} size="xs" variant="secondary">
            Cancel
          </Button>
        </div>
      ) : (
        <div className="mt-1.5 text-[0.68rem] text-muted-foreground">
          {state === 'approved' ? 'Approved — executed with the approved arguments.' : 'Denied — not executed.'}
        </div>
      )}
    </div>
  )
}

function AssistantMarkdown({ text }: { text: string }) {
  return (
    <div
      className="prose prose-sm dark:prose-invert max-w-none text-sm [&_pre]:text-xs"
      onClickCapture={event => {
        const anchor = (event.target as HTMLElement).closest('a')

        if (anchor?.href) {
          event.preventDefault()
          event.stopPropagation()
          openExternalLink(anchor.href)
        }
      }}
    >
      <Streamdown>{text}</Streamdown>
    </div>
  )
}

type LiveItem = IxChatDisplayItem & { running?: boolean }

export function CopilotTab() {
  const bridge = window.hermesDesktop?.ixAgency
  const pendingSkill = useStore($ixPendingSkill)

  // Login enforcement (renderer side of it — main enforces on every IPC).
  const [auth, setAuth] = useState<null | { authenticated: boolean; detail: string }>(null)

  // Conversation state.
  const [conversations, setConversations] = useState<IxChatConversationMeta[]>([])
  const [conversationId, setConversationId] = useState<null | string>(null)
  const [items, setItems] = useState<LiveItem[]>([])
  const [streamingText, setStreamingText] = useState('')
  const [busy, setBusy] = useState(false)

  // Composer + model + skills.
  const [draft, setDraft] = useState('')
  const [models, setModels] = useState<{ id: string; label: string }[]>([])
  const [model, setModel] = useState('')
  const [activeSkillIds, setActiveSkillIds] = useState<string[]>([])
  const [lockedSkills, setLockedSkills] = useState<null | string[]>(null)

  const scrollRef = useRef<HTMLDivElement | null>(null)
  const conversationIdRef = useRef<null | string>(null)
  const busyRef = useRef(false)

  conversationIdRef.current = conversationId
  busyRef.current = busy

  const refreshAuth = useCallback(async () => {
    if (!bridge?.authStatus) {
      return
    }

    try {
      setAuth(await bridge.authStatus())
    } catch {
      setAuth({ authenticated: false, detail: 'Auth probe failed' })
    }
  }, [bridge])

  useEffect(() => {
    void refreshAuth()

    const timer = setInterval(() => void refreshAuth(), AUTH_POLL_MS)
    const onFocus = () => void refreshAuth()

    window.addEventListener('focus', onFocus)

    return () => {
      clearInterval(timer)
      window.removeEventListener('focus', onFocus)
    }
  }, [refreshAuth])

  const refreshConversations = useCallback(async () => {
    if (!bridge?.chatList) {
      return
    }

    try {
      setConversations(await bridge.chatList())
    } catch {
      // list is cosmetic; sends still work
    }
  }, [bridge])

  useEffect(() => {
    void refreshConversations()

    void bridge
      ?.chatModels()
      .then(result => {
        setModels(result.models)
        setModel(current => current || result.defaultModel)
      })
      .catch(() => setModels([]))
  }, [bridge, refreshConversations])

  // "Run natively" hand-off from the Org skills tab.
  useEffect(() => {
    if (!pendingSkill) {
      return
    }

    $ixPendingSkill.set(null)
    setConversationId(null)
    setItems([])
    setLockedSkills(null)
    setActiveSkillIds([pendingSkill.id])
    setDraft(pendingSkill.starterPrompts[0] ?? `Run the "${pendingSkill.title}" playbook.`)
  }, [pendingSkill])

  // Streaming events from the main-process loop.
  useEffect(() => {
    if (!bridge?.onChatEvent) {
      return
    }

    return bridge.onChatEvent((event: IxChatRendererEvent) => {
      if (event.conversationId !== conversationIdRef.current) {
        // First turn of a new conversation: the main process mints the id
        // mid-flight, so adopt the stream we just started.
        if (conversationIdRef.current !== null || !busyRef.current) {
          return
        }

        conversationIdRef.current = event.conversationId
        setConversationId(event.conversationId)
      }

      if (event.type === 'text-delta') {
        setStreamingText(current => current + (event.delta ?? ''))
      } else if (event.type === 'tool-call') {
        setStreamingText(current => {
          if (current) {
            setItems(list => [...list, { kind: 'assistant', text: current, at: Date.now() }])
          }

          return ''
        })
        setItems(list => [
          ...list,
          { kind: 'tool', name: event.name, argsSummary: event.argsSummary, running: true, at: Date.now() }
        ])
      } else if (event.type === 'tool-result') {
        setItems(list => {
          const next = [...list]
          const index = next.findLastIndex(item => item.kind === 'tool' && item.running && item.name === event.name)

          if (index >= 0) {
            next[index] = { ...next[index], running: false, status: event.status, result: event.result }
          }

          return next
        })
      } else if (event.type === 'confirmation-required') {
        setItems(list => [
          ...list,
          {
            kind: 'confirm',
            nonce: event.nonce,
            name: event.tool,
            argsSummary: event.argsSummary,
            state: 'pending',
            at: Date.now()
          }
        ])
      } else if (event.type === 'error') {
        notifyError(new Error(event.message ?? 'Chat failed'), 'IX Agency chat')
      }
    })
  }, [bridge])

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight })
  }, [items, streamingText])

  const loadConversation = useCallback(
    async (id: string) => {
      if (!bridge?.chatGet) {
        return
      }

      try {
        const detail = await bridge.chatGet(id)

        if (detail) {
          setConversationId(detail.id)
          setItems(detail.display)
          setModel(detail.model)
          setLockedSkills(detail.skills)
          setStreamingText('')
        }
      } catch (error) {
        notifyError(error, 'Failed to load conversation')
      }
    },
    [bridge]
  )

  const send = useCallback(
    async (text: string) => {
      if (!bridge?.chatSend || !text.trim() || busy) {
        return
      }

      setBusy(true)
      setDraft('')
      setItems(list => [...list, { kind: 'user', text: text.trim(), at: Date.now() }])
      setStreamingText('')

      const isNew = !conversationIdRef.current

      const skills = isNew
        ? SKILLS.filter(skill => activeSkillIds.includes(skill.id))
            .slice(0, MAX_ACTIVE_SKILLS)
            .map(skill => ({ name: skill.title, content: skill.content }))
        : undefined

      try {
        // For a brand-new conversation we don't know the id until the main
        // process mints it, so early events can be missed; the final chatGet
        // sync below restores the canonical transcript either way.
        const result = await bridge.chatSend({
          conversationId: conversationIdRef.current,
          text: text.trim(),
          model,
          skills
        })

        conversationIdRef.current = result.conversationId
        setConversationId(result.conversationId)

        if (isNew && skills?.length) {
          setLockedSkills(skills.map(skill => skill.name))
        }

        const detail = await bridge.chatGet(result.conversationId)

        if (detail) {
          setItems(detail.display)
        }

        setStreamingText('')
        void refreshConversations()
      } catch (error) {
        notifyError(error, 'IX Agency chat')
      } finally {
        setBusy(false)
      }
    },
    [activeSkillIds, bridge, busy, model, refreshConversations]
  )

  const decide = useCallback(
    async (nonce: string, approve: boolean) => {
      if (!bridge?.chatConfirm || !conversationIdRef.current) {
        return
      }

      try {
        const result = await bridge.chatConfirm({ conversationId: conversationIdRef.current, nonce, approve })

        setItems(list =>
          list.map(item => (item.kind === 'confirm' && item.nonce === nonce ? { ...item, state: result.state } : item))
        )

        if (result.ok && approve) {
          // The model must re-issue the gated call; this follow-up turn gives
          // it the chance (execution then uses the frozen approved args).
          await send('Confirmed — proceed with the approved action.')
        }
      } catch (error) {
        notifyError(error, 'Confirmation failed')
      }
    },
    [bridge, send]
  )

  if (!bridge) {
    return (
      <div className="mx-auto max-w-2xl px-5 py-6 text-xs text-muted-foreground">
        The IX Agency bridge is unavailable in this build.
      </div>
    )
  }

  // ── Login gate: the NATIVE OTP form IS the pane until the probe passes ──
  if (!auth?.authenticated) {
    return <LoginPane detail={auth?.detail} onSignedIn={() => void refreshAuth()} />
  }

  const activeSkillLabels = lockedSkills ?? SKILLS.filter(s => activeSkillIds.includes(s.id)).map(s => s.title)

  return (
    <div className="flex h-full min-h-0">
      {/* Conversations rail */}
      <div className="flex w-52 shrink-0 flex-col border-r border-(--ui-border-primary)">
        <div className="flex items-center gap-2 px-3 py-2">
          <span className="min-w-0 flex-1 truncate text-[0.65rem] font-medium tracking-wider text-muted-foreground/60 uppercase">
            Conversations
          </span>
          <Tip label="New conversation" side="bottom">
            <Button
              onClick={() => {
                setConversationId(null)
                setItems([])
                setStreamingText('')
                setLockedSkills(null)
                setActiveSkillIds([])
              }}
              size="icon-xs"
              variant="ghost"
            >
              <Codicon name="add" size="0.8125rem" />
            </Button>
          </Tip>
        </div>
        <div className="min-h-0 flex-1 overflow-y-auto px-1.5 pb-2">
          {conversations.map(conversation => (
            <button
              className={cn(
                'block w-full truncate rounded px-2 py-1.5 text-left text-xs hover:bg-(--chrome-action-hover)',
                conversation.id === conversationId && 'bg-(--ui-bg-quaternary) font-medium'
              )}
              key={conversation.id}
              onClick={() => void loadConversation(conversation.id)}
              type="button"
            >
              {conversation.title || 'New conversation'}
            </button>
          ))}
          {!conversations.length && (
            <p className="px-2 py-1.5 text-[0.68rem] text-muted-foreground/70">No conversations yet.</p>
          )}
        </div>
      </div>

      {/* Chat column */}
      <div className="flex min-w-0 flex-1 flex-col">
        {/* Skill chips (locked once the conversation exists) */}
        <div className="flex items-center gap-1.5 overflow-x-auto border-b border-(--ui-border-primary) px-3 py-1.5">
          <span className="shrink-0 text-[0.65rem] font-medium tracking-wider text-muted-foreground/60 uppercase">
            Skills
          </span>
          {lockedSkills ? (
            activeSkillLabels.length ? (
              activeSkillLabels.map(label => (
                <span
                  className="shrink-0 rounded-full bg-(--ui-bg-quaternary) px-2 py-0.5 text-[0.68rem] font-medium"
                  key={label}
                >
                  {label}
                </span>
              ))
            ) : (
              <span className="text-[0.68rem] text-muted-foreground/70">none active in this conversation</span>
            )
          ) : (
            SKILLS.map(skill => {
              const active = activeSkillIds.includes(skill.id)

              return (
                <Tip key={skill.id} label={skill.description} side="bottom">
                  <button
                    className={cn(
                      'shrink-0 rounded-full px-2 py-0.5 text-[0.68rem]',
                      active
                        ? 'bg-primary font-medium text-primary-foreground'
                        : 'bg-(--ui-bg-quinary) text-muted-foreground hover:bg-(--chrome-action-hover)'
                    )}
                    onClick={() =>
                      setActiveSkillIds(current =>
                        active
                          ? current.filter(id => id !== skill.id)
                          : [...current, skill.id].slice(-MAX_ACTIVE_SKILLS)
                      )
                    }
                    type="button"
                  >
                    {skill.title}
                  </button>
                </Tip>
              )
            })
          )}
        </div>

        {/* Transcript */}
        <div className="min-h-0 flex-1 overflow-y-auto px-4 py-3" ref={scrollRef}>
          <div className="mx-auto max-w-3xl space-y-3">
            {!items.length && !streamingText && (
              <div className="py-10 text-center text-xs text-muted-foreground">
                <Codicon className="mb-2 text-muted-foreground/50" name="sparkle" size="1.5rem" />
                <p className="font-medium">IX Agency copilot — native</p>
                <p className="mt-1">
                  LiteLLM streaming with the full admin-mcp tool estate. Writes always stop for your confirmation.
                </p>
              </div>
            )}
            {items.map((item, index) => {
              if (item.kind === 'user') {
                return (
                  <div className="flex justify-end" key={index}>
                    <div className="max-w-[85%] rounded-lg bg-(--ui-bg-quaternary) px-3 py-2 text-sm whitespace-pre-wrap">
                      {item.text}
                    </div>
                  </div>
                )
              }

              if (item.kind === 'assistant') {
                return <AssistantMarkdown key={index} text={item.text ?? ''} />
              }

              if (item.kind === 'tool') {
                return <ToolCard item={item} key={index} />
              }

              return (
                <ConfirmCard item={item} key={index} onDecision={(nonce, approve) => void decide(nonce, approve)} />
              )
            })}
            {streamingText && <AssistantMarkdown text={streamingText} />}
            {busy && !streamingText && (
              <div className="flex items-center gap-2 text-xs text-muted-foreground">
                <Codicon name="loading~spin" size="0.8125rem" />
                Thinking…
              </div>
            )}
          </div>
        </div>

        {/* Composer */}
        <div className="border-t border-(--ui-border-primary) px-4 py-3">
          <div className="mx-auto max-w-3xl space-y-2">
            <Textarea
              className="min-h-16 resize-none text-sm"
              onChange={event => setDraft(event.target.value)}
              onKeyDown={event => {
                if (event.key === 'Enter' && !event.shiftKey) {
                  event.preventDefault()
                  void send(draft)
                }
              }}
              placeholder="Ask across the whole admin estate — every MCP tile is wired in…"
              value={draft}
            />
            <div className="flex items-center gap-2">
              <select
                className="h-6 rounded border border-(--ui-border-primary) bg-(--ui-bg-quinary) px-1.5 text-[0.7rem]"
                onChange={event => setModel(event.target.value)}
                value={model}
              >
                {models.map(m => (
                  <option key={m.id} value={m.id}>
                    {m.label}
                  </option>
                ))}
              </select>
              <span className="min-w-0 flex-1 truncate text-[0.65rem] text-muted-foreground/60">
                Writes require the Confirm button — the model cannot approve itself.
              </span>
              <Button disabled={busy || !draft.trim()} onClick={() => void send(draft)} size="sm">
                {busy ? <Codicon name="loading~spin" size="0.8125rem" /> : <Codicon name="send" size="0.8125rem" />}
                Send
              </Button>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
