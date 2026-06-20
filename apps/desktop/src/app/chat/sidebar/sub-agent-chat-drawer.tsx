import { useEffect, useMemo, useRef, useState } from 'react'

import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { Sheet, SheetContent, SheetDescription, SheetHeader, SheetTitle } from '@/components/ui/sheet'
import { chatWithSubAgent, type SubAgentPeer } from '@/hermes'
import { cn } from '@/lib/utils'

interface Msg {
  role: 'agent' | 'user'
  text: string
}

interface StoredSession {
  contextId: string
  title: string
  messages: Msg[]
  updatedAt: number
}

// 多会话持久在本机 localStorage:{ [子uid]: StoredSession[] }。
// 后端不动——contextId 驱动子侧 SessionDB 的会话连续性,这里只缓存消息供本机展示/恢复。
const LS_KEY = 'easyhermes.a2a.sessions.v1'

function loadSessions(uid: string): StoredSession[] {
  try {
    const all = JSON.parse(localStorage.getItem(LS_KEY) || '{}')
    return Array.isArray(all[uid]) ? all[uid] : []
  } catch {
    return []
  }
}

function saveSessions(uid: string, sessions: StoredSession[]) {
  try {
    const all = JSON.parse(localStorage.getItem(LS_KEY) || '{}')
    all[uid] = sessions
    localStorage.setItem(LS_KEY, JSON.stringify(all))
  } catch {
    /* localStorage 不可用就只在内存里活着,无妨 */
  }
}

/**
 * 子agent 对话抽屉(右侧滑出,与主对话并排)。每个子 agent 下可开多个会话:
 * 顶部会话列表(新建 + 历史)→ 下方多轮对话。会话存 localStorage(后端无改动)。
 */
export function SubAgentChatDrawer({
  onOpenChange,
  open,
  peer
}: {
  onOpenChange: (open: boolean) => void
  open: boolean
  peer: null | SubAgentPeer
}) {
  const [sessions, setSessions] = useState<StoredSession[]>([])
  // 当前会话:contextId 为空 = 尚未发送的新会话;messages 是当前展示的消息。
  const [current, setCurrent] = useState<{ contextId: string; messages: Msg[] }>({ contextId: '', messages: [] })
  const [input, setInput] = useState('')
  const [sending, setSending] = useState(false)
  const [error, setError] = useState<null | string>(null)
  const scrollRef = useRef<HTMLDivElement | null>(null)

  // 打开 / 换 peer → 载该子的历史会话,落到一个新会话上。
  useEffect(() => {
    if (open && peer) {
      setSessions(loadSessions(peer.uid))
      setCurrent({ contextId: '', messages: [] })
      setInput('')
      setError(null)
    }
  }, [open, peer?.uid])

  useEffect(() => {
    scrollRef.current?.scrollTo({ behavior: 'smooth', top: scrollRef.current.scrollHeight })
  }, [current.messages, sending])

  const sorted = useMemo(() => [...sessions].sort((a, b) => b.updatedAt - a.updatedAt), [sessions])

  const upsert = (uid: string, ctxId: string, messages: Msg[]) => {
    const title = (messages.find(m => m.role === 'user')?.text || '新会话').slice(0, 24)
    const sess: StoredSession = { contextId: ctxId, title, messages, updatedAt: Date.now() }
    const next = [sess, ...sessions.filter(s => s.contextId !== ctxId)]
    setSessions(next)
    saveSessions(uid, next)
  }

  const send = async () => {
    const text = input.trim()
    if (!text || !peer || sending) {
      return
    }

    setInput('')
    setError(null)
    const ctxId = current.contextId
    const base = current.messages
    setCurrent({ contextId: ctxId, messages: [...base, { role: 'user', text }] }) // 乐观展示
    setSending(true)

    try {
      const res = await chatWithSubAgent(peer.uid, text, ctxId)

      if (res.ok) {
        const newCtx = res.context_id || ctxId
        const msgs: Msg[] = [...base, { role: 'user', text }, { role: 'agent', text: res.answer || '(无回复)' }]
        setCurrent({ contextId: newCtx, messages: msgs })
        if (newCtx) {
          upsert(peer.uid, newCtx, msgs)
        }
      } else {
        setError(res.error || '对话失败')
      }
    } catch (e) {
      setError(String(e))
    } finally {
      setSending(false)
    }
  }

  return (
    <Sheet onOpenChange={onOpenChange} open={open}>
      <SheetContent className="flex flex-col gap-0 p-0 sm:max-w-md" side="right">
        <SheetHeader className="border-b border-(--ui-stroke-secondary) px-4 py-3">
          <SheetTitle>{peer?.name || peer?.uid || '子agent'}</SheetTitle>
          <SheetDescription>局域网直连 · 多轮对话(对方需在线)</SheetDescription>
        </SheetHeader>

        <div className="flex max-h-40 flex-col gap-px overflow-y-auto border-b border-(--ui-stroke-secondary) px-2 py-2">
          <button
            className="flex items-center gap-1.5 rounded-md px-2 py-1 text-left text-[0.8125rem] text-(--ui-text-info) hover:bg-(--chrome-action-hover)"
            onClick={() => {
              setCurrent({ contextId: '', messages: [] })
              setError(null)
            }}
            type="button"
          >
            <Codicon name="add" size="0.85rem" /> 新建会话
          </button>
          {sorted.map(s => (
            <button
              className={cn(
                'truncate rounded-md px-2 py-1 text-left text-[0.8125rem]',
                s.contextId === current.contextId
                  ? 'bg-(--ui-row-active-background) text-foreground'
                  : 'text-(--ui-text-secondary) hover:bg-(--chrome-action-hover) hover:text-foreground'
              )}
              key={s.contextId}
              onClick={() => {
                setCurrent({ contextId: s.contextId, messages: s.messages })
                setError(null)
              }}
              type="button"
            >
              {s.title || '(未命名会话)'}
            </button>
          ))}
        </div>

        <div className="flex-1 space-y-2 overflow-y-auto px-4 py-3" ref={scrollRef}>
          {current.messages.length === 0 && (
            <p className="text-sm text-muted-foreground">给「{peer?.name || '子agent'}」发条消息开始对话…</p>
          )}
          {current.messages.map((m, i) => (
            <div className={m.role === 'user' ? 'text-right' : 'text-left'} key={i}>
              <span
                className={cn(
                  'inline-block max-w-[85%] whitespace-pre-wrap rounded-lg px-3 py-1.5 text-sm',
                  m.role === 'user' ? 'bg-primary text-primary-foreground' : 'bg-muted text-foreground'
                )}
              >
                {m.text}
              </span>
            </div>
          ))}
          {sending && <p className="text-xs text-muted-foreground">对方思考中…</p>}
          {error && <p className="text-xs text-destructive">{error}</p>}
        </div>

        <form
          className="flex gap-2 border-t border-(--ui-stroke-secondary) p-3"
          onSubmit={e => {
            e.preventDefault()
            void send()
          }}
        >
          <input
            autoFocus
            className="flex-1 rounded-md border border-(--ui-stroke-secondary) bg-transparent px-3 py-1.5 text-sm outline-none focus:ring-2 focus:ring-ring/40"
            disabled={sending || !peer}
            onChange={e => setInput(e.target.value)}
            placeholder="输入消息…"
            value={input}
          />
          <Button disabled={sending || !input.trim() || !peer} type="submit">
            发送
          </Button>
        </form>
      </SheetContent>
    </Sheet>
  )
}
