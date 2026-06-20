import { useEffect, useRef, useState } from 'react'

import { Button } from '@/components/ui/button'
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from '@/components/ui/dialog'
import { chatWithSubAgent, type SubAgentPeer } from '@/hermes'
import { cn } from '@/lib/utils'

interface Msg {
  role: 'agent' | 'user'
  text: string
}

/**
 * 轻量子agent对话视图(LAN 直连,多轮)。
 *
 * 刻意**不**接 Hermes 网关 session.* 协议:就是 useState 消息列表 + 每轮 POST /api/kari/org/chat,
 * 持有 server 回的 contextId 续聊。换 peer 即开新会话(清空 + 新 contextId)。
 */
export function SubAgentChatDialog({
  onOpenChange,
  open,
  peer
}: {
  onOpenChange: (open: boolean) => void
  open: boolean
  peer: SubAgentPeer | null
}) {
  const [messages, setMessages] = useState<Msg[]>([])
  const [input, setInput] = useState('')
  const [sending, setSending] = useState(false)
  const [error, setError] = useState<null | string>(null)
  const contextIdRef = useRef('')
  const scrollRef = useRef<HTMLDivElement | null>(null)

  // 打开 / 换 peer → 重置会话(新 contextId)。
  useEffect(() => {
    if (open) {
      setMessages([])
      setInput('')
      setError(null)
      contextIdRef.current = ''
    }
  }, [open, peer?.uid])

  useEffect(() => {
    scrollRef.current?.scrollTo({ behavior: 'smooth', top: scrollRef.current.scrollHeight })
  }, [messages, sending])

  const send = async () => {
    const text = input.trim()
    if (!text || !peer || sending) {
      return
    }

    setInput('')
    setError(null)
    setMessages(m => [...m, { role: 'user', text }])
    setSending(true)

    try {
      const res = await chatWithSubAgent(peer.uid, text, contextIdRef.current)

      if (res.ok) {
        contextIdRef.current = res.context_id || contextIdRef.current
        setMessages(m => [...m, { role: 'agent', text: res.answer || '(无回复)' }])
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
    <Dialog onOpenChange={onOpenChange} open={open}>
      <DialogContent className="flex h-[32rem] max-w-lg flex-col gap-0 p-0">
        <DialogHeader className="border-b px-4 py-3">
          <DialogTitle>{peer?.name || peer?.uid || '子agent'}</DialogTitle>
          <DialogDescription>局域网直连 · 多轮对话(对方需在线)</DialogDescription>
        </DialogHeader>

        <div className="flex-1 space-y-2 overflow-y-auto px-4 py-3" ref={scrollRef}>
          {messages.length === 0 && (
            <p className="text-sm text-muted-foreground">给「{peer?.name || '子agent'}」发条消息开始对话…</p>
          )}
          {messages.map((m, i) => (
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
          className="flex gap-2 border-t p-3"
          onSubmit={e => {
            e.preventDefault()
            void send()
          }}
        >
          <input
            autoFocus
            className="flex-1 rounded-md border bg-transparent px-3 py-1.5 text-sm outline-none focus:ring-2 focus:ring-ring/40"
            disabled={sending || !peer}
            onChange={e => setInput(e.target.value)}
            placeholder="输入消息…"
            value={input}
          />
          <Button disabled={sending || !input.trim() || !peer} type="submit">
            发送
          </Button>
        </form>
      </DialogContent>
    </Dialog>
  )
}
