import { useStore } from '@nanostores/react'
import { useEffect, useRef } from 'react'

import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { cn } from '@/lib/utils'

import { $multitaskTileStates, removeMultitaskSession } from '@/store/multitask'

import type { ChatMessage } from '@/lib/chat-messages'

import { MultitaskComposer } from './tile-composer'

export interface MultitaskTileProps {
  storedSessionId: string
  requestGateway: <T>(method: string, params?: Record<string, unknown>) => Promise<T>
}

export function MultitaskTile({ storedSessionId, requestGateway }: MultitaskTileProps) {
  const state = useStore($multitaskTileStates).get(storedSessionId)
  const messagesEndRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [state?.messages.length])

  if (!state) return null

  const title = state.title || state.storedSessionId.slice(0, 10) + '…'
  const msgCount = state.messages.length

  return (
    <div className="flex min-h-0 flex-1 flex-col overflow-hidden rounded-lg border border-(--ui-stroke-tertiary) bg-(--ui-chat-surface-background)">
      {/* Header */}
      <div className="flex shrink-0 items-center gap-2 border-b border-(--ui-stroke-tertiary) px-3 py-1.5">
        <Codicon name="comment" className="size-3.5 shrink-0 text-(--ui-text-tertiary)" />
        <span className="min-w-0 flex-1 truncate text-[0.8125rem] font-medium text-foreground" title={title}>
          {title}
        </span>
        {state.busy && (
          <Codicon name="loading" className="size-3.5 animate-spin text-(--ui-text-secondary)" />
        )}
        <span className="shrink-0 text-[0.6875rem] text-(--ui-text-tertiary)">
          {msgCount} msgs
        </span>
        <Button
          className="size-5 rounded p-0 text-(--ui-text-tertiary) hover:text-foreground"
          onClick={() => removeMultitaskSession(storedSessionId)}
          title="Remove this session from the multitask view"
          type="button"
          variant="ghost"
        >
          <Codicon name="close" className="size-3.5" />
        </Button>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto px-3 py-2">
        {state.messages.length === 0 && (
          <p className="pt-4 text-center text-[0.75rem] text-(--ui-text-tertiary)">
            {state.busy ? 'Streaming…' : 'No messages yet'}
          </p>
        )}
        <div className="space-y-1.5">
          {state.messages.map((msg, i) => (
            <TileMessage key={msg.id ?? i} message={msg} />
          ))}
        </div>
        {state.error && (
          <div className="mt-2 rounded bg-red-500/10 px-2 py-1 text-[0.75rem] text-red-400">
            {state.error}
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Composer */}
      <MultitaskComposer
        busy={state.busy}
        requestGateway={requestGateway}
        runtimeSessionId={state.runtimeSessionId}
        storedSessionId={storedSessionId}
      />
    </div>
  )
}

function TileMessage({ message }: { message: ChatMessage }) {
  const text = message.parts?.map(p => ('text' in p ? p.text : '')).join('') || ''
  const isUser = message.role === 'user'

  return (
    <div className={cn('flex', isUser ? 'justify-end' : 'justify-start')}>
      <div
        className={cn(
          'max-w-[85%] rounded-lg px-2.5 py-1 text-[0.75rem] leading-relaxed',
          isUser
            ? 'bg-(--ui-control-active-background) text-foreground'
            : 'bg-(--ui-control-hover-background) text-(--ui-text-secondary)'
        )}
      >
        {text || (
          <span className="italic text-(--ui-text-tertiary)">[{message.role} message]</span>
        )}
      </div>
    </div>
  )
}
