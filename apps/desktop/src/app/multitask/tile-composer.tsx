import { type FormEvent, useRef, useState } from 'react'

import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'

import { updateTileState } from '@/store/multitask'

export interface MultitaskComposerProps {
  busy: boolean
  runtimeSessionId: string | null
  storedSessionId: string
  requestGateway: <T>(method: string, params?: Record<string, unknown>) => Promise<T>
}

export function MultitaskComposer({
  busy,
  runtimeSessionId,
  storedSessionId,
  requestGateway
}: MultitaskComposerProps) {
  const [value, setValue] = useState('')
  const inputRef = useRef<HTMLTextAreaElement>(null)

  const handleSend = async (e: FormEvent) => {
    e.preventDefault()
    const text = value.trim()
    if (!text || !runtimeSessionId || busy) return

    setValue('')

    // Optimistic: add user message
    updateTileState(storedSessionId, state => ({
      ...state,
      busy: true,
      awaitingResponse: true,
      messages: [
        ...state.messages,
        {
          id: `optimistic-${Date.now()}`,
          role: 'user',
          parts: [{ type: 'text', text }],
          created_at: Date.now() / 1000
        }
      ]
    }))

    try {
      await requestGateway('session.send', {
        session_id: runtimeSessionId,
        message: text
      })
    } catch (err) {
      updateTileState(storedSessionId, state => ({
        ...state,
        busy: false,
        awaitingResponse: false,
        error: err instanceof Error ? err.message : 'Send failed'
      }))
    }
  }

  const handleCancel = async () => {
    if (!runtimeSessionId) return
    try {
      await requestGateway('session.cancel', { session_id: runtimeSessionId })
    } catch {
      // best-effort
    }
    updateTileState(storedSessionId, state => ({
      ...state,
      busy: false,
      awaitingResponse: false
    }))
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend(e as unknown as FormEvent)
    }
  }

  return (
    <form className="flex items-center gap-1 border-t border-(--ui-stroke-tertiary) p-1.5" onSubmit={handleSend}>
      <textarea
        ref={inputRef}
        className="min-h-0 flex-1 resize-none rounded bg-transparent px-2 py-1 text-[0.75rem] text-foreground outline-none placeholder:text-(--ui-text-tertiary)"
        disabled={busy || !runtimeSessionId}
        onKeyDown={handleKeyDown}
        placeholder={busy ? 'Waiting for response…' : !runtimeSessionId ? 'Resuming…' : 'Send a message…'}
        rows={1}
        value={value}
        onChange={e => setValue(e.target.value)}
      />

      {busy ? (
        <Button
          className="size-6 rounded p-0 text-(--ui-text-secondary) hover:text-red-400"
          onClick={handleCancel}
          title="Cancel"
          type="button"
          variant="ghost"
        >
          <Codicon name="close" className="size-3.5" />
        </Button>
      ) : (
        <Button
          className="size-6 rounded p-0 text-(--ui-text-secondary) hover:text-foreground disabled:opacity-30"
          disabled={!value.trim() || !runtimeSessionId}
          title="Send"
          type="submit"
          variant="ghost"
        >
          <Codicon name="arrow-up" className="size-3.5" />
        </Button>
      )}
    </form>
  )
}
