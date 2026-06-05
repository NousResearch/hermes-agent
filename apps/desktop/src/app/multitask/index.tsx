import { useStore } from '@nanostores/react'
import { useEffect, useMemo, useState } from 'react'

import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { SidebarPanelLabel } from '../shell/sidebar-label'
import { cn } from '@/lib/utils'
import { toChatMessages } from '@/lib/chat-messages'

import {
  $multitaskLayout,
  $multitaskSessionIds,
  $multitaskTileStates,
  clearAllMultitaskSessions,
  updateTileState
} from '@/store/multitask'

import { AddSessionModal } from './add-session-modal'
import { LayoutSwitcher } from './layout-switcher'
import { MultitaskTile } from './tile'

import type { RpcEvent, SessionResumeResponse } from '@/types/hermes'
import type { HermesGateway } from '@/hermes'
import type { ChatMessage } from '@/lib/chat-messages'

export interface MultitaskViewProps {
  gateway: HermesGateway | null | undefined
}

function makeAssistantMessage(text: string): ChatMessage {
  return {
    id: `msg-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
    role: 'assistant',
    parts: [{ type: 'text', text: '' + text }],
    timestamp: Date.now() / 1000
  }
}

function makeToolMessage(name: string): ChatMessage {
  return {
    id: `msg-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
    role: 'assistant',
    parts: [{ type: 'text' as const, text: `[Tool: ${name}]` }],
    timestamp: Date.now() / 1000
  }
}

// Fallback noop request for when gateway is not ready yet
const NOOP_REQUEST = async <T,>(_method: string, _params?: Record<string, unknown>): Promise<T> =>
  Promise.reject(new Error('Gateway not connected'))

export function MultitaskView({ gateway }: MultitaskViewProps) {
  const sessionIds = useStore($multitaskSessionIds)
  const layout = useStore($multitaskLayout)
  const tileStates = useStore($multitaskTileStates)
  const [modalOpen, setModalOpen] = useState(false)
  const gatewayConnected = gateway?.connectionState === 'open'

  // ── Resume sessions when gateway connects ─────────────────────────
  useEffect(() => {
    if (!gateway || gateway.connectionState !== 'open') return
    if (!sessionIds.length) return

    let cancelled = false

    for (const storedId of sessionIds) {
      // Read fresh state from store instead of captured closure
      const currentState = $multitaskTileStates.get().get(storedId)
      if (currentState?.runtimeSessionId) continue

      gateway
        .request<SessionResumeResponse>('session.resume', { session_id: storedId, cols: 96 })
        .then(res => {
          if (cancelled) return
          updateTileState(storedId, s => ({
            ...s,
            runtimeSessionId: res.session_id,
            model: res.info?.model ?? null,
            cwd: res.info?.cwd ?? null,
            messages: toChatMessages(res.messages),
            busy: false,
            awaitingResponse: false
          }))
        })
        .catch(err => {
          if (cancelled) return
          updateTileState(storedId, s => ({
            ...s,
            error: err instanceof Error ? err.message : 'Failed to resume session'
          }))
        })
    }

    return () => void (cancelled = true)
  }, [gatewayConnected ? 'connected' : 'disconnected', sessionIds.join(',')])

  // ── Subscribe to gateway events ───────────────────────────────────
  useEffect(() => {
    if (!gateway || gateway.connectionState !== 'open') return

    const unsub = gateway.onAny((event: RpcEvent) => {
      const sid = event.session_id
      if (!sid) return

      // Match runtime session id → stored session id via fresh store read
      let storedId: string | null = null
      for (const [stored, state] of $multitaskTileStates.get()) {
        if (state.runtimeSessionId === sid) {
          storedId = stored
          break
        }
      }
      if (!storedId) return

      switch (event.type) {
        case 'message.delta': {
          const payload = event.payload as { text?: string } | undefined
          if (payload?.text) {
            updateTileState(storedId, state => {
              const msgs = state.messages.map(m => ({ ...m }))
              const last = msgs[msgs.length - 1]
              if (last?.role === 'assistant' && last.parts?.length) {
                const parts = last.parts.map(p => ({ ...p }))
                const lastPart = parts[parts.length - 1]
                if ('text' in lastPart) {
                  parts[parts.length - 1] = { ...lastPart, text: lastPart.text + payload.text! }
                } else {
                  parts.push({ type: 'text', text: payload.text! })
                }
                msgs[msgs.length - 1] = { ...last, parts }
              } else {
                msgs.push(makeAssistantMessage(payload.text!))
              }
              return { ...state, messages: msgs }
            })
          }
          break
        }
        case 'message.complete': {
          updateTileState(storedId, state => ({
            ...state,
            busy: false,
            awaitingResponse: false,
            streamId: null
          }))
          break
        }
        case 'message.start': {
          updateTileState(storedId, state => ({
            ...state,
            busy: true,
            awaitingResponse: true
          }))
          break
        }
        case 'tool.start': {
          const payload = event.payload as { name?: string } | undefined
          if (payload?.name) {
            updateTileState(storedId, state => ({
              ...state,
              messages: [...state.messages, makeToolMessage(payload.name!)]
            }))
          }
          break
        }
        case 'error': {
          const errPayload = event.payload as { message?: string } | undefined
          updateTileState(storedId, state => ({
            ...state,
            busy: false,
            awaitingResponse: false,
            error: errPayload?.message ?? 'Gateway error'
          }))
          break
        }
      }
    })

    return () => unsub()
  }, [gatewayConnected ? 'connected' : 'disconnected'])

  // ── Grid class by layout ────────────────────────────────────────
  const gridClass = useMemo(() => {
    switch (layout) {
      case 'grid-2':
        return 'grid-cols-1 sm:grid-cols-2'
      case 'grid-3':
        return 'grid-cols-1 sm:grid-cols-2 lg:grid-cols-3'
      case 'horizontal':
        return 'grid-cols-1'
      case 'vertical':
        return 'grid-cols-1 sm:grid-cols-2'
      default:
        return 'grid-cols-1 sm:grid-cols-2'
    }
  }, [layout])

  const tileHeightClass = layout === 'horizontal' ? 'min-h-[30vh]' : 'min-h-[40vh]'

  return (
    <div className="flex h-full flex-col overflow-hidden bg-(--ui-app-background)">
      {/* Toolbar */}
      <div className="flex shrink-0 items-center gap-2 border-b border-(--ui-stroke-tertiary) px-4 py-2">
        <div className="flex items-center gap-2">
          <Codicon className="text-(--theme-primary)" name="split-horizontal" size="1rem" />
          <SidebarPanelLabel>Multitask</SidebarPanelLabel>
        </div>

        <div className="mx-2 h-4 w-px bg-(--ui-stroke-tertiary)" />

        <LayoutSwitcher />

        <div className="flex-1" />

        <div className="flex items-center gap-1">
          <Button
            className="flex h-7 items-center gap-1 rounded-md px-2 text-[0.75rem] font-medium text-(--ui-text-secondary) hover:bg-(--ui-control-hover-background) hover:text-foreground"
            onClick={() => setModalOpen(true)}
            title="Add a session to the multitask view"
            type="button"
            variant="ghost"
          >
            <Codicon name="add" className="size-3.5" />
            <span>Add Session</span>
          </Button>

          {sessionIds.length > 0 && (
            <Button
              className="flex h-7 items-center gap-1 rounded-md px-2 text-[0.75rem] text-(--ui-text-tertiary) hover:bg-(--ui-control-hover-background) hover:text-red-400"
              onClick={() => clearAllMultitaskSessions()}
              title="Remove all sessions"
              type="button"
              variant="ghost"
            >
              <Codicon name="close" className="size-3.5" />
              <span>Clear All</span>
            </Button>
          )}
        </div>
      </div>

      {/* Grid */}
      <div className="flex-1 overflow-auto">
        {sessionIds.length === 0 ? (
          <div className="flex h-full flex-col items-center justify-center gap-3 px-4">
            <Codicon className="text-(--ui-text-tertiary)" name="split-horizontal" size="2rem" />
            <p className="text-center text-[0.875rem] text-(--ui-text-secondary)">
              Monitor and interact with multiple sessions at once
            </p>
            <Button
              className="flex h-8 items-center gap-1.5 rounded-md px-3 text-[0.8125rem] font-medium text-foreground hover:bg-(--ui-control-hover-background)"
              onClick={() => setModalOpen(true)}
              title="Add a session"
              type="button"
              variant="ghost"
            >
              <Codicon name="add" className="size-4" />
              <span>Add Session</span>
            </Button>
          </div>
        ) : (
          <div className={cn('grid auto-rows-fr gap-2 p-3', gridClass)}>
            {sessionIds.map(sid => (
              <div key={sid} className={cn('flex flex-col overflow-hidden rounded-lg', tileHeightClass)}>
                <MultitaskTile
                  requestGateway={gateway?.request.bind(gateway) ?? NOOP_REQUEST}
                  storedSessionId={sid}
                />
              </div>
            ))}
          </div>
        )}
      </div>

      <AddSessionModal onOpenChange={setModalOpen} open={modalOpen} />
    </div>
  )
}
