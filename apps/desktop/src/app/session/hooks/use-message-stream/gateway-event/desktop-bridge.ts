import { writeAgentTerminalChunk } from '@/app/right-sidebar/terminal/agent-terminal-stream'
import { closeAgentTerminalByProc } from '@/app/right-sidebar/terminal/terminals'
import { applyDesktopLayoutPreset, revealDesktopPane } from '@/store/pane-focus'
import { recordAgentReaction } from '@/store/reactions-local'
import { setMessages } from '@/store/session'
import { $tipsEnabled, type ActiveTip, agentTipId, showTip } from '@/store/tips'

import type { GatewayEventContext } from './types'

/** Desktop-surface bridge events: agent terminal streaming, tips, pane
 *  reveal, layouts and message reactions. The read-back REQUESTS the agent
 *  blocks on (terminal/preview/window/tour) live in `server-requests.ts`. */
export function handleDesktopBridgeEvent(ctx: GatewayEventContext): boolean {
  const { event, payload, isActiveEvent } = ctx

  if (event.type === 'agent.terminal.output') {
    // Live chunk from a background process → its read-only agent terminal tab.
    writeAgentTerminalChunk(payload?.process_id ?? '', payload?.chunk ?? '')

    return true
  }

  if (event.type === 'terminal.close') {
    // Agent closed its own read-only tab via the desktop-gated close_terminal tool.
    // The process is untouched — this only drops the view.
    closeAgentTerminalByProc(payload?.process_id ?? '')

    return true
  }

  if (event.type === 'tip.show') {
    // tip tool: point the accent bubble at something and say one line about
    // it. Fire-and-forget — a tip is not a question, and blocking the turn on
    // one would stall the sentence the agent is in the middle of, so there is
    // nothing to answer and a refusal is simply a bubble that never appears.
    // Active session only: a background turn must never paint on the user's
    // screen (desktop AGENTS.md: offer, don't hijack).
    const selector = typeof payload?.selector === 'string' ? payload.selector : ''
    const text = typeof payload?.text === 'string' ? payload.text : ''

    // A tip with nothing to point at is just a notification, and the app
    // already has those. Dropping it here also stops a malformed event from
    // replacing a rotation tip with a bubble that dismisses itself a frame
    // later.
    if ($tipsEnabled.get() && isActiveEvent && selector && text) {
      showTip({
        side: (payload?.side as ActiveTip['side']) ?? 'top',
        targets: [selector],
        text,
        tipId: agentTipId(selector, text),
        title: typeof payload?.title === 'string' ? payload.title : undefined
      })
    }

    return true
  }

  if (event.type === 'pane.reveal') {
    // Agent revealed a pane via the desktop-gated focus_pane tool, in
    // response to an explicit user request. Active session only — a
    // background turn must never move the user's focus (desktop AGENTS.md:
    // offer, don't hijack).
    if (isActiveEvent) {
      revealDesktopPane(payload?.pane ?? '')
    }

    return true
  }

  if (event.type === 'layout.apply') {
    // Agent applied a layout preset via the desktop-gated apply_layout
    // tool. Same contract as pane.reveal: active session only, and the
    // preset resolves against the SAME layouts registry the picker reads,
    // so core, plugin, and user presets are all addressable.
    if (isActiveEvent) {
      applyDesktopLayoutPreset(typeof payload?.preset === 'string' ? payload.preset : '')
    }

    return true
  }

  if (event.type === 'message.reaction') {
    // The agent reacted to a message via the desktop-gated
    // react_to_message tool. Already persisted — this only paints it now
    // instead of at the next resume. Fresh ChatMessage object per change:
    // the runtime repository caches normalized ThreadMessages in a WeakMap
    // keyed by ChatMessage identity.
    const reactedRowId = payload?.row_id

    if (typeof reactedRowId === 'number') {
      const nextReactions = Array.isArray(payload?.reactions) ? payload.reactions : []
      const reactedRole = payload?.role === 'assistant' ? 'assistant' : 'user'

      setMessages(messages => {
        // Preferred leg: the message already knows its durable row id
        // (rehydrated transcript, or a live row that has round-tripped).
        const byRowId = messages.find(message => message.rowId === reactedRowId)

        if (byRowId) {
          // Overlay survives the end-of-turn resume, which rebuilds from
          // in-memory history that doesn't carry this mid-turn DB write.
          recordAgentReaction(reactedRowId, nextReactions)

          return messages.map(message =>
            message.rowId === reactedRowId ? { ...message, reactions: nextReactions } : message
          )
        }

        // Live leg: the targeted message is still optimistic (no rowId —
        // it hasn't round-tripped through a resume). The agent's default
        // target is the newest message of that role, so stamp the reaction
        // AND the now-known row id onto it. Without this the event matches
        // nothing and the reaction only appears after a reload.
        const lastIndex = messages.findLastIndex(message => message.role === reactedRole && message.rowId === undefined)

        if (lastIndex === -1) {
          return messages
        }

        recordAgentReaction(reactedRowId, nextReactions)

        return messages.map((message, index) =>
          index === lastIndex ? { ...message, rowId: reactedRowId, reactions: nextReactions } : message
        )
      })
    }

    return true
  }

  return false
}
