import { computed } from 'nanostores'

import type { ClientSessionState } from '@/app/types'
import type { ChatMessagePart } from '@/lib/chat-messages'
import { stableRecord } from '@/lib/stable-array'
import { isSilentTool } from '@/lib/tool-render-class'

import { $statusItemsBySession, type ComposerStatusItem } from './composer-status'
import { $sessions, lineageAliases } from './session'
import { $sessionStates } from './session-states'

export type SidebarActivity = Extract<ChatMessagePart, { type: 'tool-call' }> | ComposerStatusItem

/** Only the current, pending reply can name a live tool. History can contain
 * unresolved calls after interruptions; those are not evidence of ongoing work. */
function liveTool(state: ClientSessionState | undefined): SidebarActivity | undefined {
  if (!state?.busy || state.interrupted) {
    return
  }

  for (let i = state.messages.length - 1; i >= 0; i--) {
    const message = state.messages[i]!

    if (message.role === 'user') {
      break
    }

    if (!message.pending || message.hidden) {
      continue
    }

    for (let j = message.parts.length - 1; j >= 0; j--) {
      const part = message.parts[j]!

      if (part.type === 'tool-call' && part.result === undefined && !part.completedAt && !isSilentTool(part.toolName)) {
        return part
      }
    }
  }
}

let previous: Readonly<Record<string, SidebarActivity>> = {}

/** The sidebar speaks stored ids; activity feeds speak runtime ids. Reuse the
 * dot's lineage bridge, keeping original item references so text deltas and
 * unrelated sessions never repaint rows. Todos/goals alone aren't liveness. */
export const $sidebarActivityById = computed(
  [$sessionStates, $statusItemsBySession, $sessions],
  (states, items, sessions) => {
    const next: Record<string, SidebarActivity> = {}

    for (const runtimeId of new Set([...Object.keys(states), ...Object.keys(items)])) {
      const state = states[runtimeId]

      const activity =
        liveTool(state) ??
        items[runtimeId]?.find(
          item => item.state === 'running' && (item.type === 'background' || item.type === 'subagent')
        )

      if (!activity) {
        continue
      }

      for (const alias of lineageAliases(state?.storedSessionId ?? runtimeId, sessions)) {
        // A live tool beats background work even during runtime handoff.
        if (next[alias]?.type !== 'tool-call') {
          next[alias] = activity
        }
      }
    }

    return (previous = stableRecord(previous, next))
  }
)
