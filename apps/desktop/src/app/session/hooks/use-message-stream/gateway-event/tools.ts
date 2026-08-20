import { invalidateSlashCompletions } from '@/lib/slash-completion-cache'
import { refreshBackgroundProcesses } from '@/store/composer-status'
import { flashPetActivity, setPetActivity } from '@/store/pet'
import { pruneDelegateFallbackSubagents, upsertSubagent } from '@/store/subagents'
import { reportMcpToolResult } from '@/store/suggestion-providers/repair'
import { invalidateSkillSuggestionIndex } from '@/store/suggestion-providers/skill'
import { recordToolDiff } from '@/store/tool-diffs'
import { setSessionDraftingTool } from '@/store/tool-drafting'
import { notifyWorkspaceChanged, toolChangedPath, toolMayMutateFiles } from '@/store/workspace-events'

import { SUBAGENT_EVENT_TYPES, toTodoPayload } from '../utils'

import type { GatewayEventContext } from './types'

/** tool.generating / tool.start / tool.progress / tool.complete / subagent.*. */
export function handleToolEvent(ctx: GatewayEventContext): boolean {
  const { deps, event, payload, sessionId, isActiveEvent, occurredAt } = ctx
  const { flushQueuedDeltas, nativeSubagentSessionsRef, sessionInterrupted, updateSessionState, upsertToolCall } = deps

  if (event.type === 'tool.generating') {
    // Announced while the model is still emitting the call's JSON, so it
    // carries a name and nothing else — no id, no args. Materializing a row
    // from it strands an argless placeholder whenever the bubble is sealed
    // before the real `tool.start` arrives, because the two can no longer be
    // reconciled across the boundary. It's a status, so say it as one.
    // A stopped turn can still emit a frame or two before the backend
    // notices, and naming a tool we will never run leaves the label up
    // until something else retires it. `mutateStream` drops late tool rows
    // on the same condition; the status line has to agree with it.
    if (!sessionId || sessionInterrupted(sessionId)) {
      return true
    }

    setSessionDraftingTool(sessionId, typeof payload?.name === 'string' ? payload.name : '')

    if (isActiveEvent) {
      setPetActivity({ reasoning: false, toolRunning: true })
    }

    return true
  }

  if (event.type === 'tool.start' || event.type === 'tool.progress') {
    if (!sessionId) {
      return true
    }

    flushQueuedDeltas(sessionId)
    upsertToolCall(sessionId, toTodoPayload(payload) ?? payload, 'running', event.type, occurredAt)

    if (isActiveEvent) {
      setPetActivity({ reasoning: false, toolRunning: true })
    }

    return true
  }

  if (event.type === 'tool.complete') {
    if (sessionId) {
      flushQueuedDeltas(sessionId)
      upsertToolCall(sessionId, toTodoPayload(payload) ?? payload, 'complete', event.type, occurredAt)

      if (isActiveEvent) {
        setPetActivity({ toolRunning: false })

        // A tool can fail without ending the turn when the agent recovers
        // and continues. Surface that failure as a short pet beat too;
        // otherwise only turn-level errors ever reach the failed state.
        if (payload?.error) {
          flashPetActivity({ error: true })
        }
      }

      // A pending clarify blocks the turn, so the first tool.complete after
      // one is the clarify resolving — drop the "needs input" flag here so
      // the sidebar indicator clears as soon as it's answered, not only at
      // message.complete.
      updateSessionState(sessionId, state => (state.needsInput ? { ...state, needsInput: false } : state))

      // terminal/process tool calls are the only things that spawn or reap
      // background processes — sync the composer status stack right after.
      if (!sessionInterrupted(sessionId) && (payload?.name === 'terminal' || payload?.name === 'process')) {
        void refreshBackgroundProcesses(sessionId)
      }
    }

    // The agent just created/deleted/renamed a skill, which adds or removes
    // its `/name` command. Drop the composer's cached `/` list so the new
    // skill is offerable now rather than after the hour-long TTL — and the
    // skill-suggestion provider's index with it.
    if (payload?.name === 'skill_manage') {
      invalidateSlashCompletions()
      invalidateSkillSuggestionIndex()
    }

    // MCP tool outcomes feed the connection-repair suggestion provider:
    // an auth/connection-shaped failure offers a reconnect pill; a later
    // success against the same server withdraws it.
    if (sessionId && typeof payload?.name === 'string' && payload.name.startsWith('mcp__')) {
      reportMcpToolResult(
        sessionId,
        payload.name,
        Boolean(payload.error),
        [payload.error, payload.result].filter(part => typeof part === 'string').join(' ')
      )
    }

    if (typeof payload?.inline_diff === 'string' && payload.inline_diff.trim()) {
      recordToolDiff(payload.tool_id || payload.name || '', payload.inline_diff)
    }

    // A file-mutating tool just finished — nudge the git-mirroring surfaces
    // (coding rail, review pane, file tree) to refresh. Event-driven, not
    // polled: fires exactly when the agent touches the tree.
    if (payload && toolMayMutateFiles(payload)) {
      notifyWorkspaceChanged(toolChangedPath(payload))
    }

    return true
  }

  if (SUBAGENT_EVENT_TYPES.has(event.type)) {
    if (sessionId && payload && !sessionInterrupted(sessionId)) {
      if (!nativeSubagentSessionsRef.current.has(sessionId)) {
        pruneDelegateFallbackSubagents(sessionId)
      }

      nativeSubagentSessionsRef.current.add(sessionId)
      upsertSubagent(
        sessionId,
        payload as Record<string, unknown>,
        event.type === 'subagent.spawn_requested' || event.type === 'subagent.start',
        event.type
      )
    }

    return true
  }

  return false
}
