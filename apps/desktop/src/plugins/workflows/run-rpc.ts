// Everything the canvas can tell the gateway about a run, named once.
//
// The player is a state machine over the event log; these are the only calls it
// makes to move the run underneath it. They deliberately don't swallow their own
// failures — whether a call is fire-and-forget or has to surface is the caller's
// decision, and it reads better at the call site than it would here.

import { host } from '@hermes/plugin-sdk'

import { $workflows } from './documents'
import type { RunPlan } from './graph'
import type { ProtoEvent } from './protocol'

/** Statuses in which the gateway still owns the run — anything else is over. */
export const LIVE: ReadonlySet<string> = new Set(['running', 'paused', 'waiting_human', 'waiting_world'])

export const startRun = (plan: RunPlan, source: 'manual' | 'webhook', payload?: Record<string, unknown>) =>
  host.request<{ runId: string }>('workflow.run.start', {
    payload,
    scenario: plan.scenario,
    source,
    workflowId: plan.id
  })

/** The whole log so far — the one-shot catch-up before the bus takes over. */
export const runEvents = (runId: string) =>
  host.request<{ events?: ProtoEvent[] }>('workflow.run.events', { after: -1, runId })

/** A run this workflow left going, from a previous mount or another surface. */
export const activeRun = (workflowId: string) =>
  host.request<{ events?: ProtoEvent[]; run?: { status: string }; runId?: string }>('workflow.run.active', {
    workflowId
  })

export const cancelRun = (runId: string) => host.request('workflow.run.cancel', { runId })

export const respondRun = (runId: string, nodeId: string, decision: 'approved' | 'denied', by: string) =>
  host.request('workflow.run.respond', { by, decision, nodeId, runId })

export const pauseRun = (runId: string) => host.request<{ status?: string }>('workflow.run.pause', { runId })

export const resumeRun = (runId: string) => host.request('workflow.run.resume', { runId })

/** Hermes asks in the canvas thread — missing model, a 404, something the
 *  step cannot invent. Hidden so it isn't typed as a user bubble. */
export function askInCanvas(workflowId: string, nodeId: string, prompt: string): void {
  const stored = $workflows.get().find(d => d.id === workflowId)?.sessionId

  if (!stored) {
    return
  }

  void host
    .request<{ session_id?: string }>('session.resume', { session_id: stored })
    .then(res => {
      const sid = res.session_id

      if (!sid) {
        return
      }

      return host.request('prompt.submit', {
        display_kind: 'hidden',
        session_id: sid,
        text:
          `The "${nodeId}" step on this workflow could not run:\n\n${prompt}\n\n` +
          'Explain that in this thread. If something is missing or confusing ' +
          '(a model, a Figma board, credentials, a path), ask me — use clarify ' +
          'for a short choice list. Do not pretend the step succeeded.'
      })
    })
    .catch(() => {})
}
