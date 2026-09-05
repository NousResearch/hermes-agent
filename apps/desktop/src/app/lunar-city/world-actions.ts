import type { WorldActionKind } from './world-events'

export interface WorldActionContext {
  kanban: {
    addComment: (id: string, body: string) => Promise<unknown>
    createTask: (body: Record<string, unknown>) => Promise<unknown>
    patchTask: (id: string, patch: Record<string, unknown>) => Promise<unknown>
    reclaimTask: (id: string) => Promise<unknown>
    reassignTask: (id: string, profile: string) => Promise<unknown>
  }
  createSession?: (params: Record<string, unknown>) => Promise<unknown> | unknown
  sendDialogue?: (params: { message: string; sessionId?: string; targetProfile?: string }) => Promise<unknown>
  sendVoice?: (params: { audioId?: string; transcript?: string; sessionId?: string; targetProfile?: string }) => Promise<unknown>
  respondApproval?: (params: { actionId: string; approved: boolean; sessionId?: string }) => Promise<unknown>
  requestApproval?: (params: Record<string, unknown>) => Promise<unknown>
}

export interface WorldActionContextTarget {
  board?: string
  eventId?: string
  prId?: string
  taskId?: string
}

export type WorldActionIntent =
  | { kind: 'inspect' | 'inspect_blocker' | 'show_source'; target: WorldActionContextTarget }
  | { kind: 'approval_response'; actionId: string; approved: boolean; sessionId?: string }
  | { body: string; kind: 'comment'; taskId: string }
  | { kind: 'create_session'; params: Record<string, unknown> }
  | { kind: 'create_task'; body: Record<string, unknown> }
  | { kind: 'dialogue_send'; message: string; sessionId?: string; targetProfile?: string }
  | { kind: 'recover_task'; patch: Record<string, unknown>; taskId: string }
  | { kind: 'reassign_task'; profile: string; taskId: string }
  | { kind: 'reclaim_task'; taskId: string }
  | { kind: 'request_approval'; params: Record<string, unknown> }
  | { kind: 'voice_send'; audioId?: string; transcript?: string; sessionId?: string; targetProfile?: string }

export type WorldActionResult =
  | { kind: 'completed'; ok: true; value?: unknown }
  | { kind: 'approval_required' | 'disconnected' | 'rejected'; message: string; ok: false }

export interface WorldActionRunner {
  run: (intent: WorldActionIntent) => Promise<WorldActionResult>
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error)
}

function failureKind(message: string): 'approval_required' | 'disconnected' | 'rejected' {
  const normalized = message.toLowerCase()

  if (normalized.includes('approval') || normalized.includes('forbidden') || normalized.includes('403')) {
    return 'approval_required'
  }

  if (normalized.includes('disconnect') || normalized.includes('network') || normalized.includes('closed')) {
    return 'disconnected'
  }

  return 'rejected'
}

function completed(value?: unknown): WorldActionResult {
  return { ok: true, kind: 'completed', ...(value === undefined ? {} : { value }) }
}

function failed(error: unknown): WorldActionResult {
  const message = errorMessage(error)

  return { kind: failureKind(message), message, ok: false }
}

export function createWorldActionRunner(context: WorldActionContext): WorldActionRunner {
  return {
    async run(intent) {
      try {
        switch (intent.kind) {
          case 'inspect':

          case 'inspect_blocker':

          case 'show_source':
            return completed(intent.target)

          case 'comment':
            return completed(await context.kanban.addComment(intent.taskId, intent.body))

          case 'recover_task':
            return completed(await context.kanban.patchTask(intent.taskId, intent.patch))

          case 'reassign_task':
            return completed(await context.kanban.reassignTask(intent.taskId, intent.profile))

          case 'reclaim_task':
            return completed(await context.kanban.reclaimTask(intent.taskId))

          case 'create_task':
            return completed(await context.kanban.createTask(intent.body))

          case 'create_session':
            return context.createSession
              ? completed(await context.createSession(intent.params))
              : failed(new Error('New sessions are unavailable'))

          case 'dialogue_send':
            return context.sendDialogue
              ? completed(
                  await context.sendDialogue({
                    message: intent.message,
                    sessionId: intent.sessionId,
                    targetProfile: intent.targetProfile
                  })
                )
              : failed(new Error('Dialogue is unavailable'))

          case 'voice_send':
            return context.sendVoice
              ? completed(
                  await context.sendVoice({
                    audioId: intent.audioId,
                    sessionId: intent.sessionId,
                    targetProfile: intent.targetProfile,
                    transcript: intent.transcript
                  })
                )
              : failed(new Error('Voice is unavailable'))

          case 'approval_response':
            return context.respondApproval
              ? completed(
                  await context.respondApproval({
                    actionId: intent.actionId,
                    approved: intent.approved,
                    sessionId: intent.sessionId
                  })
                )
              : failed(new Error('Approval response is unavailable'))

          case 'request_approval':
            return context.requestApproval
              ? completed(await context.requestApproval(intent.params))
              : failed(new Error('Approval is unavailable'))
        }
      } catch (error) {
        return failed(error)
      }
    }
  }
}

export type { WorldActionKind }
