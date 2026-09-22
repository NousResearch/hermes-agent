import { pluginRest, type PluginSourceScope } from '@/api/plugins'
import { requestForSessionProfile } from '@/store/session-request-router'
import { $sessionStates, $sessionTiles } from '@/store/session-states'

import {
  type CommandCurrentAuthority,
  type CommandExecutor,
  type CommandExecutors,
  type CommandPlan,
  type CommandReadback,
  CommandRejectedError
} from './command-broker'
import type { InspectorSessionTarget } from './components/entity-inspector'

const KANBAN_LOG_TAIL_BYTES = 65_536
const LIVE_SUBAGENT_STATUSES = new Set(['queued', 'running'])

const KANBAN_NON_RUNNING_TASK_STATES = new Set([
  'archived',
  'blocked',
  'done',
  'ready',
  'review',
  'scheduled',
  'todo',
  'triage'
])

export interface LunarCityCommandExecutorOptions {
  onOpenSession?: (target: InspectorSessionTarget) => void
  resolveLiveRuntime?: (
    owner: Readonly<{ connectionId: string; profile: string }>,
    storedSessionId: string
  ) => string | undefined
}

function rejectAmbientRequest<T>(): Promise<T> {
  return Promise.reject(new Error('Lunar City commands require an exact owner route'))
}

function record(value: unknown): Record<string, unknown> {
  return value && typeof value === 'object' && !Array.isArray(value) ? (value as Record<string, unknown>) : {}
}

function text(value: unknown): string | undefined {
  return typeof value === 'string' && value.trim() ? value.trim() : undefined
}

function identifier(value: unknown): string | undefined {
  if (typeof value === 'number' && Number.isSafeInteger(value)) {
    return String(value)
  }

  return text(value)
}

function bool(value: unknown): boolean | undefined {
  return typeof value === 'boolean' ? value : undefined
}

function statusCode(error: unknown): number | undefined {
  const value = record(error)
  const response = record(value.response)
  const candidate = value.status ?? value.statusCode ?? response.status

  return typeof candidate === 'number' && Number.isSafeInteger(candidate) ? candidate : undefined
}

async function kanbanMutation<T>(request: () => Promise<T>): Promise<T> {
  try {
    return await request()
  } catch (error) {
    const status = statusCode(error)

    if (status !== undefined && status >= 400 && status < 500) {
      const message = text(record(error).message) ?? text(record(record(error).response).message)

      throw new CommandRejectedError(message ?? `Kanban command rejected with status ${status}.`)
    }

    throw error
  }
}

function exactScope(plan: CommandPlan): PluginSourceScope {
  return Object.freeze({ connectionId: plan.owner.connectionId, profile: plan.owner.profile })
}

function withBoard(path: string, board: string, extra: Readonly<Record<string, string>> = {}): string {
  const query = new URLSearchParams({ board, ...extra })

  return `${path}?${query.toString()}`
}

function localReadback(plan: CommandPlan): CommandReadback {
  return {
    authority: 'authoritative',
    effect: plan.readback.expectedEffect,
    identity: plan.identity,
    observedAt: plan.plannedAt,
    operation: plan.operation,
    outcome: 'verified',
    owner: plan.owner,
    receipt: { authority: 'authoritative', planDigest: plan.digest },
    revision: plan.plannedRevision
  }
}

function defaultLiveRuntime(
  owner: Readonly<{ connectionId: string; profile: string }>,
  storedSessionId: string
): string | undefined {
  const matches = Object.entries($sessionStates.get())
    .filter(([, state]) => state?.storedSessionId === storedSessionId)
    .filter(([runtimeId, state]) => {
      const provenance = state?.transcriptProvenance

      if (provenance) {
        return (
          provenance.connectionId === owner.connectionId &&
          provenance.profile === owner.profile &&
          provenance.storedSessionId === storedSessionId
        )
      }

      return $sessionTiles.get().some(tile => {
        const route = tile.ownerRoute

        return (
          tile.runtimeId === runtimeId &&
          tile.storedSessionId === storedSessionId &&
          route?.connectionId === owner.connectionId &&
          route.profile === owner.profile
        )
      })
    })
    .map(([runtimeId]) => runtimeId.trim())
    .filter(Boolean)

  return matches.length === 1 ? matches[0] : undefined
}

function sessionExecutor(options: LunarCityCommandExecutorOptions): CommandExecutor {
  const localReceipts = new Set<string>()
  const subagentGenerations = new Map<string, string>()
  const resolveLiveRuntime = options.resolveLiveRuntime ?? defaultLiveRuntime

  return {
    async currentAuthority(plan) {
      const identity = plan.identity

      if (identity.kind !== 'session' && identity.kind !== 'subagent') {
        return null
      }

      const liveRuntimeId = resolveLiveRuntime(plan.owner, identity.sessionId)

      if (!liveRuntimeId) {
        return null
      }

      const response = await requestForSessionProfile<unknown>(plan.owner, rejectAmbientRequest, 'session.list', {
        include_hidden: true,
        limit: 200
      })

      const sessions = Array.isArray(record(response).sessions) ? (record(response).sessions as unknown[]) : []
      const matches = sessions.filter(row => identifier(record(row).id) === identity.sessionId)

      if (matches.length !== 1) {
        return null
      }

      if (identity.kind === 'subagent') {
        const statusResponse = await requestForSessionProfile<unknown>(
          plan.owner,
          rejectAmbientRequest,
          'subagent.status',
          { session_id: liveRuntimeId, subagent_id: identity.subagentId }
        )

        const statusEnvelope = record(statusResponse)
        const subagent = record(statusEnvelope.subagent)
        const generation = text(subagent.generation)

        if (
          statusEnvelope.found !== true ||
          identifier(subagent.subagent_id) !== identity.subagentId ||
          !generation ||
          !LIVE_SUBAGENT_STATUSES.has(text(subagent.status) ?? '')
        ) {
          return null
        }

        subagentGenerations.set(plan.digest, generation)

        if (subagentGenerations.size > 64) {
          const oldestDigest = subagentGenerations.keys().next().value

          if (oldestDigest) {
            subagentGenerations.delete(oldestDigest)
          }
        }
      }

      return currentAuthority(plan)
    },
    async readback(plan) {
      if (localReceipts.delete(plan.digest)) {
        return localReadback(plan)
      }

      // Gateway ACKs for steering and interrupts do not prove their eventual
      // effect. Until a causally tagged session event is available, fail
      // closed as verification_required rather than inventing completion.
      return null
    },
    async send(plan) {
      if (plan.method === 'session.open') {
        const identity = plan.identity

        if (identity.kind !== 'session' && identity.kind !== 'subagent') {
          throw new CommandRejectedError('Exact session identity is unavailable.')
        }

        if (!options.onOpenSession) {
          throw new CommandRejectedError('Open session is unavailable from this route.')
        }

        options.onOpenSession(
          Object.freeze({
            connectionId: plan.owner.connectionId,
            profile: plan.owner.profile,
            sessionId: identity.sessionId,
            storedSessionId: identity.sessionId
          })
        )
        localReceipts.add(plan.digest)

        return { opened: true }
      }

      if (plan.method === 'evidence.inspect') {
        throw new Error('Authoritative session evidence is unavailable from the Lunar City route.')
      }

      const storedSessionId = text(plan.params.session_id)
      const liveRuntimeId = storedSessionId ? resolveLiveRuntime(plan.owner, storedSessionId) : undefined

      if (!storedSessionId || !liveRuntimeId) {
        throw new Error('The exact-owner live runtime is unavailable; no session command was sent.')
      }

      const identity = plan.identity
      const expectedGeneration = identity.kind === 'subagent' ? subagentGenerations.get(plan.digest) : undefined

      if (identity.kind === 'subagent') {
        subagentGenerations.delete(plan.digest)

        if (!expectedGeneration) {
          throw new Error('The exact live subagent generation is unavailable; no subagent command was sent.')
        }
      }

      const response = await requestForSessionProfile<unknown>(plan.owner, rejectAmbientRequest, plan.method, {
        ...plan.params,
        ...(expectedGeneration ? { expected_generation: expectedGeneration } : {}),
        session_id: liveRuntimeId
      })

      const status = text(record(response).status)

      if (status === 'rejected' || bool(record(response).ok) === false) {
        throw new CommandRejectedError(text(record(response).error) ?? 'The exact session owner rejected the command.')
      }

      return response
    }
  }
}

function currentAuthority(plan: CommandPlan): CommandCurrentAuthority {
  return {
    authority: 'authoritative',
    identity: plan.identity,
    observedAt: Date.now(),
    owner: plan.owner
  }
}

function exactTask(plan: CommandPlan, response: unknown): Record<string, unknown> | undefined {
  if (plan.identity.kind !== 'kanban') {
    return undefined
  }

  const task = record(record(response).task)

  return identifier(task.id) === plan.identity.taskId ? task : undefined
}

function exactTaskLog(plan: CommandPlan, response: unknown): Record<string, unknown> | undefined {
  if (plan.identity.kind !== 'kanban') {
    return undefined
  }

  const log = record(response)
  const content = log.content
  const sizeBytes = log.size_bytes

  if (
    identifier(log.task_id) !== plan.identity.taskId ||
    typeof log.exists !== 'boolean' ||
    typeof log.path !== 'string' ||
    typeof content !== 'string' ||
    typeof log.truncated !== 'boolean' ||
    typeof sizeBytes !== 'number' ||
    !Number.isSafeInteger(sizeBytes) ||
    sizeBytes < 0 ||
    new TextEncoder().encode(content).byteLength > KANBAN_LOG_TAIL_BYTES
  ) {
    return undefined
  }

  return log
}

function taskEffectVerified(plan: CommandPlan, task: Record<string, unknown>): boolean {
  const expected = plan.readback.expectedEffect

  if (expected.kind === 'task-reclaimed') {
    return (
      typeof task.status === 'string' && KANBAN_NON_RUNNING_TASK_STATES.has(task.status) && task.current_run_id === null
    )
  }

  if (expected.kind === 'task-reassigned') {
    return text(task.assignee) === expected.value
  }

  if (expected.kind === 'task-state-changed') {
    return text(task.status) === expected.value
  }

  return expected.kind === 'evidence-present'
}

function kanbanTaskExecutor(): CommandExecutor {
  return {
    async currentAuthority(plan) {
      if (plan.identity.kind !== 'kanban') {
        return null
      }

      const response = await pluginRest<unknown>(
        'kanban',
        withBoard(`/tasks/${encodeURIComponent(plan.identity.taskId)}`, plan.identity.board),
        { method: 'GET', scope: exactScope(plan) }
      )

      return exactTask(plan, response) ? currentAuthority(plan) : null
    },
    async readback(plan) {
      if (plan.operation === 'inspect-evidence') {
        return null
      }

      if (plan.identity.kind !== 'kanban') {
        return null
      }

      const response = await pluginRest<unknown>(
        'kanban',
        withBoard(`/tasks/${encodeURIComponent(plan.identity.taskId)}`, plan.identity.board),
        { method: 'GET', scope: exactScope(plan) }
      )

      const task = exactTask(plan, response)

      if (!task) {
        return null
      }

      const verified = taskEffectVerified(plan, task)

      return {
        authority: 'authoritative',
        ...(verified ? { effect: plan.readback.expectedEffect } : {}),
        identity: plan.identity,
        observedAt: Date.now(),
        operation: plan.operation,
        outcome: verified ? 'verified' : 'unknown',
        owner: plan.owner,
        receipt: { authority: 'authoritative', planDigest: plan.digest },
        revision: plan.plannedRevision
      }
    },
    async send(plan) {
      if (plan.identity.kind !== 'kanban') {
        throw new CommandRejectedError('Exact Kanban identity is unavailable.')
      }

      const base = `/tasks/${encodeURIComponent(plan.identity.taskId)}`
      const board = plan.identity.board

      if (plan.operation === 'inspect-evidence') {
        const logEvidence = plan.intent.kind === 'inspect-evidence' && plan.intent.evidence === 'log'
        const path = logEvidence ? `${base}/log` : base

        const response = await pluginRest(
          'kanban',
          withBoard(path, board, logEvidence ? { tail: String(KANBAN_LOG_TAIL_BYTES) } : {}),
          {
            method: 'GET',
            scope: exactScope(plan)
          }
        )

        if (logEvidence ? !exactTaskLog(plan, response) : !exactTask(plan, response)) {
          throw new Error(
            logEvidence
              ? 'Authoritative exact worker-log evidence is unavailable from the Kanban source.'
              : 'Authoritative exact-task evidence is unavailable from the Kanban source.'
          )
        }

        return response
      }

      let path: string
      let body: Record<string, unknown>

      if (plan.operation === 'reclaim-task') {
        path = `${base}/reclaim`
        body = { reason: 'Operator confirmed in Lunar City' }
      } else if (plan.operation === 'reassign-task') {
        path = `${base}/reassign`
        body = {
          profile: plan.params.assignee,
          reclaim_first: true,
          reason: 'Operator confirmed in Lunar City'
        }
      } else if (plan.operation === 'change-task-state') {
        path = base
        body = { status: plan.params.status }
      } else {
        throw new CommandRejectedError(`${plan.operation} has no exact Kanban task endpoint.`)
      }

      return kanbanMutation(() =>
        pluginRest('kanban', withBoard(path, board), {
          body,
          method: plan.method === 'kanban.task.patch' ? 'PATCH' : 'POST',
          scope: exactScope(plan)
        })
      )
    }
  }
}

function kanbanRunExecutor(): CommandExecutor {
  return {
    async currentAuthority(plan) {
      if (plan.identity.kind !== 'kanban' || !plan.identity.runId) {
        return null
      }

      const response = await pluginRest<unknown>(
        'kanban',
        withBoard(`/runs/${encodeURIComponent(plan.identity.runId)}`, plan.identity.board),
        { method: 'GET', scope: exactScope(plan) }
      )

      const run = record(record(response).run)

      return identifier(run.id) === plan.identity.runId && identifier(run.task_id) === plan.identity.taskId
        ? currentAuthority(plan)
        : null
    },
    async readback(plan) {
      if (plan.identity.kind !== 'kanban' || !plan.identity.runId) {
        return null
      }

      if (plan.operation === 'inspect-evidence') {
        return null
      }

      const response = await pluginRest<unknown>(
        'kanban',
        withBoard(`/runs/${encodeURIComponent(plan.identity.runId)}`, plan.identity.board),
        { method: 'GET', scope: exactScope(plan) }
      )

      const run = record(record(response).run)
      const endedAt = run.ended_at

      const ended =
        identifier(run.id) === plan.identity.runId &&
        identifier(run.task_id) === plan.identity.taskId &&
        ((typeof endedAt === 'number' && Number.isFinite(endedAt)) || Boolean(text(endedAt))) &&
        run.outcome === 'reclaimed'

      return {
        authority: 'authoritative',
        ...(ended ? { effect: plan.readback.expectedEffect } : {}),
        identity: plan.identity,
        observedAt: Date.now(),
        operation: plan.operation,
        outcome: ended ? 'verified' : 'unknown',
        owner: plan.owner,
        receipt: { authority: 'authoritative', planDigest: plan.digest },
        revision: plan.plannedRevision
      }
    },
    async send(plan) {
      if (plan.identity.kind !== 'kanban' || !plan.identity.runId) {
        throw new CommandRejectedError('Exact Kanban run identity is unavailable.')
      }

      const base = `/runs/${encodeURIComponent(plan.identity.runId)}`
      const board = plan.identity.board

      if (plan.operation === 'inspect-evidence') {
        const response = await pluginRest('kanban', withBoard(base, board), {
          method: 'GET',
          scope: exactScope(plan)
        })

        const run = record(record(response).run)

        if (identifier(run.id) !== plan.identity.runId || identifier(run.task_id) !== plan.identity.taskId) {
          throw new Error('Authoritative exact-run evidence is unavailable from the Kanban source.')
        }

        return response
      }

      return kanbanMutation(() =>
        pluginRest('kanban', withBoard(`${base}/terminate`, board), {
          body: { reason: 'Operator confirmed in Lunar City' },
          method: 'POST',
          scope: exactScope(plan)
        })
      )
    }
  }
}

export function createLunarCityCommandExecutors(options: LunarCityCommandExecutorOptions = {}): CommandExecutors {
  const sessions = sessionExecutor(options)

  return {
    kanbanRun: kanbanRunExecutor(),
    kanbanTask: kanbanTaskExecutor(),
    session: sessions,
    subagent: sessions
  }
}
