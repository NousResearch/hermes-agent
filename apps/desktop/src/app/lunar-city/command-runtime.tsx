import { useStore } from '@nanostores/react'
import { useEffect, useMemo, useState } from 'react'

import { Button } from '@/components/ui/button'

import {
  type CommandExecutors,
  type CommandIntent,
  type CommandOperation,
  type CommandPlan,
  type CommandPlanningSnapshot,
  type CommandReceipt,
  type CommandTargetState,
  executeCommand,
  planCommand
} from './command-broker'
import { createLunarCityCommandExecutors } from './command-executors'
import { CommandConfirmation } from './components/command-confirmation'
import { EntityInspector, type EntityInspectorData, type InspectorSessionTarget } from './components/entity-inspector'
import type { EntityIdentity, EntityKey, LunarCitySnapshot, LunarEntity, SourceHealth } from './model'
import { $lunarCitySnapshot } from './store'

export interface LunarCityCommandRuntimeProps {
  executors?: CommandExecutors
  onOpenSession?: (target: InspectorSessionTarget) => void
  selectedEntityKey?: EntityKey
}

interface CommandChoice {
  intent: CommandIntent
  label: string
}

interface CommandLifetime {
  active: boolean
  generation: number
}

const READY_PATCH_SOURCE_STATES = new Set(['archived', 'blocked', 'done', 'review', 'scheduled', 'todo', 'triage'])

function sourceName(identity: EntityIdentity): string {
  if (identity.kind === 'profile') {
    return `fleet:${identity.connectionId}`
  }

  if (identity.kind === 'kanban') {
    return `kanban:${encodeURIComponent(identity.connectionId)}:${encodeURIComponent(identity.profile)}`
  }

  return `session:${identity.connectionId}`
}

function compatibleOperations(entity: LunarEntity): readonly CommandOperation[] {
  const identity = entity.identity

  if (identity.kind === 'session') {
    return ['open-session', 'send-guidance', 'interrupt-session']
  }

  if (identity.kind === 'subagent') {
    return ['open-session', 'send-guidance', 'interrupt-subagent']
  }

  if (identity.kind === 'kanban') {
    const running = entity.sourceState === 'running'

    if (!entity.sourceState) {
      return ['inspect-evidence']
    }

    return [
      'inspect-evidence',
      ...(running && identity.runId ? (['terminate-run'] as const) : []),
      ...(running ? (['reclaim-task'] as const) : []),
      ...(READY_PATCH_SOURCE_STATES.has(entity.sourceState) ? (['change-task-state'] as const) : [])
    ]
  }

  return []
}

function readbackCapabilities(identity: EntityIdentity): readonly CommandTargetState['readbackCapabilities'][number][] {
  if (identity.kind === 'session') {
    return ['session']
  }

  if (identity.kind === 'subagent') {
    return ['session', 'subagent']
  }

  if (identity.kind === 'kanban') {
    return ['kanban-task', ...(identity.runId ? (['kanban-run'] as const) : [])]
  }

  return []
}

function exactSource(entity: LunarEntity, city: LunarCitySnapshot): SourceHealth | undefined {
  const expected = sourceName(entity.identity)
  const matches = city.sources.filter(source => source.source === expected)

  return matches.length === 1 ? matches[0] : undefined
}

export function buildLunarCityCommandSnapshot(city: LunarCitySnapshot): CommandPlanningSnapshot {
  const targets = new Map<EntityKey, CommandTargetState>()

  for (const entity of city.entities.values()) {
    const source = exactSource(entity, city)

    if (!source) {
      continue
    }

    const owner = Object.freeze({ connectionId: entity.identity.connectionId, profile: entity.identity.profile })
    targets.set(entity.key, {
      availableOperations: compatibleOperations(entity),
      ...(entity.projectId ? { canonicalProjectId: entity.projectId } : {}),
      entity,
      observedState: {
        animation: entity.animation,
        authority: entity.authority,
        destination: entity.destination,
        observedAt: entity.observedAt,
        source: source.source,
        value:
          entity.identity.kind === 'kanban'
            ? (entity.sourceState ?? 'unknown')
            : entity.animation === 'work'
              ? 'running'
              : entity.animation
      },
      ownerCandidates: [owner],
      readbackCapabilities: readbackCapabilities(entity.identity),
      source,
      sourceOwner: owner
    })
  }

  return { city, targets }
}

function disabledWriteReason(entity: LunarEntity, source: SourceHealth): string | undefined {
  if (entity.authority === 'stale' || source.authority === 'stale') {
    return 'Refresh exact authoritative state before staging a write.'
  }

  if (entity.authority === 'partial' || source.authority === 'partial') {
    return 'Partial state cannot authorize a write.'
  }

  if (entity.authority !== 'authoritative' || source.authority !== 'authoritative') {
    return 'Unknown or unavailable state cannot authorize a write.'
  }

  return undefined
}

function inspectorData(entity: LunarEntity, source: SourceHealth): EntityInspectorData {
  const identity = entity.identity

  const owningSession =
    identity.kind === 'session' || identity.kind === 'subagent'
      ? {
          connectionId: identity.connectionId,
          profile: identity.profile,
          sessionId: identity.sessionId,
          storedSessionId: identity.sessionId
        }
      : undefined

  return {
    identity,
    ...(owningSession ? { owningSession } : {}),
    ...(entity.presentation ? { presentation: entity.presentation } : {}),
    ...(identity.kind === 'kanban'
      ? {
          ...(identity.runId ? { run: { id: identity.runId, status: entity.sourceState ?? 'unknown' } } : {}),
          task: { id: identity.taskId, state: entity.sourceState ?? 'unknown', title: `Task ${identity.taskId}` }
        }
      : {}),
    source
  }
}

function receiptText(receipt: CommandReceipt): string {
  const labels: Readonly<Record<CommandReceipt['verification'], string>> = {
    rejected: 'Rejected',
    timed_out: 'Timed out',
    verification_required: 'Verification required',
    verified: 'Verified'
  }

  return `${labels[receipt.verification]}${receipt.error ? `: ${receipt.error}` : ''}`
}

function disruptiveChoices(entity: LunarEntity): readonly CommandChoice[] {
  const identity = entity.identity

  if (identity.kind === 'session') {
    return [{ intent: { entityKey: entity.key, kind: 'interrupt-session' }, label: 'Interrupt session' }]
  }

  if (identity.kind === 'subagent') {
    return [{ intent: { entityKey: entity.key, kind: 'interrupt-subagent' }, label: 'Interrupt subagent' }]
  }

  if (identity.kind !== 'kanban') {
    return []
  }

  const running = entity.sourceState === 'running'
  const canMoveToReady = entity.sourceState ? READY_PATCH_SOURCE_STATES.has(entity.sourceState) : false

  if (!entity.sourceState) {
    return []
  }

  return [
    ...(running && identity.runId
      ? [{ intent: { entityKey: entity.key, kind: 'terminate-run' } as const, label: 'Terminate run' }]
      : []),
    ...(running
      ? [{ intent: { entityKey: entity.key, kind: 'reclaim-task' } as const, label: 'Reclaim task' }]
      : canMoveToReady
        ? [
            {
              intent: { entityKey: entity.key, kind: 'change-task-state', state: 'ready' } as const,
              label: 'Move task to ready'
            }
          ]
        : [])
  ]
}

export function LunarCityCommandRuntime({
  executors: suppliedExecutors,
  onOpenSession,
  selectedEntityKey
}: LunarCityCommandRuntimeProps) {
  const city = useStore($lunarCitySnapshot)
  const planning = useMemo(() => buildLunarCityCommandSnapshot(city), [city])
  const entity = selectedEntityKey ? city.entities.get(selectedEntityKey) : undefined
  const target = selectedEntityKey ? planning.targets.get(selectedEntityKey) : undefined
  const [guidance, setGuidance] = useState('')
  const [staged, setStaged] = useState<CommandPlan | undefined>()
  const [submitting, setSubmitting] = useState(false)
  const [status, setStatus] = useState<string | undefined>()
  const [evidence, setEvidence] = useState<unknown>()
  const [lifetime] = useState<CommandLifetime>(() => ({ active: true, generation: 0 }))

  const executors = useMemo(
    () => suppliedExecutors ?? createLunarCityCommandExecutors({ onOpenSession }),
    [onOpenSession, suppliedExecutors]
  )

  useEffect(() => {
    lifetime.active = true

    return () => {
      lifetime.active = false
      lifetime.generation += 1
    }
  }, [lifetime])

  useEffect(() => {
    if (submitting) {
      lifetime.generation += 1
      setSubmitting(false)
      setStatus('Command result ignored because the selection or source changed.')
    }

    if (staged) {
      setStaged(undefined)
      setStatus('Confirmation cancelled because the selection or source changed.')
    }

    setEvidence(undefined)
    // A staged plan is deliberately invalidated by either dependency. It is
    // excluded so staging itself does not immediately cancel the dialog.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [city.revision, lifetime, selectedEntityKey])

  if (!entity || !target) {
    return null
  }

  const writeDisabledReason = disabledWriteReason(entity, target.source)

  const run = async (plan: CommandPlan, confirmed: boolean): Promise<void> => {
    const generation = ++lifetime.generation
    setSubmitting(true)
    setStatus('Sending once and awaiting authoritative readback…')

    const receipt = await executeCommand(plan, executors, {
      confirmed,
      latestSnapshot: () => buildLunarCityCommandSnapshot($lunarCitySnapshot.get())
    })

    if (!lifetime.active || generation !== lifetime.generation) {
      return
    }

    setSubmitting(false)
    setStaged(undefined)

    if (plan.operation === 'inspect-evidence' && receipt.response !== undefined) {
      setEvidence(receipt.response)
    }

    setStatus(receiptText(receipt))
  }

  const stageOrRun = (intent: CommandIntent): void => {
    let plan: CommandPlan

    try {
      plan = planCommand(intent, buildLunarCityCommandSnapshot($lunarCitySnapshot.get()))
    } catch (error) {
      setStatus(error instanceof Error ? error.message : String(error))

      return
    }

    if (plan.confirmation) {
      setStaged(plan)

      return
    }

    void run(plan, false)
  }

  return (
    <section
      aria-label="Lunar City worker controls"
      className="pointer-events-auto absolute inset-y-20 left-3 z-40 w-[min(23rem,calc(100%-1.5rem))] overflow-y-auto rounded-xl border border-(--ui-stroke-tertiary) bg-background/95 p-3 shadow-xl backdrop-blur-xl sm:left-5"
    >
      <EntityInspector
        data={inspectorData(entity, target.source)}
        onInspectEvidence={
          entity.identity.kind === 'kanban'
            ? kind => stageOrRun({ entityKey: entity.key, evidence: kind, kind: 'inspect-evidence' })
            : undefined
        }
        onOpenSession={session => {
          if (onOpenSession) {
            stageOrRun({ entityKey: entity.key, kind: 'open-session' })
          }

          // The executor owns the canonical frozen route; this local value is
          // intentionally not dispatched directly.
          void session
        }}
      />

      {entity.identity.kind !== 'profile' ? (
        <div className="mt-3 space-y-2 border-t border-(--ui-stroke-tertiary) pt-3">
          {entity.identity.kind === 'kanban' ? (
            <Button
              onClick={() => stageOrRun({ entityKey: entity.key, evidence: 'task', kind: 'inspect-evidence' })}
              size="xs"
              variant="secondary"
            >
              Inspect evidence
            </Button>
          ) : null}

          {entity.identity.kind === 'session' || entity.identity.kind === 'subagent' ? (
            <>
              <label className="block text-xs" htmlFor="lunar-city-guidance">
                Guidance for selected entity
              </label>
              <textarea
                className="min-h-16 w-full rounded border border-(--ui-stroke-tertiary) bg-(--ui-bg-editor) p-2 text-xs"
                disabled={Boolean(writeDisabledReason) || submitting}
                id="lunar-city-guidance"
                onChange={event => setGuidance(event.target.value)}
                value={guidance}
              />
              <Button
                disabled={Boolean(writeDisabledReason) || submitting || !guidance.trim()}
                onClick={() => stageOrRun({ entityKey: entity.key, kind: 'send-guidance', text: guidance })}
                size="xs"
              >
                Send guidance
              </Button>
            </>
          ) : null}

          {disruptiveChoices(entity).map(choice => (
            <Button
              disabled={Boolean(writeDisabledReason) || submitting}
              key={choice.intent.kind}
              onClick={() => stageOrRun(choice.intent)}
              size="xs"
              variant="destructive"
            >
              {choice.label}
            </Button>
          ))}

          {writeDisabledReason ? <p className="text-xs text-(--ui-orange)">{writeDisabledReason}</p> : null}
        </div>
      ) : null}

      {status ? (
        <p aria-live="polite" className="mt-3 text-xs" role="status">
          {status}
        </p>
      ) : null}

      {evidence !== undefined ? (
        <pre aria-label="Exact source evidence" className="mt-3 max-h-48 overflow-auto text-xs">
          {JSON.stringify(evidence, null, 2)}
        </pre>
      ) : null}

      {staged ? (
        <CommandConfirmation
          getLatestSnapshot={() => buildLunarCityCommandSnapshot($lunarCitySnapshot.get())}
          onCancel={() => setStaged(undefined)}
          onConfirm={plan => void run(plan, true)}
          open
          plan={staged}
          submitting={submitting}
        />
      ) : null}
    </section>
  )
}
