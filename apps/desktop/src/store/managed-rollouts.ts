import { atom } from 'nanostores'

import {
  type PlanChange,
  type RolloutCapabilities,
  type RolloutCommand,
  type RolloutDraft,
  type RolloutPlan,
  type RolloutSnapshot,
  validateRolloutCapabilities,
  validateRolloutCommand,
  validateRolloutPlan,
  validateRolloutSnapshot,
  validateRolloutTarget
} from '@/lib/managed-rollout-contract'

export type ManagedRolloutSnapshot = RolloutSnapshot

export interface ManagedRolloutInventoryObservation {
  installId: string
  connectionId: string
  aliasConnectionIds: string[]
  codeRoot: string
  repositoryId: string
  headSha: string
  requiredScopeIds: string[]
  source: {
    connectionId: string
    verifiedHostKeyFingerprint: string
  }
}

export interface ManagedRolloutInventory {
  inventoryRevision: string
  capturedMono: number
  observations: ManagedRolloutInventoryObservation[]
}

export interface ManagedRolloutResolution {
  resolutionId: string
  expiresAt: number
  inventoryRevision: string
  plan: RolloutPlan
}

export interface ManagedRolloutPreflight {
  ok: boolean
  token: string | null
  requestId: string | null
  rolloutId: string | null
  expiresAt: number | null
  planDigest: string | null
  canonicalPlan: RolloutPlan
  changes: PlanChange[]
  blockers: string[]
}

export interface ManagedRolloutHistoryEntry {
  id: string
  revision: number
  phase: string
  updatedAt: string
  unresolvedInstallIds: string[]
  archived: boolean
}

export interface ManagedRolloutsState {
  status: 'idle' | 'loading' | 'ready' | 'reconnecting' | 'unsupported' | 'error'
  revision: number | null
  snapshot: ManagedRolloutSnapshot | null
  error: string | null
  active: boolean
  capabilities: RolloutCapabilities | null
}

export interface RolloutResponse {
  revision: number
  snapshot: ManagedRolloutSnapshot | null
}

export interface ManagedRolloutsBridge {
  capabilities: () => Promise<{ available: boolean; reason: string | null; maxConcurrency?: number; maxInstallations?: number }>
  inventory?: () => Promise<unknown>
  resolveTarget?: (request: { connectionIds: string[]; inventoryRevision: string; retryOf: string | null }) => Promise<unknown>
  preflight?: (draft: RolloutDraft) => Promise<unknown>
  start?: (request: { token: string; requestId: string }) => Promise<unknown>
  activeRevision: () => Promise<number | null>
  read: (sinceRevision: number | null) => Promise<RolloutResponse>
  get?: (id: string) => Promise<unknown | null>
  command: (payload: Record<string, unknown>) => Promise<unknown>
  history?: (page: { cursor?: string; limit: number }) => Promise<unknown>
}

const initial: ManagedRolloutsState = {
  status: 'idle',
  revision: null,
  snapshot: null,
  error: null,
  active: false,
  capabilities: null
}

export const $managedRollouts = atom<ManagedRolloutsState>(initial)

let bridgeOverride: ManagedRolloutsBridge | null = null
let pollTimer: ReturnType<typeof setTimeout> | undefined
let pollGeneration = 0
let pollInFlight: Promise<void> | null = null
let commandInFlight: { key: string; promise: Promise<unknown> } | null = null
let pollingActive = false
let pollingInterval = 5_000
let retryDelay = 1_000

function canonical(value: unknown): string {
  if (Array.isArray(value)) {return `[${value.map(canonical).join(',')}]`}

  if (value && typeof value === 'object') {
    return `{${Object.keys(value as Record<string, unknown>).sort().map(key => `${JSON.stringify(key)}:${canonical((value as Record<string, unknown>)[key])}`).join(',')}}`
  }

  return JSON.stringify(value)
}

function record(value: unknown, name: string): Record<string, unknown> {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {throw new Error(`managed-rollouts-invalid-${name}`)}

  return value as Record<string, unknown>
}

function string(value: unknown, name: string): string {
  if (typeof value !== 'string' || !value || value.length > 4096) {throw new Error(`managed-rollouts-invalid-${name}`)}

  return value
}

function stringList(value: unknown, name: string): string[] {
  if (!Array.isArray(value) || value.length > 500 || value.some(item => typeof item !== 'string' || !item)) {
    throw new Error(`managed-rollouts-invalid-${name}`)
  }

  return value as string[]
}

function parseInventory(value: unknown): ManagedRolloutInventory {
  const input = record(value, 'inventory')
  const observations = input.observations

  if (!Array.isArray(observations) || observations.length > 500) {throw new Error('managed-rollouts-invalid-inventory')}

  const parsed = observations.map(value => {
    const item = record(value, 'inventory-row')
    const source = record(item.source, 'inventory-source')

    return {
      installId: string(item.installId, 'install-id'),
      connectionId: string(item.connectionId, 'connection-id'),
      aliasConnectionIds: stringList(item.aliasConnectionIds, 'alias-ids'),
      codeRoot: string(item.codeRoot, 'code-root'),
      repositoryId: string(item.repositoryId, 'repository-id'),
      headSha: string(item.headSha, 'head-sha'),
      requiredScopeIds: stringList(item.requiredScopeIds, 'scope-ids'),
      source: {
        connectionId: string(source.connectionId, 'source-connection-id'),
        verifiedHostKeyFingerprint: string(source.verifiedHostKeyFingerprint, 'host-key-fingerprint')
      }
    }
  })

  if (new Set(parsed.map(row => row.installId)).size !== parsed.length) {throw new Error('managed-rollouts-duplicate-installation')}

  if (!Number.isFinite(input.capturedMono) || (input.capturedMono as number) < 0) {throw new Error('managed-rollouts-invalid-capture-time')}

  return {
    inventoryRevision: string(input.inventoryRevision, 'inventory-revision'),
    capturedMono: input.capturedMono as number,
    observations: parsed
  }
}

function parseResolution(value: unknown, request: { connectionIds: string[]; inventoryRevision: string; retryOf: string | null }): ManagedRolloutResolution {
  const input = record(value, 'resolution')
  const plan = validateRolloutPlan(input.plan)
  const resolutionId = string(input.resolutionId, 'resolution-id')
  const expiresAt = input.expiresAt

  if (!Number.isSafeInteger(expiresAt) || (expiresAt as number) <= Date.now()) {throw new Error('managed-rollouts-expired-resolution')}

  if (input.inventoryRevision !== request.inventoryRevision || plan.inventoryRevision !== request.inventoryRevision || plan.retryOf !== request.retryOf) {
    throw new Error('managed-rollouts-resolution-provenance-mismatch')
  }

  const actual = plan.rows.map(row => row.connectionId).sort()

  if (canonical(actual) !== canonical([...request.connectionIds].sort())) {throw new Error('managed-rollouts-resolution-membership-mismatch')}
  validateRolloutTarget(input.target)

  if (canonical(input.target) !== canonical(plan.target)) {throw new Error('managed-rollouts-resolution-target-mismatch')}

  return { resolutionId, expiresAt: expiresAt as number, inventoryRevision: request.inventoryRevision, plan }
}

function parseChanges(value: unknown): PlanChange[] {
  if (!Array.isArray(value) || value.length > 500) {throw new Error('managed-rollouts-invalid-changes')}

  return value.map(change => {
    const item = record(change, 'change')
    const field = string(item.field, 'change-field')

    if (!['membership', 'identity', 'source', 'head', 'scopes', 'eligibility'].includes(field)) {
      throw new Error('managed-rollouts-invalid-change-field')
    }

    return {
      installId: string(item.installId, 'change-install-id'),
      field: field as PlanChange['field'],
      before: item.before === null ? null : string(item.before, 'change-before'),
      after: item.after === null ? null : string(item.after, 'change-after')
    }
  })
}

function parsePreflight(value: unknown): ManagedRolloutPreflight {
  const input = record(value, 'preflight')

  if (typeof input.ok !== 'boolean') {throw new Error('managed-rollouts-invalid-preflight')}
  const canonicalPlan = validateRolloutPlan(input.canonicalPlan)
  const token = input.token === null ? null : string(input.token, 'review-token')
  const requestId = input.requestId === null ? null : string(input.requestId, 'start-request-id')
  const rolloutId = input.rolloutId === null ? null : string(input.rolloutId, 'rollout-id')
  const expiresAt = input.expiresAt === null ? null : input.expiresAt
  const planDigest = input.planDigest === null ? null : string(input.planDigest, 'plan-digest')

  if (expiresAt !== null && !Number.isSafeInteger(expiresAt)) {throw new Error('managed-rollouts-invalid-preflight-expiry')}
  const changes = parseChanges(input.changes)
  const blockers = stringList(input.blockers, 'blockers')

  if (input.ok && (!token || !requestId || !rolloutId || !planDigest || expiresAt === null || (expiresAt as number) <= Date.now() || blockers.length || changes.length)) {
    throw new Error('managed-rollouts-invalid-preflight-approval')
  }

  return { ok: input.ok, token, requestId, rolloutId, expiresAt: expiresAt as number | null, planDigest, canonicalPlan, changes, blockers }
}

function parseResponse(value: unknown): RolloutResponse {
  const input = record(value, 'response')

  if (!Number.isSafeInteger(input.revision) || (input.revision as number) < 0) {throw new Error('managed-rollouts-invalid-response')}

  if (input.snapshot === null) {return { revision: input.revision as number, snapshot: null }}
  const snapshot = validateRolloutSnapshot(input.snapshot)

  if (snapshot.revision !== input.revision) {throw new Error('managed-rollouts-invalid-response')}

  return { revision: input.revision as number, snapshot }
}

function bridge(): ManagedRolloutsBridge | null {
  if (bridgeOverride) {return bridgeOverride}

  if (typeof window === 'undefined') {return null}

  return window.hermesDesktop?.connections?.managedRollouts ?? null
}

function setUnsupported(reason: string, capabilities: RolloutCapabilities | null = null): void {
  const current = $managedRollouts.get()
  $managedRollouts.set({ ...current, status: 'unsupported', error: reason, capabilities })
}

function schedulePoll(delayMs: number, generation = pollGeneration): void {
  if (!pollingActive) {return}

  if (pollTimer !== undefined) {clearTimeout(pollTimer)}
  pollTimer = setTimeout(() => {
    if (generation === pollGeneration && pollingActive) {void pollManagedRollouts()}
  }, Math.max(0, delayMs))
}

async function capabilityAvailable(requestBridge: ManagedRolloutsBridge): Promise<boolean> {
  try {
    const result = validateRolloutCapabilities(await requestBridge.capabilities())

    if (!result.available) {
      setUnsupported(result.reason ?? 'managed-rollouts-unavailable', result)

      return false
    }

    $managedRollouts.set({ ...$managedRollouts.get(), capabilities: result })

    return true
  } catch (error: unknown) {
    setUnsupported(error instanceof Error ? error.message : String(error))

    return false
  }
}

export async function pollManagedRollouts(): Promise<void> {
  if (pollInFlight) {return pollInFlight}
  const requestBridge = bridge()
  const generation = pollGeneration

  const run = (async () => {
    if (!requestBridge || !(await capabilityAvailable(requestBridge))) {
      if (pollingActive && generation === pollGeneration) {schedulePoll(pollingInterval, generation)}

      return
    }

    const previous = $managedRollouts.get()
    $managedRollouts.set({
      ...previous,
      status: previous.snapshot ? 'reconnecting' : 'loading',
      error: null
    })

    try {
      const activeRevision = await requestBridge.activeRevision()

      if (activeRevision !== null && (!Number.isSafeInteger(activeRevision) || activeRevision < 0)) {
        throw new Error('managed-rollouts-invalid-active-revision')
      }

      const current = $managedRollouts.get()

      if (activeRevision === null) {
        let settled = current.snapshot

        if (settled && requestBridge.get && !['completed', 'completed-with-exclusions', 'stopped'].includes(settled.phase)) {
          const latest = await requestBridge.get(settled.id)

          if (latest !== null) {settled = validateRolloutSnapshot(latest)}
        }

        if (generation !== pollGeneration) {return}
        $managedRollouts.set({
          ...$managedRollouts.get(),
          status: 'ready',
          revision: settled?.revision ?? current.revision,
          snapshot: settled,
          active: false,
          error: null
        })
        retryDelay = 1_000
        schedulePoll(pollingInterval, generation)

        return
      }

      if (activeRevision === current.revision && current.revision !== null && current.active) {
        $managedRollouts.set({ ...current, status: 'ready', error: null, active: true })
        retryDelay = 1_000
        schedulePoll(pollingInterval, generation)

        return
      }

      const response = parseResponse(await requestBridge.read(null))

      if (generation !== pollGeneration) {return}
      const currentAfterRead = $managedRollouts.get()

      if (
        response.snapshot &&
        currentAfterRead.snapshot?.id === response.snapshot.id &&
        response.revision < currentAfterRead.snapshot.revision
      ) {
        $managedRollouts.set({ ...currentAfterRead, status: 'ready', error: null })
        schedulePoll(pollingInterval, generation)

        return
      }

      if (response.snapshot === null) {
        // A terminal null is an acknowledgement for this revision, not proof
        // that the last settled snapshot never existed.
        $managedRollouts.set({ ...currentAfterRead, status: 'ready', error: null, active: true })
        retryDelay = 1_000
        schedulePoll(pollingInterval, generation)

        return
      }

      $managedRollouts.set({
        status: 'ready',
        revision: response.revision,
        snapshot: response.snapshot,
        error: null,
        active: true,
        capabilities: currentAfterRead.capabilities
      })
      retryDelay = 1_000
      schedulePoll(pollingInterval, generation)
    } catch (error: unknown) {
      if (generation !== pollGeneration) {return}
      const message = error instanceof Error ? error.message : String(error)
      $managedRollouts.set({ ...$managedRollouts.get(), status: 'error', error: message })
      schedulePoll(retryDelay, generation)
      retryDelay = Math.min(retryDelay * 2, 30_000)
    }
  })()

  pollInFlight = run.finally(() => {
    pollInFlight = null
  })

  return pollInFlight
}

export async function readManagedRolloutInventory(): Promise<ManagedRolloutInventory> {
  const requestBridge = bridge()

  if (!requestBridge?.inventory) {throw new Error('managed-rollouts-inventory-unavailable')}

  return parseInventory(await requestBridge.inventory())
}

export async function resolveManagedRolloutTarget(request: {
  connectionIds: string[]
  inventoryRevision: string
  retryOf: string | null
}): Promise<ManagedRolloutResolution> {
  const requestBridge = bridge()

  if (!requestBridge?.resolveTarget) {throw new Error('managed-rollouts-resolution-unavailable')}

  if (!request.connectionIds.length || new Set(request.connectionIds).size !== request.connectionIds.length) {
    throw new Error('managed-rollouts-invalid-selection')
  }

  return parseResolution(await requestBridge.resolveTarget(request), request)
}

export async function preflightManagedRollout(draft: RolloutDraft): Promise<ManagedRolloutPreflight> {
  const requestBridge = bridge()

  if (!requestBridge?.preflight) {throw new Error('managed-rollouts-preflight-unavailable')}

  if (!draft.waves.length || draft.waves.some(wave => !wave.length)) {throw new Error('managed-rollouts-invalid-waves')}

  return parsePreflight(await requestBridge.preflight(draft))
}

export async function startManagedRollout(request: { token: string; requestId: string }): Promise<{ id: string; revision: number }> {
  const requestBridge = bridge()

  if (!requestBridge?.start) {throw new Error('managed-rollouts-start-unavailable')}
  const response = record(await requestBridge.start(request), 'start-response')

  if (response.ok !== true || !Number.isSafeInteger(response.revision) || (response.revision as number) < 0) {
    throw new Error(typeof response.code === 'string' ? response.code : 'managed-rollouts-start-refused')
  }

  return { id: string(response.id, 'started-id'), revision: response.revision as number }
}

export async function getManagedRolloutSnapshot(id: string): Promise<RolloutSnapshot | null> {
  const requestBridge = bridge()

  if (!requestBridge?.get) {throw new Error('managed-rollouts-get-unavailable')}
  const result = await requestBridge.get(id)

  return result === null ? null : validateRolloutSnapshot(result)
}

export async function readManagedRolloutHistory(): Promise<ManagedRolloutHistoryEntry[]> {
  const requestBridge = bridge()

  if (!requestBridge?.history) {throw new Error('managed-rollouts-history-unavailable')}
  const page = record(await requestBridge.history({ limit: 50 }), 'history')

  if (!Array.isArray(page.items) || page.items.length > 50) {throw new Error('managed-rollouts-invalid-history')}

  return page.items.map(value => {
    const item = record(value, 'history-row')

    if (!Number.isSafeInteger(item.revision) || (item.revision as number) < 0 || typeof item.archived !== 'boolean') {
      throw new Error('managed-rollouts-invalid-history-row')
    }

    return {
      id: string(item.id, 'history-id'),
      revision: item.revision as number,
      phase: string(item.phase, 'history-phase'),
      updatedAt: string(item.updatedAt, 'history-updated-at'),
      unresolvedInstallIds: stringList(item.unresolvedInstallIds, 'unresolved-install-ids'),
      archived: item.archived
    }
  })
}

export async function sendManagedRolloutCommand(command: RolloutCommand): Promise<{
  ok: boolean
  id: string | null
  revision: number | null
  code: string | null
  message: string | null
  changes: PlanChange[]
}> {
  const requestBridge = bridge()

  if (!requestBridge?.command) {throw new Error('managed-rollouts-command-unavailable')}

  if (!command.requestId) {throw new Error('managed-rollouts-request-id-required')}
  const validated = validateRolloutCommand(command)
  const requestId = validated.requestId
  const key = `${requestId}:${canonical(command)}`

  if (commandInFlight) {
    if (commandInFlight.key !== key) {throw new Error('managed-rollouts-command-conflict')}

    return commandInFlight.promise as ReturnType<typeof sendManagedRolloutCommand>
  }

  const promise = requestBridge.command({ ...validated }).then(value => {
    const response = record(value, 'command-response')

    if (typeof response.ok !== 'boolean' || (response.id !== null && typeof response.id !== 'string') ||
        (response.revision !== null && !Number.isSafeInteger(response.revision))) {
      throw new Error('managed-rollouts-invalid-command-response')
    }

    return {
      ok: response.ok,
      id: response.id as string | null,
      revision: response.revision as number | null,
      code: response.code === null ? null : string(response.code, 'command-code'),
      message: response.message === null ? null : string(response.message, 'command-message'),
      changes: parseChanges(response.changes)
    }
  }).finally(() => {
    if (commandInFlight?.key === key) {commandInFlight = null}
  })

  commandInFlight = { key, promise }

  return promise
}

export function startManagedRolloutPolling(intervalMs = 5_000): () => void {
  stopManagedRolloutPolling()
  pollingActive = true
  pollingInterval = Math.max(250, intervalMs)
  retryDelay = 1_000
  pollGeneration += 1
  void pollManagedRollouts()

  return stopManagedRolloutPolling
}

export function stopManagedRolloutPolling(): void {
  pollingActive = false
  pollGeneration += 1

  if (pollTimer !== undefined) {clearTimeout(pollTimer)}
  pollTimer = undefined
}

/** @internal */
export function _setManagedRolloutsBridgeForTests(next: ManagedRolloutsBridge | null): void {
  bridgeOverride = next
}

/** @internal */
export function _resetManagedRolloutsForTests(): void {
  stopManagedRolloutPolling()
  pollInFlight = null
  commandInFlight = null
  bridgeOverride = null
  $managedRollouts.set(initial)
}
