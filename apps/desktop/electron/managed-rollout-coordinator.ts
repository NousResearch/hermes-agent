/**
 * Main-owned managed rollout coordinator.
 *
 * This deliberately contains no Electron, IPC, filesystem, or SSH code. It
 * reduces the durable rollout projection and serializes the two operations
 * that can authorize a remote mutation: persisting `launch-authorized` and
 * handing its opaque capability to the managed SSH service. Adapters own the
 * real journal, evidence sweep, and service implementations.
 */

export type ManagedRolloutPhase =
  | 'queued'
  | 'running'
  | 'awaiting-promotion'
  | 'paused'
  | 'stopped'
  | 'attention-required'
  | 'reconciling'
  | 'completed'
  | 'completed-with-exclusions'

export type ManagedRolloutPolicy = 'manual' | 'auto-after-canary'
export type ManagedRolloutAttemptState =
  | 'none'
  | 'intent-recorded'
  | 'authorized'
  | 'observed'
  | 'updated'
  | 'already-current'
  | 'failed'
  | 'refused'
  | 'unverified'
  | 'recovery-required'
  | 'skipped'
  | 'cancelled-before-launch'

/** Matches the CLI's canonical reviewed-source wire object. */
export interface ManagedRolloutReviewedSource {
  repositoryRoot: string
  originUrl: string
  resolvedRef: string
  targetSha: string
  assuranceProfile: string
  assuranceEvidenceSha256: string
  assuranceGeneration: number
}

export interface ManagedRolloutTarget {
  installId: string
  installationFingerprint: string
  connectionId: string
  sourceFingerprint: string
  targetSha: string
  reviewedSource: ManagedRolloutReviewedSource
  correlationId: string
  wave: number
}

export interface ManagedRolloutAttempt extends ManagedRolloutTarget {
  state: ManagedRolloutAttemptState
  excluded?: boolean
  reason?: string
}

export interface ManagedRolloutState {
  id: string
  revision: number
  queueGeneration: number
  phase: ManagedRolloutPhase
  policy: ManagedRolloutPolicy
  currentWave: number
  canaryApproved: boolean
  continuationRequired: boolean
  stopRequested: boolean
  attempts: Record<string, ManagedRolloutAttempt>
}

export interface ManagedRolloutPlan {
  id: string
  revision: number
  queueGeneration: number
  policy: ManagedRolloutPolicy
  targets: readonly ManagedRolloutTarget[]
}

export type ManagedRolloutAction =
  | { kind: 'start' }
  | { kind: 'record-intent'; installId: string }
  | { kind: 'launch-authorized'; installId: string }
  | { kind: 'launch-observed'; installId: string }
  | { kind: 'terminal'; installId: string; outcome: 'updated' | 'already-current' | 'failed' | 'refused' | 'unverified' }
  | { kind: 'pause' }
  | { kind: 'resume' }
  | { kind: 'stop' }
  | { kind: 'promote'; auto?: boolean }
  | { kind: 'restart' }
  | { kind: 'reconciled'; phase: Extract<ManagedRolloutPhase, 'paused' | 'awaiting-promotion' | 'attention-required' | 'completed' | 'completed-with-exclusions'> }
  | { kind: 'exclude'; installId: string }

export interface ManagedRolloutTransition {
  ok: boolean
  state: ManagedRolloutState
  reason?: string
}

export interface ManagedRolloutAuthorization {
  rolloutId: string
  installId: string
  installationFingerprint: string
  sourceFingerprint: string
  targetSha: string
  reviewedSource: ManagedRolloutReviewedSource
  correlationId: string
  queueGeneration: number
}

/** An unforgeable adapter-owned capability; it must never cross IPC. */
export type ManagedRolloutLaunchCapability = object

export interface ManagedRolloutJournalAdapter {
  persistAuthorization: (authorization: ManagedRolloutAuthorization) => Promise<void>
  persistEvent?: (event: { kind: string; rolloutId: string; installId?: string; correlationId?: string }) => Promise<void>
}

export interface ManagedRolloutServiceAdapter {
  issueCapability: (authorization: ManagedRolloutAuthorization) => ManagedRolloutLaunchCapability
  launch: (authorization: ManagedRolloutAuthorization, capability: ManagedRolloutLaunchCapability) => Promise<void>
}

export interface ManagedRolloutEvidenceProof {
  rolloutId: string
  revision: number
  queueGeneration: number
  processGeneration: number
  valid: boolean
  reason: string | null
  admissions: Array<{
    installId: string
    installationFingerprint: string
    sourceFingerprint: string
    reviewedSource: ManagedRolloutReviewedSource
    observationGeneration: number
    observedAt: string
  }>
}

export interface ManagedRolloutEvidenceAdapter {
  sweep: (state: ManagedRolloutState) => Promise<ManagedRolloutEvidenceProof>
}

export interface ManagedRolloutCoordinatorDependencies {
  journal: ManagedRolloutJournalAdapter
  service: ManagedRolloutServiceAdapter
  evidence: ManagedRolloutEvidenceAdapter
  processGeneration?: number
}

function cloneState(state: ManagedRolloutState): ManagedRolloutState {
  return {
    ...state,
    attempts: Object.fromEntries(Object.entries(state.attempts).map(([id, attempt]) => [id, { ...attempt }]))
  }
}

function refuse(state: ManagedRolloutState, reason: string): ManagedRolloutTransition {
  return { ok: false, state, reason }
}

function pending(attempt: ManagedRolloutAttempt): boolean {
  return attempt.state === 'none' || attempt.state === 'intent-recorded'
}

function committed(attempt: ManagedRolloutAttempt): boolean {
  return attempt.state === 'authorized' || attempt.state === 'observed' || attempt.state === 'unverified'
}

function healthy(attempt: ManagedRolloutAttempt): boolean {
  return attempt.state === 'updated' || attempt.state === 'already-current'
}

function isTerminal(state: ManagedRolloutPhase): boolean {
  return state === 'stopped' || state === 'completed' || state === 'completed-with-exclusions'
}

function currentWaveAttempts(state: ManagedRolloutState): ManagedRolloutAttempt[] {
  return Object.values(state.attempts).filter(attempt => attempt.wave === state.currentWave && !attempt.excluded)
}

function canFinishWave(state: ManagedRolloutState): boolean {
  const wave = currentWaveAttempts(state)
  return wave.length > 0 && wave.every(healthy)
}

function hasLaterWork(state: ManagedRolloutState): boolean {
  return Object.values(state.attempts).some(attempt => !attempt.excluded && attempt.wave > state.currentWave)
}

function sameReviewedSource(left: ManagedRolloutReviewedSource, right: ManagedRolloutReviewedSource): boolean {
  return (
    left.repositoryRoot === right.repositoryRoot &&
    left.originUrl === right.originUrl &&
    left.resolvedRef === right.resolvedRef &&
    left.targetSha === right.targetSha &&
    left.assuranceProfile === right.assuranceProfile &&
    left.assuranceEvidenceSha256 === right.assuranceEvidenceSha256 &&
    left.assuranceGeneration === right.assuranceGeneration
  )
}

function proofMatchesState(state: ManagedRolloutState, proof: ManagedRolloutEvidenceProof, processGeneration: number): boolean {
  if (
    !proof.valid ||
    proof.rolloutId !== state.id ||
    proof.revision !== state.revision ||
    proof.queueGeneration !== state.queueGeneration ||
    proof.processGeneration !== processGeneration
  ) {
    return false
  }

  const active = Object.values(state.attempts).filter(attempt => !attempt.excluded)
  return active.every(attempt => {
    const admissions = proof.admissions.filter(admission => admission.installId === attempt.installId)
    const admission = admissions[0]
    return (
      admissions.length === 1 &&
      admission.installationFingerprint === attempt.installationFingerprint &&
      admission.sourceFingerprint === attempt.sourceFingerprint &&
      admission.reviewedSource.targetSha === attempt.targetSha &&
      sameReviewedSource(admission.reviewedSource, attempt.reviewedSource)
    )
  })
}

/**
 * The reducer is intentionally total: it returns the old state plus a reason
 * for every illegal edge. This makes command rejection observable without a
 * second, divergent state machine in an adapter or renderer.
 */
export function reduceManagedRollout(state: ManagedRolloutState, action: ManagedRolloutAction): ManagedRolloutTransition {
  const next = cloneState(state)
  const attempt = 'installId' in action ? next.attempts[action.installId] : undefined

  if (action.kind === 'start') {
    if (state.phase !== 'queued') return refuse(state, 'rollout-already-started')
    next.phase = 'running'
    return { ok: true, state: next }
  }

  if (action.kind === 'restart') {
    if (isTerminal(state.phase)) return refuse(state, 'terminal-rollout-cannot-restart')
    next.phase = 'reconciling'
    next.continuationRequired = true
    return { ok: true, state: next }
  }

  if (action.kind === 'reconciled') {
    if (state.phase !== 'reconciling') return refuse(state, 'rollout-is-not-reconciling')
    next.phase = action.phase
    return { ok: true, state: next }
  }

  if (action.kind === 'pause') {
    if (!['running', 'awaiting-promotion'].includes(state.phase)) return refuse(state, 'rollout-cannot-pause')
    next.phase = 'paused'
    return { ok: true, state: next }
  }

  if (action.kind === 'resume') {
    if (state.phase !== 'paused') return refuse(state, 'rollout-is-not-paused')
    if (state.continuationRequired) return refuse(state, 'restart-requires-explicit-promotion')
    next.phase = 'running'
    return { ok: true, state: next }
  }

  if (action.kind === 'stop') {
    if (isTerminal(state.phase)) return refuse(state, 'rollout-already-terminal')
    for (const row of Object.values(next.attempts)) {
      if (pending(row)) {
        row.state = row.state === 'intent-recorded' ? 'cancelled-before-launch' : 'skipped'
        row.reason = 'stopped'
      }
    }
    next.stopRequested = true
    next.phase = Object.values(next.attempts).some(committed) ? 'paused' : 'stopped'
    return { ok: true, state: next }
  }

  if (action.kind === 'record-intent') {
    if (!attempt) return refuse(state, 'unknown-installation')
    if (
      state.phase !== 'running' ||
      attempt.wave !== state.currentWave ||
      attempt.state !== 'none' ||
      Object.values(state.attempts).some(committed)
    ) {
      return refuse(state, 'intent-not-admissible')
    }
    attempt.state = 'intent-recorded'
    return { ok: true, state: next }
  }

  if (action.kind === 'launch-authorized') {
    if (!attempt) return refuse(state, 'unknown-installation')
    if (state.phase !== 'running' || attempt.state !== 'intent-recorded') return refuse(state, 'authorization-not-admissible')
    attempt.state = 'authorized'
    return { ok: true, state: next }
  }

  if (action.kind === 'launch-observed') {
    if (!attempt) return refuse(state, 'unknown-installation')
    if (attempt.state !== 'authorized') return refuse(state, 'observation-not-admissible')
    attempt.state = 'observed'
    return { ok: true, state: next }
  }

  if (action.kind === 'terminal') {
    if (!attempt) return refuse(state, 'unknown-installation')
    if (!['authorized', 'observed', 'unverified'].includes(attempt.state)) return refuse(state, 'terminal-correlation-not-admissible')
    attempt.state = action.outcome
    if (!healthy(attempt)) next.phase = 'attention-required'
    else if (state.phase === 'paused') {
      if (next.stopRequested && !Object.values(next.attempts).some(committed)) next.phase = 'stopped'
    }
    else if (canFinishWave(next)) {
      if (hasLaterWork(next)) next.phase = 'awaiting-promotion'
      else next.phase = Object.values(next.attempts).some(row => row.excluded) ? 'completed-with-exclusions' : 'completed'
    }
    return { ok: true, state: next }
  }

  if (action.kind === 'promote') {
    if (state.phase !== 'awaiting-promotion') return refuse(state, 'promotion-not-admissible')
    if (!canFinishWave(state)) return refuse(state, 'unhealthy-wave-cannot-promote')
    if (state.currentWave === 0 && action.auto) return refuse(state, 'canary-requires-manual-promotion')
    if (action.auto && (!state.canaryApproved || state.continuationRequired || state.policy !== 'auto-after-canary')) {
      return refuse(state, 'auto-promotion-not-admissible')
    }
    if (state.currentWave === 0) next.canaryApproved = true
    if (!action.auto) next.continuationRequired = false
    next.currentWave += 1
    next.phase = 'running'
    return { ok: true, state: next }
  }

  if (action.kind === 'exclude') {
    if (!attempt) return refuse(state, 'unknown-installation')
    if (attempt.wave < state.currentWave || attempt.state !== 'none') {
      return refuse(state, 'only-uncommitted-next-wave-work-can-be-excluded')
    }
    const remaining = Object.values(next.attempts).filter(row => row.wave === attempt.wave && !row.excluded && row.installId !== attempt.installId)
    if (remaining.length === 0) return refuse(state, 'cannot-exclude-every-target-in-wave')
    attempt.excluded = true
    attempt.state = 'skipped'
    attempt.reason = 'excluded'
    next.queueGeneration += 1
    return { ok: true, state: next }
  }

  return refuse(state, 'unknown-rollout-action')
}

export function createManagedRolloutState(plan: ManagedRolloutPlan): ManagedRolloutState {
  const attempts = Object.fromEntries(plan.targets.map(target => [target.installId, { ...target, state: 'none' as const }]))
  return {
    id: plan.id,
    revision: plan.revision,
    queueGeneration: plan.queueGeneration,
    phase: 'queued',
    policy: plan.policy,
    currentWave: 0,
    canaryApproved: false,
    continuationRequired: false,
    stopRequested: false,
    attempts
  }
}

export function createManagedRolloutCoordinator(
  initial: ManagedRolloutState,
  deps: ManagedRolloutCoordinatorDependencies
) {
  let state = cloneState(initial)
  let queue = Promise.resolve()
  const acceptedStarts = new Map<string, Promise<ManagedRolloutState>>()
  const processGeneration = deps.processGeneration ?? 1

  const admit = <T>(operation: () => Promise<T>): Promise<T> => {
    const next = queue.then(operation, operation)
    queue = next.then(() => undefined, () => undefined)
    return next
  }

  const apply = (action: ManagedRolloutAction): ManagedRolloutTransition => {
    const transition = reduceManagedRollout(state, action)
    if (transition.ok) state = transition.state
    return transition
  }

  const authorizationFor = (attempt: ManagedRolloutAttempt): ManagedRolloutAuthorization => ({
    rolloutId: state.id,
    installId: attempt.installId,
    installationFingerprint: attempt.installationFingerprint,
    sourceFingerprint: attempt.sourceFingerprint,
    targetSha: attempt.targetSha,
    reviewedSource: attempt.reviewedSource,
    correlationId: attempt.correlationId,
    queueGeneration: state.queueGeneration
  })

  const authorize = (installId: string) =>
    admit(async () => {
      const intent = apply({ kind: 'record-intent', installId })
      if (!intent.ok) return intent
      const attempt = state.attempts[installId]
      const authorization = authorizationFor(attempt)

      // This awaited write remains inside queue admission. A Stop queued after
      // it cannot be admitted until capability consumption/service handoff.
      try {
        await deps.journal.persistAuthorization(authorization)
      } catch (error) {
        return { ok: false, state, reason: `authorization-persist-failed:${String(error)}` }
      }

      const committed = apply({ kind: 'launch-authorized', installId })
      if (!committed.ok) return committed
      const capability = deps.service.issueCapability(authorization)
      const launch = deps.service.launch(authorization, capability)
      void launch.then(
        () => undefined,
        () => {
          void admit(async () => apply({ kind: 'terminal', installId, outcome: 'unverified' }))
        }
      )
      return committed
    })

  const command = (action: Extract<ManagedRolloutAction, { kind: 'pause' | 'resume' | 'stop' | 'restart' | 'exclude' }>) =>
    admit(async () => apply(action))

  const start = (requestId: string) => {
    const duplicate = acceptedStarts.get(requestId)
    if (duplicate) return duplicate
    const accepted = admit(async () => {
      const transition = apply({ kind: 'start' })
      if (!transition.ok) throw new Error(transition.reason)
      return cloneState(state)
    })
    acceptedStarts.set(requestId, accepted)
    return accepted
  }

  const promote = async (auto = false): Promise<ManagedRolloutTransition> => {
    const snapshot = cloneState(state)
    const proof = await deps.evidence.sweep(snapshot)
    return admit(async () => {
      if (!proofMatchesState(state, proof, processGeneration)) {
        return refuse(state, 'promotion-proof-is-stale-or-invalid')
      }
      return apply({ kind: 'promote', auto })
    })
  }

  const terminal = (installId: string, correlationId: string, outcome: Extract<ManagedRolloutAction, { kind: 'terminal' }>['outcome']) =>
    admit(async () => {
      if (state.attempts[installId]?.correlationId !== correlationId) return refuse(state, 'terminal-correlation-mismatch')
      return apply({ kind: 'terminal', installId, outcome })
    })

  return {
    get snapshot() {
      return cloneState(state)
    },
    start,
    authorize,
    command,
    promote,
    terminal,
    reduce: apply
  }
}
