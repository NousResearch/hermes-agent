import type { ManagedRolloutMessages } from '../managed-rollouts-types'

export const managedRolloutsEn: ManagedRolloutMessages = {
  title: 'Managed rollouts',
  noActive: 'No active rollout snapshot.',
  history: 'Rollout history',
  unresolved: count => `Unresolved fences: ${count}`,
  archived: 'archived',
  active: 'active',
  sections: {
    fleet: 'Managed rollout fleet',
    preparation: 'Managed rollout preparation',
    configuration: 'Managed rollout configuration',
    preflight: 'Managed rollout preflight review',
    wavePreview: 'Managed rollout wave preview',
    active: 'Active managed rollout',
    controls: 'Managed rollout controls',
    recovery: 'Managed rollout recovery',
    history: 'Managed rollout history',
    summary: 'Managed rollout summary'
  },
  actions: {
    select: 'Select',
    selected: 'Selected',
    prepare: 'Prepare selected targets',
    preparing: 'Preparing…',
    recheckEligibility: 'Recheck eligibility',
    continueToPreflight: 'Continue to preflight',
    start: 'Start rollout',
    starting: 'Starting…',
    pause: 'Pause',
    pausing: 'Pausing…',
    stop: 'Stop',
    stopping: 'Stopping…',
    resume: 'Resume',
    verifyBeforePromotion: 'Verify before promotion',
    verifyingBeforePromotion: 'Verifying before promotion',
    recheckOutcome: 'Recheck',
    recoverConnections: 'Recover',
    retry: 'Retry',
    exclude: 'Exclude',
    stopAndPlanRetry: 'Stop and plan retry',
    archiveStoppedRollout: 'Archive stopped rollout'
  },
  status: {
    activePhase: phase => `Active phase: ${phase}`,
    queued: 'Queued',
    preparing: 'Preparing',
    ready: 'Ready',
    running: 'Running',
    awaitingPromotion: 'Awaiting promotion',
    attentionRequired: 'Attention required',
    paused: 'Paused',
    stopped: 'Stopped',
    completed: 'Completed',
    completedWithExclusions: 'Completed with exclusions',
    failed: 'Failed',
    refused: 'Refused',
    unknown: 'Unknown outcome',
    unverified: 'Unverified',
    recoveryRequired: 'Recovery required',
    fenced: 'Recovery fence retained',
    alreadyCurrent: 'Already current',
    pending: 'Pending evidence'
  },
  policy: {
    manual: 'Manual approval',
    automatic: 'Automatic after canary approval, while healthy',
    mode: mode => `Mode: ${mode}`,
    canaryGate: gate => `Canary gate: ${gate}`
  },
  labels: {
    progressionMode: 'Progression mode',
    concurrency: 'Concurrency',
    targetDetails: (label, alias, machineId, installId) =>
      `${label} · ${alias ? `${alias} · ` : ''}${machineId} · ${installId}`,
    wave: (number, targets) => `Wave ${number}: ${targets || 'none'}`,
    emptyWave: number => `Wave ${number}: no targets`,
    progress: (completed, total) => `Progress: ${completed} of ${total} targets completed.`,
    receipt: (outcome, correlationId) => `Receipt: ${outcome} (${correlationId})`,
    readiness: value => `Readiness: ${value}`,
    reason: value => `Reason: ${value}`,
    attempt: (installId, phase) => `${installId} · ${phase}`,
    historyEntry: (id, phase, updatedAt, unresolved, archived) =>
      `${id} · ${phase} · ${updatedAt} · unresolved ${unresolved} · ${archived ? 'archived' : 'active'}`,
    archive: archived => `Archive: ${archived ? 'archived' : 'active'}`
  },
  warnings: {
    sharedMachine: 'Shared machine; review ownership before preparation.',
    unsupportedTarget: 'Unsupported target.',
    preparation: 'Preparation follows the configured branch tip; it is not a pinned rollout.',
    preparationMayDisconnect: 'Preparation may temporarily disconnect sessions.',
    preparationInvalidatesReview:
      'Preparation invalidates prior review and requires requalification after target, alias, source, or scope changes.',
    changedPlan: 'Configuration changed; renew the review token before continuing.',
    projectionUnavailable: 'Wave projection is unavailable; preflight cannot advance.',
    incompatible: 'Current capability is incompatible with this plan; Start is unavailable.',
    commitmentBoundary:
      'Stop blocks new authorizations; committed updates may still drain, apply, and restore.',
    unknownOutcome: 'The remote outcome is unknown; do not treat an absent receipt as success.',
    recoveryFence: 'A recovery fence is retained; new unsafe work remains blocked.',
    unknownAndFenced:
      'The outcome is unknown and the recovery fence is retained; new unsafe work is blocked.',
    estimateUnavailable: 'Estimate unavailable.',
    unavailable: 'Managed rollout data is unavailable. No new work was started.',
    stale: 'This rollout snapshot may be stale. Recheck before taking action.',
    reconnecting: 'Reconnecting to managed rollout status…'
  },
  descriptions: {
    serialCapability: 'Serial capability is required by the active target contract. The selected canary remains stable until the draft changes.',
    canonicalPlan:
      'The submitted draft is exactly the previewed canonical wave plan. Renewed or changed rows require reconfirmation.',
    manualPolicy: 'Manual approval is required after every settled wave.',
    automaticPolicy: 'Automatic progression is allowed only after explicit canary approval and fresh evidence while the rollout remains healthy.',
    commitmentBoundary:
      'A committed update may finish after you stop the rollout. Stop prevents new authorizations but cannot cancel work already handed to the remote updater.',
    lastSettledLocalFenceNotLive: 'Last settled, plus local fence — not live. Remote deterioration outside this Desktop observation is not established.',
    archiveRetainsFence: 'Archiving changes presentation and history only; it does not erase evidence or release an unresolved fence.',
    noAutomaticResume: 'After restart, reconcile first and explicitly resume or promote; a stored policy is not renewed permission.',
    historicalEstimate: 'Historical estimates are informational only and do not authorize new work.'
  },
  summary: {
    outcome: phase => `Outcome: ${phase}`,
    excludedTargets: count => `Excluded targets: ${count}`,
    unresolvedFences: count => `Unresolved fences: ${count}`,
    archive: archived => `Archive: ${archived ? 'archived' : 'active'}`,
    reason: reason => `Reason: ${reason}`
  },
  a11y: {
    confirmPreparation: 'I confirm these targets are the intended separate preparation set.',
    confirmPreflight: 'I confirm this preflight review.',
    selectTarget: label => `Select target ${label}`,
    selectedTarget: label => `Deselect target ${label}`,
    warning: message => `Warning: ${message}`,
    status: message => `Status: ${message}`,
    action: message => `Action: ${message}`,
    historyEntry: id => `Open rollout ${id} details`,
    attemptToggle: (installId, expanded) => `${expanded ? 'Collapse' : 'Expand'} attempt ${installId}`
  }
}

export const managedRollouts = managedRolloutsEn
export default managedRolloutsEn
