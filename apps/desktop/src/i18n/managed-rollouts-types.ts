import type { TranslationOverride } from '@hermes/shared/i18n'

import type { Translations } from './types'

/**
 * The managed-rollout namespace is intentionally separate from the desktop
 * locale godfiles. The catalog widens this namespace at its merge seam while
 * the existing `Translations` contract remains compatible with older callers.
 */
export type ManagedRolloutMessages = {
  title: string
  noActive: string
  history: string
  unresolved: (count: number) => string
  archived: string
  active: string
  sections: {
    inventory: string
    fleet: string
    preparation: string
    configuration: string
    preflight: string
    wavePreview: string
    active: string
    controls: string
    recovery: string
    history: string
    summary: string
  }
  actions: {
    checking: string
    renewReview: string
    reviewSelected: string
    retryInventory: string
    select: string
    selected: string
    prepare: string
    preparing: string
    recheckEligibility: string
    continueToPreflight: string
    start: string
    starting: string
    pause: string
    pausing: string
    stop: string
    stopping: string
    resume: string
    verifyBeforePromotion: string
    verifyingBeforePromotion: string
    recheckOutcome: string
    recoverConnections: string
    retry: string
    exclude: string
    stopAndPlanRetry: string
    archiveStoppedRollout: string
  }
  status: {
    activePhase: (phase: string) => string
    queued: string
    preparing: string
    ready: string
    running: string
    awaitingPromotion: string
    attentionRequired: string
    paused: string
    stopped: string
    completed: string
    completedWithExclusions: string
    failed: string
    refused: string
    unknown: string
    unverified: string
    recoveryRequired: string
    fenced: string
    alreadyCurrent: string
    pending: string
  }
  policy: {
    manual: string
    automatic: string
    mode: (mode: string) => string
    canaryGate: (gate: string) => string
  }
  labels: {
    preparationOutcome: (installId: string, outcome: string, correlationId: string, restoreOk: boolean) => string
    exclusionReason: string
    receiptPending: string
    pinnedTarget: string
    planChange: (installId: string, field: string, before: string, after: string) => string
    blocked: (reason: string) => string
    aliases: (count: number) => string
    observedHead: string
    eligibility: string
    observationTime: string
    unknownFact: string
    selectedCount: (count: number) => string
    showingTargets: (visible: number, total: number) => string
    searchTargets: string
    progressionMode: string
    concurrency: string
    inventoryRevision: (revision: string, capturedMono: number) => string
    targetDetails: (label: string, alias: string | null | undefined, machineId: string, installId: string) => string
    wave: (number: number, targets: string) => string
    emptyWave: (number: number) => string
    progress: (completed: number, total: number) => string
    receipt: (outcome: string, correlationId: string) => string
    readiness: (value: string) => string
    reason: (value: string) => string
    attempt: (installId: string, phase: string) => string
    historyEntry: (id: string, phase: string, updatedAt: string, unresolved: number, archived: boolean) => string
    archive: (archived: boolean) => string
  }
  warnings: {
    restartReviewRequired: string
    reviewUnavailable: string
    sharedMachine: string
    unsupportedTarget: string
    preparation: string
    preparationMayDisconnect: string
    preparationInvalidatesReview: string
    changedPlan: string
    projectionUnavailable: string
    incompatible: string
    commitmentBoundary: string
    unknownOutcome: string
    recoveryFence: string
    unknownAndFenced: string
    estimateUnavailable: string
    unavailable: string
    stale: string
    reconnecting: string
  }
  descriptions: {
    recoveryActions: string
    configuration: string
    rolloutCapacityUnavailable: (maxInstallations: number | null, reason: string | null) => string
    noObservedInstallations: string
    noMatchingTargets: string
    preparationSeparate: string
    inventoryLoading: string
    preparationFinished: (outcomes: string) => string
    inventoryObserved: string
    serialCapability: string
    canonicalPlan: string
    manualPolicy: string
    automaticPolicy: string
    commitmentBoundary: string
    lastSettledLocalFenceNotLive: string
    archiveRetainsFence: string
    noAutomaticResume: string
    historicalEstimate: string
  }
  summary: {
    outcome: (phase: string) => string
    excludedTargets: (count: number) => string
    unresolvedFences: (count: number) => string
    archive: (archived: boolean) => string
    reason: (reason: string) => string
  }
  a11y: {
    confirmPreparation: string
    confirmPreflight: string
    selectTarget: (label: string) => string
    selectedTarget: (label: string) => string
    warning: (message: string) => string
    status: (message: string) => string
    action: (message: string) => string
    historyEntry: (id: string) => string
    attemptToggle: (installId: string, expanded: boolean) => string
  }
}

export type ManagedRolloutOverrides = TranslationOverride<ManagedRolloutMessages>

export type CatalogTranslations = Translations & {
  settings: { managedRollouts: ManagedRolloutMessages }
}
