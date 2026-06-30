import { atom } from 'nanostores'

import type { ComposerAttachment } from './composer'

export type PendingPlanState = 'planning' | 'ready'

export interface PendingPlan {
  attachments: ComposerAttachment[]
  createdAt: number
  originalText: string
  state: PendingPlanState
}

export const PLAN_PENDING_SESSION = '__pending_plan_session__'

export const $planModeEnabled = atom(false)
export const $pendingPlansBySession = atom<Record<string, PendingPlan>>({})

export function setPlanModeEnabled(enabled: boolean) {
  $planModeEnabled.set(enabled)
}

export function setPendingPlan(sid: string, plan: Omit<PendingPlan, 'createdAt'> & { createdAt?: number }) {
  if (!sid) {
    return
  }

  $pendingPlansBySession.set({
    ...$pendingPlansBySession.get(),
    [sid]: { ...plan, createdAt: plan.createdAt ?? Date.now() }
  })
}

export function markPendingPlanReady(sid: string) {
  const current = $pendingPlansBySession.get()
  const plan = current[sid]

  if (!plan || plan.state === 'ready') {
    return
  }

  $pendingPlansBySession.set({
    ...current,
    [sid]: { ...plan, state: 'ready' }
  })
}

export function clearPendingPlan(sid: string) {
  const current = $pendingPlansBySession.get()

  if (!(sid in current)) {
    return
  }

  const { [sid]: _drop, ...rest } = current
  $pendingPlansBySession.set(rest)
}

export function migratePendingPlan(fromSid: string, toSid: string) {
  if (!fromSid || !toSid || fromSid === toSid) {
    return
  }

  const current = $pendingPlansBySession.get()
  const plan = current[fromSid]

  if (!plan || current[toSid]) {
    return
  }

  const { [fromSid]: _drop, ...rest } = current
  $pendingPlansBySession.set({ ...rest, [toSid]: plan })
}
