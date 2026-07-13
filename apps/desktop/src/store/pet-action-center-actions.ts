import { $clarifyRequests, type ClarifyRequest, clearClarifyRequest } from './clarify'
import {
  $petActionCenter,
  type PetActionCenterApprovalItem,
  type PetActionCenterClarifyItem,
  type PetActionCenterErrorCode,
  type PetActionCenterItem,
  selectPetActionCenterItem,
  setPetActionCenterActionStatus
} from './pet-action-center'
import type { PetActionCenterControl } from './pet-overlay'
import { normalizeProfileKey } from './profile'
import { promptIdentityKey, type StoredPromptRequest } from './prompt-identity'
import { $approvalRequests, type ApprovalRequest, clearApprovalRequest } from './prompts'
import { $cronSessions, $messagingSessions, $sessions } from './session'

export interface PetActionCenterGateway {
  readonly connectionState: string
  request<T = unknown>(method: string, params?: Record<string, unknown>): Promise<T>
}

export interface PetActionCenterActionDependencies {
  ensureProfile: (profile: string) => Promise<void>
  gatewayForProfile: (profile: string) => PetActionCenterGateway | null
  resumeSession: (profile: string, storedSessionId: string) => Promise<boolean> | boolean
}

export interface PetActionCenterActions {
  handle: (control: PetActionCenterControl) => Promise<void>
}

const BACKEND_APPROVAL_CHOICE = {
  'approve-always': 'always',
  'approve-once': 'once',
  'approve-session': 'session',
  deny: 'deny'
} as const

function currentItem(itemId: string): PetActionCenterItem | null {
  return $petActionCenter.get().items.find(item => item.id === itemId) ?? null
}

function start(itemId: string): void {
  setPetActionCenterActionStatus({ status: 'submitting', itemId })
}

function finish(
  itemId: string,
  result:
    | { status: 'success' | 'stale' }
    | {
        status: 'error'
        errorCode: PetActionCenterErrorCode
      }
): void {
  const active = $petActionCenter.get().action

  if (active?.status !== 'submitting' || active.itemId !== itemId) {
    return
  }

  setPetActionCenterActionStatus({ ...result, itemId })
}

function failWithoutRpc(
  itemId: string,
  errorCode: 'capability-denied' | 'disconnected' | 'invalid-item' | 'item-not-found' | 'session-unverified'
): void {
  setPetActionCenterActionStatus({ status: 'error', itemId, errorCode })
}

function resultRecord(result: unknown): Record<string, unknown> | null {
  if (typeof result !== 'object' || result === null) {
    return null
  }

  return result as Record<string, unknown>
}

function approvalResultStatus(result: unknown): 'error' | 'stale' | 'success' {
  const response = resultRecord(result)

  if (response?.resolved === false || response?.resolved === 0) {
    return 'stale'
  }

  return response?.resolved === true ? 'success' : 'error'
}

function clarifyResultSucceeded(result: unknown): boolean {
  const response = resultRecord(result)

  return response?.status === 'ok'
}

function clarifyErrorIsStale(error: unknown): boolean {
  return error instanceof Error && error.message === 'no pending answer request'
}

function approvalRequestFor(item: PetActionCenterApprovalItem): StoredPromptRequest<ApprovalRequest> | null {
  return $approvalRequests.get()[promptIdentityKey(item.profile, item.sessionId)] ?? null
}

function clarifyRequestFor(item: PetActionCenterClarifyItem): StoredPromptRequest<ClarifyRequest> | null {
  return $clarifyRequests.get()[promptIdentityKey(item.profile, item.sessionId)] ?? null
}

function clearApprovalIfCurrent(
  item: PetActionCenterApprovalItem,
  requestIdentity: string
): boolean {
  const current = approvalRequestFor(item)

  if (current?.requestIdentity !== requestIdentity) {
    return false
  }

  clearApprovalRequest({ profile: item.profile, sessionId: item.sessionId })

  return true
}

function clearClarifyIfCurrent(item: PetActionCenterClarifyItem, requestIdentity: string): boolean {
  const current = clarifyRequestFor(item)

  if (current?.requestIdentity !== requestIdentity) {
    return false
  }

  clearClarifyRequest({ profile: item.profile, sessionId: item.sessionId, requestId: current.requestId })

  return true
}

function storedSessionIsVerified(profile: string, storedSessionId: string): boolean {
  const normalizedProfile = normalizeProfileKey(profile)

  return [...$sessions.get(), ...$cronSessions.get(), ...$messagingSessions.get()].some(
    session =>
      normalizeProfileKey(session.profile) === normalizedProfile &&
      (session.id === storedSessionId || session._lineage_root_id === storedSessionId)
  )
}

async function respondToApproval(
  control: Extract<PetActionCenterControl, { type: 'action-center-approval' }>,
  dependencies: PetActionCenterActionDependencies
): Promise<void> {
  const item = currentItem(control.itemId)

  if (!item) {
    failWithoutRpc(control.itemId, 'item-not-found')

    return
  }

  if (item.kind !== 'approval' || !item.allowedActions.includes(control.choice)) {
    failWithoutRpc(control.itemId, 'capability-denied')

    return
  }

  if (!item.sessionId) {
    failWithoutRpc(control.itemId, 'invalid-item')

    return
  }

  const original = approvalRequestFor(item)

  if (!original) {
    failWithoutRpc(control.itemId, 'item-not-found')

    return
  }

  const gateway = dependencies.gatewayForProfile(item.profile)

  if (!gateway || gateway.connectionState !== 'open') {
    failWithoutRpc(control.itemId, 'disconnected')

    return
  }

  start(control.itemId)

  try {
    const params: Record<string, unknown> = {
      choice: BACKEND_APPROVAL_CHOICE[control.choice],
      session_id: item.sessionId
    }

    if (control.choice === 'deny' && control.reason !== undefined) {
      params.reason = control.reason
    }

    const result = await gateway.request('approval.respond', params)

    const status = approvalResultStatus(result)

    if (status === 'stale') {
      clearApprovalIfCurrent(item, original.requestIdentity)
      finish(control.itemId, { status: 'stale' })

      return
    }

    if (status === 'error') {
      finish(control.itemId, { status: 'error', errorCode: 'rpc-failed' })

      return
    }

    clearApprovalIfCurrent(item, original.requestIdentity)
    finish(control.itemId, { status: 'success' })
  } catch {
    finish(control.itemId, { status: 'error', errorCode: 'rpc-failed' })
  }
}

async function respondToClarify(
  control: Extract<PetActionCenterControl, { type: 'action-center-clarify' }>,
  dependencies: PetActionCenterActionDependencies
): Promise<void> {
  const item = currentItem(control.itemId)

  if (!item) {
    failWithoutRpc(control.itemId, 'item-not-found')

    return
  }

  const capability = control.answer.length > 0 ? 'clarify-respond' : 'clarify-skip'

  if (item.kind !== 'clarify' || !item.allowedActions.includes(capability)) {
    failWithoutRpc(control.itemId, 'capability-denied')

    return
  }

  if (!item.sessionId) {
    failWithoutRpc(control.itemId, 'invalid-item')

    return
  }

  const original = clarifyRequestFor(item)

  if (!original) {
    failWithoutRpc(control.itemId, 'item-not-found')

    return
  }

  const gateway = dependencies.gatewayForProfile(item.profile)

  if (!gateway || gateway.connectionState !== 'open') {
    failWithoutRpc(control.itemId, 'disconnected')

    return
  }

  start(control.itemId)

  try {
    const result = await gateway.request('clarify.respond', {
      answer: control.answer,
      request_id: original.requestId
    })

    if (!clarifyResultSucceeded(result)) {
      finish(control.itemId, { status: 'error', errorCode: 'rpc-failed' })

      return
    }

    clearClarifyIfCurrent(item, original.requestIdentity)
    finish(control.itemId, { status: 'success' })
  } catch (error) {
    if (clarifyErrorIsStale(error)) {
      clearClarifyIfCurrent(item, original.requestIdentity)
      finish(control.itemId, { status: 'stale' })

      return
    }

    finish(control.itemId, { status: 'error', errorCode: 'rpc-failed' })
  }
}

async function openExactSession(
  control: Extract<PetActionCenterControl, { type: 'action-center-open-session' }>,
  dependencies: PetActionCenterActionDependencies
): Promise<void> {
  const item = currentItem(control.itemId)

  if (!item) {
    failWithoutRpc(control.itemId, 'item-not-found')

    return
  }

  if (!item.allowedActions.includes('open-in-app') || !item.storedSessionId) {
    failWithoutRpc(control.itemId, 'session-unverified')

    return
  }

  if (!storedSessionIsVerified(item.profile, item.storedSessionId)) {
    failWithoutRpc(control.itemId, 'session-unverified')

    return
  }

  start(control.itemId)

  try {
    await dependencies.ensureProfile(item.profile)

    const current = currentItem(control.itemId)

    if (
      !current ||
      !current.allowedActions.includes('open-in-app') ||
      current.storedSessionId !== item.storedSessionId ||
      !storedSessionIsVerified(item.profile, item.storedSessionId)
    ) {
      finish(control.itemId, { status: 'error', errorCode: 'session-unverified' })

      return
    }

    const opened = await dependencies.resumeSession(item.profile, item.storedSessionId)

    if (!opened) {
      finish(control.itemId, { status: 'error', errorCode: 'open-failed' })

      return
    }

    finish(control.itemId, { status: 'success' })
  } catch {
    finish(control.itemId, { status: 'error', errorCode: 'open-failed' })
  }
}

export function createPetActionCenterActions(
  dependencies: PetActionCenterActionDependencies
): PetActionCenterActions {
  return {
    async handle(control) {
      if (control.type !== 'action-center-select' && $petActionCenter.get().action?.status === 'submitting') {
        return
      }

      switch (control.type) {
        case 'action-center-select':
          selectPetActionCenterItem(control.itemId)

          return

        case 'action-center-approval':
          await respondToApproval(control, dependencies)

          return

        case 'action-center-clarify':
          await respondToClarify(control, dependencies)

          return

        case 'action-center-open-session':
          await openExactSession(control, dependencies)
      }
    }
  }
}
