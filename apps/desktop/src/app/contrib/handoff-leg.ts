/** Failed/uncertain submits retain the original session;
 * they must never close it or start a second build. */
import { JsonRpcGatewayError } from '@hermes/shared'

import type { ClientSessionState } from '@/app/types'
import type { HandoffPlan } from '@/components/onboarding-chat/setup-profile'
import type { SessionMessage } from '@/types/hermes'

import type { AmbientGatewayRequest } from './session-rpc-dispatcher'

export const BUILD_PROFILE = 'default'

export interface HandoffTask {
  task: string
  brief: string
  plan: HandoffPlan
}

export interface HandoffReceipt extends HandoffTask {
  runtimeId: string
  storedId: string
  /** `connectionId: null` is the ambient route for the profile (a local-only
   *  install, or a legacy primary with no registry id), never a missing owner. */
  owner: { connectionId: null | string; profile: typeof BUILD_PROFILE }
  status: 'created' | 'submitting' | 'accepted'
}

export interface HandoffSnapshot {
  session_id: string
  session_key: string
  running: boolean
  hydrating?: boolean
  messages_omitted?: boolean
  messages?: SessionMessage[]
}

export interface HandoffDeps {
  create: () => Promise<Pick<HandoffReceipt, 'runtimeId' | 'storedId' | 'owner'>>
  personalize: () => Promise<void>
  request: <T>(
    owner: HandoffReceipt['owner'],
    method: string,
    params: NonNullable<Parameters<AmbientGatewayRequest>[1]>
  ) => Promise<T>
  read: () => HandoffReceipt | null
  save: (receipt: HandoffReceipt) => void
  bind: (receipt: HandoffReceipt, running: boolean, snapshot?: HandoffSnapshot) => void
}

/** Only preflight refusals in methods_prompt authorize another submit. A
 * generic server error, like a lost ACK, may follow a side effect. */
const PREFLIGHT_REJECTIONS = new Set([4001, 4004, 4009, 4018, 4090, 4091, 4120, 4121, 5070, 5071, 5072, 5122])

interface HydratedHandoffSnapshot extends HandoffSnapshot {
  messages: SessionMessage[]
}

function verifyHandoffSnapshot(snapshot: HandoffSnapshot): asserts snapshot is HydratedHandoffSnapshot {
  if (
    snapshot.hydrating ||
    snapshot.messages_omitted ||
    !snapshot.session_id ||
    !snapshot.session_key ||
    !Array.isArray(snapshot.messages) ||
    (snapshot.running !== true && snapshot.running !== false)
  ) {
    throw new Error('Could not verify the first build. Retry when the connection recovers.')
  }
}

export async function startHandoff(deps: HandoffDeps, task: HandoffTask, recoverGone = true): Promise<HandoffReceipt> {
  let receipt = deps.read()

  if (!receipt) {
    await deps.personalize()
    const identity = await deps.create()
    receipt = { ...task, ...identity, status: 'created' }
    deps.save(receipt)
  } else {
    const snapshot = await deps.request<HandoffSnapshot>(receipt.owner, 'session.resume', {
      session_id: receipt.storedId,
      omit_messages: false
    })

    verifyHandoffSnapshot(snapshot)

    receipt = { ...receipt, runtimeId: snapshot.session_id }

    // A visible user turn in this dedicated session is durable acceptance,
    // even when the build has finished or its context has been compressed.
    // A confirmed refusal (created) cannot be overturned by a stale busy flag.
    if (
      (receipt.status === 'submitting' && snapshot.running) ||
      snapshot.messages.some(message => message.role === 'user' && message.display_kind !== 'hidden')
    ) {
      receipt = { ...receipt, status: 'accepted' }
    }

    deps.save(receipt)
    deps.bind(receipt, snapshot.running, snapshot)

    if (receipt.status === 'accepted') {
      return receipt
    }

    if (snapshot.running) {
      throw new Error(
        'The first build has no confirmed start, but its session still reports running. Retry when it is idle; no duplicate was sent.'
      )
    }

    if (receipt.status === 'submitting') {
      throw new Error(
        'The first build has not acknowledged its start. Check its session before retrying; no duplicate was sent.'
      )
    }
  }

  deps.bind(receipt, true)
  receipt = { ...receipt, status: 'submitting' }
  deps.save(receipt)

  try {
    const response = await deps.request<{ status?: string }>(receipt.owner, 'prompt.submit', {
      session_id: receipt.runtimeId,
      text: receipt.brief
    })

    if (response.status !== 'streaming') {
      throw new Error('The first build did not acknowledge starting. Check its session before retrying.')
    }
  } catch (error) {
    const code = error instanceof JsonRpcGatewayError ? error.code : undefined

    if (code !== undefined && PREFLIGHT_REJECTIONS.has(code)) {
      receipt = { ...receipt, status: 'created' }
      deps.save(receipt)

      if (code === 4001 && recoverGone) {
        return startHandoff(deps, task, false)
      }
    }

    throw error
  }

  receipt = { ...receipt, status: 'accepted' }
  deps.save(receipt)

  return receipt
}

export function paintHandoffBrief(state: ClientSessionState, brief: string, storedId: string): ClientSessionState {
  const id = `user-handoff-brief-${storedId}`

  return {
    ...state,
    messages: state.messages.some(message => message.id === id)
      ? state.messages
      : [
          ...state.messages,
          {
            id,
            role: 'user',
            parts: [{ text: brief, type: 'text' }],
            timestamp: Date.now() / 1000
          }
        ],
    busy: true,
    awaitingResponse: true,
    turnStartedAt: state.turnStartedAt ?? Date.now()
  }
}
