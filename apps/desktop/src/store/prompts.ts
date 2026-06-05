import { atom } from 'nanostores'

// Blocking interactive prompts the gateway raises mid-turn. Each maps to a
// `*.request` event the Python side emits while it blocks the agent thread
// waiting for a `*.respond` RPC. Without a renderer for these, the agent
// silently stalls until its timeout (default 5 min) and the tool is BLOCKED
// — the desktop app previously handled clarify.request but not these three,
// so dangerous-command approval, sudo, and secret prompts never surfaced.

export interface ApprovalRequest {
  command: string
  description: string
  sessionId: string | null
}

// Approval is session-keyed on the backend (one in-flight approval per
// session, resolved via approval.respond {choice, session_id}). It carries
// no request_id, unlike sudo/secret which are _block()-style request/response.
export const $approvalRequest = atom<ApprovalRequest | null>(null)

export function setApprovalRequest(request: ApprovalRequest): void {
  $approvalRequest.set(request)
}

export function clearApprovalRequest(): void {
  $approvalRequest.set(null)
}

export interface SudoRequest {
  requestId: string
}

export const $sudoRequest = atom<SudoRequest | null>(null)

export function setSudoRequest(request: SudoRequest): void {
  $sudoRequest.set(request)
}

export function clearSudoRequest(requestId?: string): void {
  const current = $sudoRequest.get()

  if (!current) {
    return
  }

  if (requestId && current.requestId !== requestId) {
    return
  }

  $sudoRequest.set(null)
}

export interface SecretRequest {
  requestId: string
  envVar: string
  prompt: string
}

export const $secretRequest = atom<SecretRequest | null>(null)

export function setSecretRequest(request: SecretRequest): void {
  $secretRequest.set(request)
}

export function clearSecretRequest(requestId?: string): void {
  const current = $secretRequest.get()

  if (!current) {
    return
  }

  if (requestId && current.requestId !== requestId) {
    return
  }

  $secretRequest.set(null)
}

// Drop every in-flight prompt. Called when a turn ends (message.complete /
// error) so a stale overlay can't linger past the turn that raised it — e.g.
// if the agent was interrupted while a prompt was open.
export function clearAllPrompts(): void {
  $approvalRequest.set(null)
  $sudoRequest.set(null)
  $secretRequest.set(null)
}
