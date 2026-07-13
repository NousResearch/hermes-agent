import { normalizeProfileKey } from './profile'

export interface PromptIdentity {
  profile: string
  sessionId: string | null
}

export interface PromptRequestMetadata {
  receivedAt: number
  requestIdentity: string
  storedSessionId: string | null
}

export type PromptRequestInput<T extends PromptIdentity> = T & { storedSessionId?: string | null }
export type StoredPromptRequest<T extends PromptIdentity> = T & PromptRequestMetadata

export interface PromptTarget extends PromptIdentity {
  requestId?: string
}

export function promptIdentityKey(profile: string | null | undefined, sessionId: string | null | undefined): string {
  return JSON.stringify([normalizeProfileKey(profile), sessionId ?? ''])
}

export function normalizePromptIdentity<T extends PromptIdentity>(request: T): T {
  return { ...request, profile: normalizeProfileKey(request.profile) }
}

let requestIdentitySequence = 0

export function stampPromptRequest<T extends PromptIdentity>(
  kind: string,
  request: PromptRequestInput<T>
): StoredPromptRequest<T> {
  const normalized = normalizePromptIdentity(request)
  const receivedAt = Date.now()

  requestIdentitySequence += 1

  return {
    ...normalized,
    receivedAt,
    requestIdentity: `${kind}:${receivedAt.toString(36)}:${requestIdentitySequence.toString(36)}`,
    storedSessionId: request.storedSessionId?.trim() || null
  }
}
