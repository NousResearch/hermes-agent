import { normalizeProfileKey } from './profile'

export interface PromptIdentity {
  profile: string
  sessionId: string | null
}

export interface PromptTarget extends PromptIdentity {
  requestId?: string
}

export function promptIdentityKey(profile: string | null | undefined, sessionId: string | null | undefined): string {
  return JSON.stringify([normalizeProfileKey(profile), sessionId ?? ''])
}

export function normalizePromptIdentity<T extends PromptIdentity>(request: T): T {
  return { ...request, profile: normalizeProfileKey(request.profile) }
}
