import type { ClerkOrganizationLike, ClerkUserLike } from './clerk-session-org'

export interface ClerkSessionResourceLike {
  clearCache?: () => void
  getToken: (options?: { skipCache?: boolean }) => Promise<null | string>
}

export interface ClerkSessionClerkLike {
  loaded?: boolean
  createOrganization: (input: { name: string; slug?: string }) => Promise<ClerkOrganizationLike>
  session?: ClerkSessionResourceLike | null
  setActive: (params: { organization: null | string }) => Promise<void>
  signOut: () => Promise<void>
  user?: ClerkUserLike | null
}

export type SubmitClerkToken = (token: string) => Promise<unknown>

export interface ClerkTokenSyncResult {
  tokenSyncError: null | string
}

export function clerkErrorMessage(error: unknown): string {
  if (error instanceof Error && error.message) {
    return error.message
  }

  return String(error || 'Clerk request failed')
}

async function syncSessionToken(
  clerk: ClerkSessionClerkLike,
  submitToken: SubmitClerkToken | undefined
): Promise<ClerkTokenSyncResult> {
  clerk.session?.clearCache?.()

  const token = (await clerk.session?.getToken({ skipCache: true })) ?? null

  if (!token || !submitToken) {
    return { tokenSyncError: null }
  }

  try {
    await submitToken(token)

    return { tokenSyncError: null }
  } catch (error) {
    return { tokenSyncError: clerkErrorMessage(error) }
  }
}

export async function applyActiveOrganization(
  clerk: ClerkSessionClerkLike,
  organizationId: null | string,
  submitToken?: SubmitClerkToken
): Promise<ClerkTokenSyncResult> {
  if (clerk.loaded === false) {
    throw new Error('Clerk is not ready')
  }

  await clerk.setActive({ organization: organizationId })

  return syncSessionToken(clerk, submitToken)
}

export async function refreshClerkSession(
  clerk: ClerkSessionClerkLike,
  submitToken?: SubmitClerkToken
): Promise<ClerkTokenSyncResult> {
  if (clerk.loaded === false) {
    throw new Error('Clerk is not ready')
  }

  return syncSessionToken(clerk, submitToken)
}

export async function createClerkOrganization(
  clerk: ClerkSessionClerkLike,
  input: { name: string; slug?: string }
): Promise<ClerkOrganizationLike> {
  if (clerk.loaded === false) {
    throw new Error('Clerk is not ready')
  }

  return clerk.createOrganization(input)
}
