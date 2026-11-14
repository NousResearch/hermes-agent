import { atom, type ReadableAtom } from 'nanostores'

export type ClerkSessionStatus = 'disabled' | 'error' | 'loading' | 'signed-in' | 'signed-out'

export interface ClerkUserSummary {
  email: null | string
  fullName: null | string
  id: string
  imageUrl: null | string
  username: null | string
}

export interface ClerkOrganizationSummary {
  id: string
  imageUrl: null | string
  membersCount: number
  name: string
  role: null | string
  slug: null | string
}

export interface ClerkSessionSnapshot {
  activeOrganization: ClerkOrganizationSummary | null
  enabled: boolean
  error: null | string
  organizations: ClerkOrganizationSummary[]
  status: ClerkSessionStatus
  tokenSyncError: null | string
  user: ClerkUserSummary | null
}

export interface ClerkOrganizationInput {
  name: string
  slug?: string
}

export interface ClerkSessionActions {
  createOrganization: (input: ClerkOrganizationInput) => Promise<ClerkOrganizationSummary>
  refresh: () => Promise<void>
  setActiveOrganization: (organizationId: null | string) => Promise<void>
  signOut: () => Promise<void>
}

export const DISABLED_CLERK_SNAPSHOT: ClerkSessionSnapshot = {
  activeOrganization: null,
  enabled: false,
  error: null,
  organizations: [],
  status: 'disabled',
  tokenSyncError: null,
  user: null
}

export const $clerkSession = atom<ClerkSessionSnapshot>({ ...DISABLED_CLERK_SNAPSHOT })

let currentActions: ClerkSessionActions | null = null

export function registerClerkSessionActions(actions: ClerkSessionActions | null): () => void {
  currentActions = actions

  return () => {
    if (currentActions === actions) {
      currentActions = null
    }
  }
}

export function requireClerkSessionActions(): ClerkSessionActions {
  if (!currentActions) {
    throw new Error('Clerk session is not ready')
  }

  return currentActions
}

export function publishClerkSession(snapshot: ClerkSessionSnapshot): void {
  $clerkSession.set(snapshot)
}

export function patchClerkSession(patch: Partial<ClerkSessionSnapshot>): void {
  $clerkSession.set({ ...$clerkSession.get(), ...patch })
}

export const clerkSessionHost = {
  snapshot: $clerkSession as ReadableAtom<ClerkSessionSnapshot>,
  setActiveOrganization: async (organizationId: null | string): Promise<void> =>
    requireClerkSessionActions().setActiveOrganization(organizationId),
  createOrganization: async (input: ClerkOrganizationInput): Promise<ClerkOrganizationSummary> =>
    requireClerkSessionActions().createOrganization(input),
  refresh: async (): Promise<void> => requireClerkSessionActions().refresh(),
  signOut: async (): Promise<void> => requireClerkSessionActions().signOut()
} satisfies {
  snapshot: ReadableAtom<ClerkSessionSnapshot>
  setActiveOrganization: (organizationId: null | string) => Promise<void>
  createOrganization: (input: ClerkOrganizationInput) => Promise<ClerkOrganizationSummary>
  refresh: () => Promise<void>
  signOut: () => Promise<void>
}
