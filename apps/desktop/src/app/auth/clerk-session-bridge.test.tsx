// @vitest-environment jsdom
import { act, cleanup, render, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { $clerkSession, clerkSessionHost, DISABLED_CLERK_SNAPSHOT } from '@/store/clerk-session'

import { ClerkSessionBridge } from './clerk-session-bridge'

const mocks = vi.hoisted(() => {
  const getToken = vi.fn(async () => 'token-1')
  const setActive = vi.fn(async (_params: { organization: null | string }) => undefined)

  const createOrganization = vi.fn(async (input: { name: string; slug?: string }) => ({
    id: 'org_new',
    imageUrl: null,
    membersCount: 1,
    name: input.name,
    slug: input.slug ?? null
  }))

  const signOut = vi.fn(async () => undefined)

  return {
    clerk: {
      createOrganization,
      loaded: true,
      session: { clearCache: vi.fn(), getToken },
      setActive,
      signOut
    },
    createOrganization,
    getToken,
    organization: {
      id: 'org_1',
      imageUrl: null,
      membersCount: 2,
      name: 'Acme',
      slug: 'acme'
    },
    setActive,
    signOut,
    submitClerkToken: vi.fn(async () => ({
      ok: true,
      provider: 'unitalk',
      user_id: 'user_1'
    })),
    user: {
      fullName: 'Ada Lovelace',
      getOrganizationMemberships: vi.fn(async () => ({
        data: [
          {
            organization: {
              id: 'org_1',
              imageUrl: null,
              membersCount: 2,
              name: 'Acme',
              slug: 'acme'
            },
            role: 'org:admin'
          }
        ],
        total_count: 1
      })),
      id: 'user_1',
      imageUrl: 'https://example.test/ada.png',
      primaryEmailAddress: { emailAddress: 'ada@example.test' },
      username: 'ada'
    }
  }
})

vi.mock('@clerk/electron/react', () => ({
  useAuth: () => ({ getToken: mocks.getToken, isLoaded: true, isSignedIn: true }),
  useClerk: () => mocks.clerk,
  useOrganization: () => ({ organization: mocks.organization }),
  useUser: () => ({ user: mocks.user })
}))

vi.mock('@/api/config', () => ({
  submitClerkToken: mocks.submitClerkToken
}))

let relayListener: ((payload: unknown) => void) | null = null

const installRelay = () => {
  Object.defineProperty(window, 'hermesDesktop', {
    configurable: true,
    value: {
      windowRelay: {
        onMessage: (listener: (payload: unknown) => void) => {
          relayListener = listener

          return () => {
            relayListener = null
          }
        },
        send: vi.fn()
      }
    }
  })
}

describe('ClerkSessionBridge', () => {
  beforeEach(() => {
    installRelay()
    relayListener = null
    mocks.getToken.mockClear()
    mocks.submitClerkToken.mockClear()
    mocks.setActive.mockClear()
    mocks.createOrganization.mockClear()
    mocks.signOut.mockClear()
    $clerkSession.set({ ...DISABLED_CLERK_SNAPSHOT })
  })

  afterEach(() => {
    cleanup()
    Object.defineProperty(window, 'hermesDesktop', { configurable: true, value: undefined })
  })

  it('publishes the user and active organization', async () => {
    render(<ClerkSessionBridge />)

    await waitFor(() => expect($clerkSession.get().status).toBe('signed-in'))

    expect($clerkSession.get().user?.email).toBe('ada@example.test')
    expect($clerkSession.get().activeOrganization?.name).toBe('Acme')
    expect($clerkSession.get().organizations).toHaveLength(1)
  })

  it('switches organization, refreshes the token, and syncs it to the backend', async () => {
    render(<ClerkSessionBridge />)

    await waitFor(() => expect($clerkSession.get().status).toBe('signed-in'))

    await act(async () => {
      await clerkSessionHost.setActiveOrganization('org_1')
    })

    expect(mocks.setActive).toHaveBeenCalledWith({ organization: 'org_1' })
    expect(mocks.getToken).toHaveBeenCalledWith({ skipCache: true })
    expect(mocks.submitClerkToken).toHaveBeenCalledWith('token-1')
  })

  it('creates and activates an organization', async () => {
    render(<ClerkSessionBridge />)

    await waitFor(() => expect($clerkSession.get().status).toBe('signed-in'))

    await act(async () => {
      await clerkSessionHost.createOrganization({ name: 'New Org' })
    })

    expect(mocks.createOrganization).toHaveBeenCalledWith({ name: 'New Org' })
    expect(mocks.setActive).toHaveBeenCalledWith({ organization: 'org_new' })
  })

  it('signs out through Clerk', async () => {
    render(<ClerkSessionBridge />)

    await waitFor(() => expect($clerkSession.get().status).toBe('signed-in'))

    await act(async () => {
      await clerkSessionHost.signOut()
    })

    expect(mocks.signOut).toHaveBeenCalled()
    expect($clerkSession.get().status).toBe('signed-out')
  })

  it('applies an organization switch relayed from another window', async () => {
    render(<ClerkSessionBridge />)

    await waitFor(() => expect($clerkSession.get().status).toBe('signed-in'))
    expect(relayListener).not.toBeNull()

    await act(async () => {
      relayListener?.({
        kind: 'unitalk.clerk.session.changed',
        organizationId: 'org_1',
        source: 'other-window'
      })
    })

    await waitFor(() => expect(mocks.setActive).toHaveBeenCalledWith({ organization: 'org_1' }))
  })
})
