import { describe, expect, it, vi } from 'vitest'

import { applyActiveOrganization, createClerkOrganization, refreshClerkSession } from './clerk-session-operations'

function makeClerk() {
  const getToken = vi.fn(async () => 'token-1')
  const clearCache = vi.fn()
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
    clearCache,
    clerk: {
      createOrganization,
      loaded: true,
      session: { clearCache, getToken },
      setActive,
      signOut
    },
    createOrganization,
    getToken,
    setActive,
    signOut
  }
}

describe('Clerk session operations', () => {
  it('switches the active organization and submits a fresh token', async () => {
    const { clerk, clearCache, getToken, setActive } = makeClerk()
    const submitToken = vi.fn(async () => undefined)

    const result = await applyActiveOrganization(clerk, 'org_1', submitToken)

    expect(setActive).toHaveBeenCalledWith({ organization: 'org_1' })
    expect(clearCache).toHaveBeenCalled()
    expect(getToken).toHaveBeenCalledWith({ skipCache: true })
    expect(submitToken).toHaveBeenCalledWith('token-1')
    expect(result.tokenSyncError).toBeNull()
  })

  it('removes the active organization for a personal account', async () => {
    const { clerk, setActive } = makeClerk()

    await applyActiveOrganization(clerk, null)

    expect(setActive).toHaveBeenCalledWith({ organization: null })
  })

  it('keeps the switch when the backend token submit fails', async () => {
    const { clerk, setActive } = makeClerk()

    const submitToken = vi.fn(async () => {
      throw new Error('gateway offline')
    })

    const result = await applyActiveOrganization(clerk, 'org_1', submitToken)

    expect(setActive).toHaveBeenCalledWith({ organization: 'org_1' })
    expect(result.tokenSyncError).toBe('gateway offline')
  })

  it('refreshes the token without changing the active organization', async () => {
    const { clerk, clearCache, getToken } = makeClerk()
    const submitToken = vi.fn(async () => undefined)

    const result = await refreshClerkSession(clerk, submitToken)

    expect(clearCache).toHaveBeenCalled()
    expect(getToken).toHaveBeenCalledWith({ skipCache: true })
    expect(submitToken).toHaveBeenCalledWith('token-1')
    expect(result.tokenSyncError).toBeNull()
  })

  it('creates an organization through Clerk', async () => {
    const { clerk, createOrganization } = makeClerk()

    const created = await createClerkOrganization(clerk, { name: 'Acme', slug: 'acme' })

    expect(createOrganization).toHaveBeenCalledWith({ name: 'Acme', slug: 'acme' })
    expect(created.id).toBe('org_new')
  })

  it('refuses to act before Clerk has loaded', async () => {
    const { clerk } = makeClerk()

    await expect(applyActiveOrganization({ ...clerk, loaded: false }, 'org_1')).rejects.toThrow('Clerk is not ready')
  })
})
