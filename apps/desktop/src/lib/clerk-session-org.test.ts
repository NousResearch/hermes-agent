import { describe, expect, it, vi } from 'vitest'

import { loadClerkOrganizations, summarizeClerkUser, withActiveOrganization } from './clerk-session-org'

describe('Clerk organization summaries', () => {
  it('pages through every membership and keeps the membership role', async () => {
    const getOrganizationMemberships = vi
      .fn()
      .mockResolvedValueOnce({
        data: [
          {
            organization: { id: 'org_1', imageUrl: null, membersCount: 2, name: 'Acme', slug: 'acme' },
            role: 'org:admin'
          }
        ],
        total_count: 2
      })
      .mockResolvedValueOnce({
        data: [
          {
            organization: { id: 'org_2', imageUrl: null, membersCount: 4, name: 'Beta', slug: null },
            roleName: 'Member'
          }
        ],
        total_count: 2
      })

    const organizations = await loadClerkOrganizations({ getOrganizationMemberships, id: 'user_1' }, 1)

    expect(getOrganizationMemberships).toHaveBeenNthCalledWith(1, { initialPage: 1, pageSize: 1 })
    expect(getOrganizationMemberships).toHaveBeenNthCalledWith(2, { initialPage: 2, pageSize: 1 })
    expect(organizations).toEqual([
      { id: 'org_1', imageUrl: null, membersCount: 2, name: 'Acme', role: 'org:admin', slug: 'acme' },
      { id: 'org_2', imageUrl: null, membersCount: 4, name: 'Beta', role: 'Member', slug: null }
    ])
  })

  it('returns an empty list when the user has no membership loader', async () => {
    expect(await loadClerkOrganizations({ id: 'user_1' })).toEqual([])
  })

  it('adds the active organization when the membership page did not include it', () => {
    const active = {
      id: 'org_2',
      imageUrl: null,
      membersCount: 1,
      name: 'Beta',
      role: null,
      slug: null
    }

    expect(
      withActiveOrganization(
        [
          {
            id: 'org_1',
            imageUrl: null,
            membersCount: 1,
            name: 'Acme',
            role: 'admin',
            slug: 'acme'
          }
        ],
        active
      )
    ).toEqual([active, expect.objectContaining({ id: 'org_1' })])
  })

  it('summarizes the primary email and display name', () => {
    expect(
      summarizeClerkUser({
        fullName: 'Ada Lovelace',
        id: 'user_1',
        imageUrl: 'https://example.test/ada.png',
        primaryEmailAddress: { emailAddress: 'ada@example.test' },
        username: 'ada'
      })
    ).toEqual({
      email: 'ada@example.test',
      fullName: 'Ada Lovelace',
      id: 'user_1',
      imageUrl: 'https://example.test/ada.png',
      username: 'ada'
    })
  })
})
