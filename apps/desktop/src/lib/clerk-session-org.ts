import type { ClerkOrganizationSummary, ClerkUserSummary } from '@/store/clerk-session'

export interface ClerkOrganizationLike {
  id: string
  imageUrl?: null | string
  membersCount?: number
  name: string
  slug?: null | string
}

export interface ClerkMembershipLike {
  organization: ClerkOrganizationLike
  role?: null | string
  roleName?: null | string
}

export interface ClerkUserLike {
  fullName?: null | string
  getOrganizationMemberships?: (params?: {
    initialPage?: number
    pageSize?: number
  }) => Promise<{ data: ClerkMembershipLike[]; total_count?: number }>
  id: string
  imageUrl?: null | string
  primaryEmailAddress?: null | { emailAddress?: null | string }
  username?: null | string
}

export function summarizeClerkUser(user: ClerkUserLike | null | undefined): ClerkUserSummary | null {
  if (!user?.id) {
    return null
  }

  return {
    email: user.primaryEmailAddress?.emailAddress ?? null,
    fullName: user.fullName ?? null,
    id: user.id,
    imageUrl: user.imageUrl ?? null,
    username: user.username ?? null
  }
}

export function summarizeClerkOrganization(
  organization: ClerkOrganizationLike | null | undefined,
  role: null | string = null
): ClerkOrganizationSummary | null {
  if (!organization?.id) {
    return null
  }

  return {
    id: organization.id,
    imageUrl: organization.imageUrl ?? null,
    membersCount: organization.membersCount ?? 0,
    name: organization.name,
    role,
    slug: organization.slug ?? null
  }
}

export async function loadClerkOrganizations(
  user: ClerkUserLike | null | undefined,
  pageSize = 100
): Promise<ClerkOrganizationSummary[]> {
  if (!user?.getOrganizationMemberships) {
    return []
  }

  const byId = new Map<string, ClerkOrganizationSummary>()
  let page = 1
  let total = Number.POSITIVE_INFINITY

  while (byId.size < total) {
    const response = await user.getOrganizationMemberships({ initialPage: page, pageSize })
    const rows = Array.isArray(response?.data) ? response.data : []

    if (rows.length === 0) {
      break
    }

    for (const membership of rows) {
      const summary = summarizeClerkOrganization(
        membership.organization,
        membership.roleName ?? membership.role ?? null
      )

      if (summary) {
        byId.set(summary.id, summary)
      }
    }

    total = typeof response.total_count === 'number' ? response.total_count : byId.size
    page += 1
  }

  return [...byId.values()]
}

export function withActiveOrganization(
  organizations: ClerkOrganizationSummary[],
  activeOrganization: ClerkOrganizationSummary | null
): ClerkOrganizationSummary[] {
  if (!activeOrganization || organizations.some(org => org.id === activeOrganization.id)) {
    return organizations
  }

  return [activeOrganization, ...organizations]
}
