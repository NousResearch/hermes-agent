// The lane badge: a branch lane in a project shows its PR the same way a
// session row does. Regression — the whole `projects/` folder never touched
// the PR store, so "Show PR" badged the rows under a lane but never the lane
// itself. These tests drive the REAL component with the real stores for
// layout + pull-requests; only the git bridge and the heavy neighbors are
// stubbed.
import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { HermesBranchPullRequest } from '@/global'

import type { SidebarSessionGroup as Lane } from './workspace-groups'

afterEach(cleanup)

vi.mock('@/lib/desktop-git', () => ({
  desktopGit: vi.fn(() => undefined)
}))

// Partial: the real @/store/profile (kept above) imports this module's config
// and profile calls at load. Only the PR transcript scan needs stubbing — it
// would otherwise shell out through the gateway.
vi.mock('@/hermes', async importOriginal => ({
  ...((await importOriginal()) as Record<string, unknown>),
  scanSessionPullRequests: vi.fn()
}))

vi.mock('@/i18n', () => ({
  useI18n: () => ({
    t: {
      profiles: { switchToProfile: (name: string) => `Switch to ${name}` },
      sidebar: {
        newSessionIn: (label: string) => `New session in ${label}`,
        noSessions: 'No sessions yet',
        projects: {
          copyPath: 'Copy path',
          menu: 'Actions',
          removeWorktree: 'Remove worktree',
          reveal: 'Reveal in file manager',
          toggle: (label: string, open: boolean) => `${open ? 'Expand' : 'Collapse'} ${label}`
        },
        showMoreIn: (n: number, label: string) => `Show ${n} more in ${label}`
      },
      statusStack: { coding: { switchFailed: (b: string) => `Switch to ${b} failed` } }
    }
  })
}))

vi.mock('@/store/projects', () => ({
  copyPath: vi.fn(),
  revealPath: vi.fn(),
  switchBranchInRepo: vi.fn()
}))

// Partial mock: layout.ts computes off this module's atoms ($showAllProfiles),
// so replacing it wholesale breaks the real store this test drives.
vi.mock('@/store/profile', async importOriginal => ({
  ...((await importOriginal()) as Record<string, unknown>),
  newSessionInProfile: vi.fn(),
  selectProfile: vi.fn()
}))

vi.mock('@/store/notifications', () => ({
  notifyError: vi.fn()
}))

vi.mock('@/store/coding-status', () => ({
  openWorktreeDialog: vi.fn()
}))

const { $sidebarRowMeta } = await import('@/store/layout')
const { $pullRequestsByBranch, branchPrKey } = await import('@/store/pull-requests')
const { SidebarWorkspaceGroup } = await import('./workspace-group')

const REPO = '/home/ethie/src/hermes-agent'

const openPr: HermesBranchPullRequest = {
  branch: 'ethie/title-gen',
  draft: false,
  number: 8123,
  state: 'open',
  title: 'fix: title generation',
  url: 'https://github.com/x/y/pull/8123'
} as never

const lane = (label: string, extra: Partial<Lane> = {}): Lane => ({
  id: `${REPO}::branch::${label}`,
  label,
  path: REPO,
  sessions: [],
  ...extra
})

const renderLane = (label: string, extra: Partial<Lane> = {}) =>
  render(<SidebarWorkspaceGroup group={lane(label, extra)} renderRows={() => null} repoPath={REPO} />)

beforeEach(() => {
  $pullRequestsByBranch.set({ [branchPrKey(REPO, 'ethie/title-gen')]: openPr })
  $sidebarRowMeta.set(['pr'])
})

describe('branch lane PR badge', () => {
  it('badges the lane with the PR for its branch when Show PR is on', () => {
    renderLane('ethie/title-gen')

    expect(screen.getByRole('button', { name: 'Open pull request #8123' })).toBeTruthy()
  })

  it('shows no badge when Show PR is off — the toggle governs the lane like it governs a row', () => {
    $sidebarRowMeta.set([])
    renderLane('ethie/title-gen')

    expect(screen.queryByRole('button', { name: 'Open pull request #8123' })).toBeNull()
  })

  it('shows no badge on a lane whose branch has no PR', () => {
    renderLane('ethie/some-other-branch')

    expect(screen.queryByRole('button', { name: /Open pull request/ })).toBeNull()
  })

  it('never badges the kanban aggregate — it is many branches in one lane', () => {
    $pullRequestsByBranch.set({ [branchPrKey(REPO, 'Tasks')]: { ...openPr, branch: 'Tasks' } })
    renderLane('Tasks', { isKanban: true })

    expect(screen.queryByRole('button', { name: /Open pull request/ })).toBeNull()
  })

  it('keeps the lane label clickable beside the badge (the chip is a sibling, not nested)', () => {
    renderLane('ethie/title-gen')

    const chip = screen.getByRole('button', { name: 'Open pull request #8123' })
    // A button inside a button is invalid HTML and unreachable by keyboard.
    expect(chip.closest('button')).toBe(chip)
  })
})
