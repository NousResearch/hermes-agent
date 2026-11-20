import { cleanup, fireEvent, render, renderHook, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { SidebarProvider } from '@/components/ui/sidebar'
import type { HermesApiRequest } from '@/global'
import { $dismissedWorktreeIds, dismissWorktree } from '@/store/layout'
import { $startWorkSessionRequest, $worktreeDialog } from '@/store/projects'

import { EnteredProjectContent } from './entered-content'
import { useRepoWorktreeMap } from './model'
import type { SidebarProjectTree } from './workspace-groups'
import { StartWorkButton } from './workspace-header'
import { WorktreeDialog } from './worktree-dialog'

type ActGlobal = typeof globalThis & { IS_REACT_ACT_ENVIRONMENT?: boolean }

const STUDIO = { connectionId: 'studio', profile: 'research' }

// A project only the Studio gateway holds, as withGatewayProjectRows builds it.
const studioApp = {
  connectionId: 'studio',
  id: JSON.stringify(['gateway-project', 'studio', '/code/app']),
  isAuto: true,
  label: 'app · Studio',
  path: '/code/app',
  profile: 'research',
  repos: [],
  sessionCount: 1
} as SidebarProjectTree

// This window's own git (Electron) and the REST bridge every gateway's /api/git rides.
function installBridge() {
  const localGit = {
    branchList: vi.fn(async () => [{ isRemote: false, name: 'main' }]),
    branchSwitch: vi.fn(async () => {}),
    baseBranchList: vi.fn(async () => [{ isDefault: true, isRemote: true, name: 'origin/main' }]),
    worktreeAdd: vi.fn(async () => ({ branch: 'local', path: '/local' })),
    worktreeList: vi.fn(async () => [])
  }

  const api = vi.fn(async (request: HermesApiRequest) => {
    if (request.path.startsWith('/api/git/base-branches')) {
      return { branches: [{ isDefault: true, isRemote: true, name: 'origin/main' }] }
    }

    if (request.path === '/api/git/worktree/add') {
      const body = request.body as { branch: string }

      return { branch: body.branch, path: `/code/app/.worktrees/${body.branch}` }
    }

    if (request.path.startsWith('/api/git/branches')) {
      return { branches: [{ isRemote: false, name: 'main' }] }
    }

    if (request.path === '/api/git/branch/switch') {
      return {}
    }

    if (request.path.startsWith('/api/git/worktrees')) {
      return { worktrees: [{ branch: 'feat', path: '/code/app/.worktrees/feat' }] }
    }

    throw new Error(`unexpected ${request.path}`)
  })

  ;(window as { hermesDesktop?: unknown }).hermesDesktop = { api, git: localGit }

  return { api, localGit }
}

beforeEach(() => {
  ;(globalThis as ActGlobal).IS_REACT_ACT_ENVIRONMENT = false
})

afterEach(() => {
  cleanup()
  $worktreeDialog.set(null)
  $dismissedWorktreeIds.set([])
  $startWorkSessionRequest.set(null)
  delete (window as { hermesDesktop?: unknown }).hermesDesktop
  ;(globalThis as ActGlobal).IS_REACT_ACT_ENVIRONMENT = true
})

describe("another gateway's project git", () => {
  it('cuts a new worktree on the owning gateway and starts its session there', async () => {
    const { api, localGit } = installBridge()

    render(
      <>
        <StartWorkButton project={studioApp} />
        <WorktreeDialog />
      </>
    )

    fireEvent.click(screen.getByRole('button', { name: 'New worktree' }))
    fireEvent.change(await screen.findByPlaceholderText('e.g. my-feature'), { target: { value: 'my-work' } })
    await screen.findByText('origin/main')
    fireEvent.click(screen.getAllByRole('button', { name: 'New worktree' }).at(-1)!)

    await waitFor(() => expect($startWorkSessionRequest.get()).not.toBeNull())

    const add = api.mock.calls.map(([request]) => request).find(request => request.path === '/api/git/worktree/add')

    expect(add).toMatchObject({ ...STUDIO, method: 'POST', body: { branch: 'my-work', path: '/code/app' } })
    expect(localGit.worktreeAdd).not.toHaveBeenCalled()
    expect($startWorkSessionRequest.get()).toMatchObject({ owner: STUDIO, path: '/code/app/.worktrees/my-work' })
  })

  it("lists the entered project's worktrees on the owning gateway", async () => {
    const { api, localGit } = installBridge()

    const { result } = renderHook(() => useRepoWorktreeMap(['/code/app'], true, studioApp))

    await waitFor(() => expect(result.current[0]['/code/app']).toHaveLength(1))
    expect(api.mock.calls[0][0]).toMatchObject({ ...STUDIO, path: '/api/git/worktrees?path=%2Fcode%2Fapp' })
    expect(localGit.worktreeList).not.toHaveBeenCalled()
  })

  it("switches an entered project's lane branch on the owning gateway before its new session", async () => {
    const { api, localGit } = installBridge()
    const onNewSession = vi.fn()
    const main = { id: '/code/app::branch::main', isMain: true, label: 'main', path: '/code/app', sessions: [] }

    render(
      <SidebarProvider>
        <EnteredProjectContent
          onNewSession={onNewSession}
          project={{
            ...studioApp,
            repos: [{ groups: [main], id: '/code/app', label: 'app', path: '/code/app', sessionCount: 0 }]
          }}
          renderRows={() => null}
        />
      </SidebarProvider>
    )

    fireEvent.click(screen.getByRole('button', { name: /new session/i }))

    await waitFor(() => expect(onNewSession).toHaveBeenCalledWith('/code/app', STUDIO))
    expect(api.mock.calls.map(([request]) => request)).toEqual([
      expect.objectContaining({ ...STUDIO, path: '/api/git/branches?path=%2Fcode%2Fapp' }),
      expect.objectContaining({
        ...STUDIO,
        body: { branch: 'main', path: '/code/app' },
        path: '/api/git/branch/switch'
      })
    ])
    expect(localGit.branchList).not.toHaveBeenCalled()
    expect(localGit.branchSwitch).not.toHaveBeenCalled()
  })
  it("hides another gateway's lane apart from a local lane at the same path", async () => {
    installBridge()

    const feat = { branch: 'feat', detached: false, isMain: false, locked: false, path: '/code/app/.worktrees/feat' }
    const main = { id: '/code/app::branch::main', isMain: true, label: 'main', path: '/code/app', sessions: [] }

    // This device hid its own checkout at that path.
    dismissWorktree('/code/app/.worktrees/feat')

    render(
      <SidebarProvider>
        <EnteredProjectContent
          project={{
            ...studioApp,
            repos: [{ groups: [main], id: '/code/app', label: 'app', path: '/code/app', sessionCount: 0 }]
          }}
          renderRows={() => null}
          repoWorktrees={{ '/code/app': [feat] }}
        />
      </SidebarProvider>
    )

    expect(screen.getByTitle(/feat/)).toBeTruthy()

    fireEvent.pointerDown(screen.getAllByRole('button', { name: 'Actions' }).at(-1)!, { button: 0, ctrlKey: false })
    fireEvent.click(await screen.findByText('Remove worktree…'))
    fireEvent.click(await screen.findByRole('button', { name: 'Hide from sidebar' }))

    await waitFor(() => expect(screen.queryByTitle(/feat/)).toBeNull())
    expect($dismissedWorktreeIds.get()).toEqual([
      '/code/app/.worktrees/feat',
      JSON.stringify(['gateway-lane', 'studio', 'research', '/code/app/.worktrees/feat'])
    ])
  })
})
