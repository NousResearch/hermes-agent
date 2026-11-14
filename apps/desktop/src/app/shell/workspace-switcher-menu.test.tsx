import { act, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { MemoryRouter, useLocation } from 'react-router'
import { afterEach, beforeAll, describe, expect, it, vi } from 'vitest'

import type { SidebarProjectTree } from '@/app/chat/sidebar/projects/workspace-groups'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuTrigger
} from '@/components/ui/dropdown-menu'
import { en } from '@/i18n/en'
import { $defaultProjectId, $projectsRpcAvailable, $projectTree } from '@/store/projects'

import type { StatusbarMenuItem } from './statusbar-controls'
import { WorkspaceSwitcherMenu } from './workspace-switcher-menu'

// Radix calls these on open; jsdom doesn't implement them.
beforeAll(() => {
  Element.prototype.scrollIntoView = vi.fn()
  Element.prototype.hasPointerCapture = vi.fn(() => false)
  Element.prototype.releasePointerCapture = vi.fn()
})

const moveSessionToProject = vi.fn<(sessionId: string, projectId: string, profile?: null | string, cwd?: string) => Promise<void>>()
const goToProject = vi.fn<(id: string, options?: { cwd?: string; newSession?: boolean }) => void>()
const setDefaultProject = vi.fn<(id: null | string) => Promise<void>>()
const openFolderAsProject = vi.fn<(dir?: string) => Promise<void>>()
const refreshProjectTree = vi.fn<() => Promise<void>>()

vi.mock('@/store/projects', async importOriginal => ({
  ...((await importOriginal()) as Record<string, unknown>),
  goToProject: (...args: [string, { cwd?: string; newSession?: boolean }?]) => goToProject(...args),
  moveSessionToProject: (...args: [string, string, (null | string)?, string?]) =>
    Promise.resolve(moveSessionToProject(...args)),
  openFolderAsProject: (...args: [string?]) => Promise.resolve(openFolderAsProject(...args)),
  refreshProjectTree: () => Promise.resolve(refreshProjectTree()),
  setDefaultProject: (...args: [null | string]) => Promise.resolve(setDefaultProject(...args))
}))

const notify = vi.fn()
const notifyError = vi.fn()

vi.mock('@/store/notifications', async importOriginal => ({
  ...((await importOriginal()) as Record<string, unknown>),
  notify: (...args: unknown[]) => {
    notify(...args)

    return 'id'
  },
  notifyError: (...args: unknown[]) => {
    notifyError(...args)

    return 'id'
  }
}))

const p = en.sidebar.projects

const proj = (overrides: Partial<SidebarProjectTree> & { id: string; label: string }): SidebarProjectTree => ({
  path: `/work/${overrides.id}`,
  repos: [],
  sessionCount: 0,
  ...overrides
})

const ALPHA = proj({ id: 'p_alpha', label: 'Alpha', color: '#f00', path: '/work/alpha' })
const BETA = proj({ id: 'p_beta', label: 'Beta', path: '/work/beta' })

const MULTI = proj({
  id: 'p_multi',
  label: 'Multi',
  path: '/work/multi',
  repos: [{ groups: [], id: 'r1', label: 'api', path: '/work/multi-api', sessionCount: 0 }]
})

const HOME = proj({ id: 'no-project', isNoProject: true, label: 'Home', path: null })

interface MenuOptions {
  busy?: boolean
  currentProjectId?: null | string
  profile?: null | string
  sessionId?: null | string
  sessionRemote?: boolean
  tree?: SidebarProjectTree[]
}

const onClose = vi.fn()

function renderMenu(options: MenuOptions = {}) {
  const {
    busy = false,
    currentProjectId = null,
    profile = 'default',
    sessionId = 'sess-1',
    sessionRemote = false,
    tree = [ALPHA, BETA]
  } = options

  $projectTree.set(tree)
  $projectsRpcAvailable.set(true)

  const changeWorkspaceItem: StatusbarMenuItem = {
    id: 'change-workspace-path',
    label: 'Change working directory',
    onSelect: () => {}
  }

  const copyPathItem: StatusbarMenuItem = { id: 'copy-workspace-path', label: 'Copy Path', onSelect: () => {} }

  const revealItems: StatusbarMenuItem[] = [
    { id: 'reveal-workspace-finder', label: 'Reveal in File Manager', onSelect: () => {} }
  ]

  return render(
    <MemoryRouter>
      <DropdownMenu open>
        <DropdownMenuTrigger asChild>
          <button type="button">chip</button>
        </DropdownMenuTrigger>
        <DropdownMenuContent>
          <WorkspaceSwitcherMenu
            busy={busy}
            changeWorkspaceItem={changeWorkspaceItem}
            copyPathItem={copyPathItem}
            currentCwd="/work/alpha"
            currentProjectId={currentProjectId}
            onClose={onClose}
            profile={profile}
            projectsCopy={p}
            revealItems={revealItems}
            sessionId={sessionId}
            sessionRemote={sessionRemote}
          />
        </DropdownMenuContent>
      </DropdownMenu>
    </MemoryRouter>
  )
}

/** Open the submenu belonging to the trigger row carrying `label`. */
async function openProjectSub(label: string): Promise<HTMLElement> {
  const trigger = screen
    .getAllByRole('menuitem')
    .find(el => el.getAttribute('data-slot') === 'dropdown-menu-sub-trigger' && within(el).queryByText(label))

  expect(trigger).toBeDefined()

  fireEvent.pointerMove(trigger as HTMLElement, { pointerType: 'mouse' })
  fireEvent.focus(trigger as HTMLElement)

  // Radix portals the submenu into its own role="menu" surface alongside the
  // main one — wait for the second menu and use the newest.
  await waitFor(() => expect(screen.getAllByRole('menu').length).toBeGreaterThanOrEqual(2))

  const menus = screen.getAllByRole('menu')

  return menus[menus.length - 1] as HTMLElement
}

afterEach(() => {
  vi.clearAllMocks()
  $projectTree.set([])
  $defaultProjectId.set(null)
  $projectsRpcAvailable.set(null)
})

describe('WorkspaceSwitcherMenu — project list', () => {
  it('lists every active-profile project with icon, path and a current marker', () => {
    renderMenu({ currentProjectId: 'p_alpha' })

    const alpha = screen
      .getAllByRole('menuitem')
      .find(el => el.getAttribute('data-slot') === 'dropdown-menu-sub-trigger' && within(el).queryByText('Alpha'))

    const beta = screen
      .getAllByRole('menuitem')
      .find(el => el.getAttribute('data-slot') === 'dropdown-menu-sub-trigger' && within(el).queryByText('Beta'))

    expect(alpha).toBeDefined()
    expect(beta).toBeDefined()
    expect(within(alpha as HTMLElement).queryByText('/work/alpha')).not.toBeNull()
    expect(alpha?.getAttribute('aria-current')).toBe('true')
    expect(beta?.getAttribute('aria-current')).toBeNull()
  })

  it('excludes projects owned by another gateway', () => {
    renderMenu({ tree: [ALPHA, proj({ connectionId: 'conn-1', id: 'p_foreign', label: 'Foreign' })] })

    expect(screen.queryByText('Foreign')).toBeNull()
    expect(screen.getByText('Alpha')).not.toBeNull()
  })

  it('shows the empty state when there are no projects', () => {
    renderMenu({ tree: [] })

    expect(screen.getByText(p.noProjects)).not.toBeNull()
  })
})

describe('WorkspaceSwitcherMenu — switch current chat (two-click single-root)', () => {
  it('moves the current chat into a single-root project on the submenu action', async () => {
    renderMenu({ currentProjectId: 'p_beta' })

    const sub = await openProjectSub('Alpha')

    fireEvent.click(within(sub).getByRole('menuitem', { name: p.switchCurrentChat }))

    await waitFor(() =>
      expect(moveSessionToProject).toHaveBeenCalledWith('sess-1', 'p_alpha', 'default', undefined)
    )
    expect(notify).toHaveBeenCalledWith(expect.objectContaining({ kind: 'success', message: p.movedTo('Alpha') }))
    expect(onClose).toHaveBeenCalled()
  })

  it('requires an explicit folder for a multi-root project (no silent primary fallback)', async () => {
    renderMenu({ currentProjectId: 'p_beta', tree: [MULTI] })

    const sub = await openProjectSub('Multi')

    expect(within(sub).queryByRole('menuitem', { name: p.switchCurrentChat })).toBeNull()
    expect(within(sub).getByText(p.chooseFolder)).not.toBeNull()

    fireEvent.click(within(sub).getAllByRole('menuitem', { name: '/work/multi-api' })[0])

    await waitFor(() =>
      expect(moveSessionToProject).toHaveBeenCalledWith('sess-1', 'p_multi', 'default', '/work/multi-api')
    )
  })

  it('offers no move target for the path-less Home bucket', async () => {
    renderMenu({ tree: [HOME] })

    const sub = await openProjectSub('Home')

    expect(within(sub).queryByRole('menuitem', { name: p.switchCurrentChat })).toBeNull()
  })

  it('omits the move for the project the chat already occupies, keeping open/new/default', async () => {
    renderMenu({ currentProjectId: 'p_alpha' })

    const sub = await openProjectSub('Alpha')

    expect(within(sub).queryByRole('menuitem', { name: p.switchCurrentChat })).toBeNull()
    expect(within(sub).getByRole('menuitem', { name: p.openProject })).not.toBeNull()
    expect(within(sub).getByRole('menuitem', { name: p.newChat })).not.toBeNull()
    expect(within(sub).getByRole('menuitem', { name: p.defaultForNewChats })).not.toBeNull()
    expect(moveSessionToProject).not.toHaveBeenCalled()

    // A different project still offers the move.
    const beta = await openProjectSub('Beta')

    expect(within(beta).getByRole('menuitem', { name: p.switchCurrentChat })).not.toBeNull()
  })
})

describe('WorkspaceSwitcherMenu — separated per-project actions', () => {
  it('opens a project without moving the chat', async () => {
    renderMenu()

    const sub = await openProjectSub('Alpha')

    fireEvent.click(within(sub).getByRole('menuitem', { name: p.openProject }))

    expect(goToProject).toHaveBeenCalledWith('p_alpha')
    expect(goToProject).toHaveBeenCalledTimes(1)
    expect(moveSessionToProject).not.toHaveBeenCalled()
  })

  it('starts a new chat in a single-root project and closes the menu', async () => {
    renderMenu()

    const sub = await openProjectSub('Alpha')

    fireEvent.click(within(sub).getByRole('menuitem', { name: p.newChat }))

    expect(goToProject).toHaveBeenCalledWith('p_alpha', { cwd: '/work/alpha', newSession: true })
    expect(moveSessionToProject).not.toHaveBeenCalled()
    expect(onClose).toHaveBeenCalled()
  })

  it('starts a new chat only at an explicitly chosen multi-root folder', async () => {
    renderMenu({ tree: [MULTI] })

    await openProjectSub('Multi')

    // The same folder appears twice: once under the switch list, once under
    // the new-chat list. The second one starts a new chat there.
    const folderItems = screen.getAllByRole('menuitem', { name: '/work/multi-api' })

    expect(folderItems).toHaveLength(2)
    expect(screen.getByText(p.chooseFolder)).not.toBeNull()

    fireEvent.click(folderItems[1])

    expect(goToProject).toHaveBeenCalledWith('p_multi', { cwd: '/work/multi-api', newSession: true })
    expect(moveSessionToProject).not.toHaveBeenCalled()
  })

  it('starts a detached new chat for the path-less Home bucket', async () => {
    renderMenu({ tree: [HOME] })

    const sub = await openProjectSub('Home')

    fireEvent.click(within(sub).getByRole('menuitem', { name: p.newChat }))

    expect(goToProject).toHaveBeenCalledWith('no-project', { newSession: true })
  })

  it('sets and unsets the default project for new chats without touching the chat or scope', async () => {
    renderMenu()

    const sub = await openProjectSub('Alpha')

    fireEvent.click(within(sub).getByRole('menuitem', { name: p.defaultForNewChats }))

    await waitFor(() => expect(setDefaultProject).toHaveBeenCalledWith('p_alpha'))
    expect(goToProject).not.toHaveBeenCalled()
    expect(moveSessionToProject).not.toHaveBeenCalled()

    // Second click on the default project clears the pointer.
    act(() => $defaultProjectId.set('p_alpha'))
    fireEvent.click(within(sub).getByRole('menuitem', { name: p.defaultForNewChats }))

    await waitFor(() => expect(setDefaultProject).toHaveBeenCalledWith(null))
  })

  it('marks the default project on its row', () => {
    $defaultProjectId.set('p_beta')
    renderMenu()

    const beta = screen
      .getAllByRole('menuitem')
      .find(el => el.getAttribute('data-slot') === 'dropdown-menu-sub-trigger' && within(el).queryByText('Beta'))

    expect(within(beta as HTMLElement).getByTitle(p.defaultForNewChats)).not.toBeNull()
  })

  it('surfaces a clear error when setting the default fails', async () => {
    const err = new Error('boom')
    setDefaultProject.mockRejectedValueOnce(err)
    renderMenu()

    const sub = await openProjectSub('Alpha')

    fireEvent.click(within(sub).getByRole('menuitem', { name: p.defaultForNewChats }))

    await waitFor(() => expect(notifyError).toHaveBeenCalledWith(err, p.setDefaultFailed))
  })
})

describe('WorkspaceSwitcherMenu — gates', () => {
  it('busy chat: move is disabled with guidance and never sent; new chat stays available', async () => {
    renderMenu({ busy: true })

    expect(screen.getByText(p.moveBusy)).not.toBeNull()

    const sub = await openProjectSub('Alpha')
    const switchItem = within(sub).getByRole('menuitem', { name: p.switchCurrentChat })

    expect(switchItem.getAttribute('aria-disabled')).toBe('true')

    fireEvent.click(switchItem)
    expect(moveSessionToProject).not.toHaveBeenCalled()

    fireEvent.click(within(sub).getByRole('menuitem', { name: p.newChat }))
    expect(goToProject).toHaveBeenCalledWith('p_alpha', { cwd: '/work/alpha', newSession: true })
  })

  it('stale backend: writes are disabled with the remedy, reads stay offered', async () => {
    renderMenu()
    act(() => $projectsRpcAvailable.set(false))

    expect(screen.getByText(p.staleBackend)).not.toBeNull()

    const sub = await openProjectSub('Alpha')

    expect(within(sub).getByRole('menuitem', { name: p.switchCurrentChat }).getAttribute('aria-disabled')).toBe(
      'true'
    )
    expect(within(sub).getByRole('menuitem', { name: p.defaultForNewChats }).getAttribute('aria-disabled')).toBe(
      'true'
    )

    fireEvent.click(within(sub).getByRole('menuitem', { name: p.switchCurrentChat }))
    fireEvent.click(within(sub).getByRole('menuitem', { name: p.defaultForNewChats }))
    expect(moveSessionToProject).not.toHaveBeenCalled()
    expect(setDefaultProject).not.toHaveBeenCalled()

    fireEvent.click(within(sub).getByRole('menuitem', { name: p.openProject }))
    expect(goToProject).toHaveBeenCalledWith('p_alpha')
  })

  it('remote or other-gateway focused session: no move is offered, open/new remain', async () => {
    renderMenu({ sessionRemote: true })

    const sub = await openProjectSub('Alpha')

    expect(within(sub).queryByRole('menuitem', { name: p.switchCurrentChat })).toBeNull()
    expect(within(sub).getByRole('menuitem', { name: p.newChat })).not.toBeNull()
    expect(moveSessionToProject).not.toHaveBeenCalled()
  })

  it('no stored session: no move is offered', async () => {
    renderMenu({ sessionId: null })

    const sub = await openProjectSub('Alpha')

    expect(within(sub).queryByRole('menuitem', { name: p.switchCurrentChat })).toBeNull()
  })
})

describe('WorkspaceSwitcherMenu — footer and existing workspace actions', () => {
  function LocationProbe() {
    const location = useLocation()

    return <div data-testid="location">{location.pathname + location.search}</div>
  }

  it('opens a folder as a project from the footer', () => {
    renderMenu()

    fireEvent.click(screen.getByRole('menuitem', { name: en.commandCenter.openFolder }))

    expect(openFolderAsProject).toHaveBeenCalledWith()
    expect(onClose).toHaveBeenCalled()
  })

  it('navigates to Settings → Workspace → Projects from the footer', () => {
    render(
      <MemoryRouter>
        <LocationProbe />
        <DropdownMenu open>
          <DropdownMenuTrigger asChild>
            <button type="button">chip</button>
          </DropdownMenuTrigger>
          <DropdownMenuContent>
            <WorkspaceSwitcherMenu
              busy={false}
              changeWorkspaceItem={{ id: 'change', label: 'Change', onSelect: () => {} }}
              copyPathItem={{ id: 'copy', label: 'Copy', onSelect: () => {} }}
              currentCwd="/work/alpha"
              currentProjectId={null}
              onClose={onClose}
              profile="default"
              projectsCopy={p}
              revealItems={[]}
              sessionId="sess-1"
              sessionRemote={false}
            />
          </DropdownMenuContent>
        </DropdownMenu>
      </MemoryRouter>
    )

    fireEvent.click(screen.getByRole('menuitem', { name: p.manageProjects }))

    expect(screen.getByTestId('location').textContent).toContain('/settings?tab=config:workspace&page=projects')
    expect(onClose).toHaveBeenCalled()
  })

  it('keeps change-working-directory, copy-path and reveal actions working', () => {
    const changeSelect = vi.fn()
    const copySelect = vi.fn()
    const revealSelect = vi.fn()

    render(
      <MemoryRouter>
        <DropdownMenu open>
          <DropdownMenuTrigger asChild>
            <button type="button">chip</button>
          </DropdownMenuTrigger>
          <DropdownMenuContent>
            <WorkspaceSwitcherMenu
              busy={false}
              changeWorkspaceItem={{ id: 'change-workspace-path', label: 'Change working directory', onSelect: changeSelect }}
              copyPathItem={{ id: 'copy-workspace-path', label: 'Copy Path', onSelect: copySelect }}
              currentCwd="/work/alpha"
              currentProjectId={null}
              onClose={onClose}
              profile="default"
              projectsCopy={p}
              revealItems={[{ id: 'reveal-workspace-finder', label: 'Reveal in File Manager', onSelect: revealSelect }]}
              sessionId="sess-1"
              sessionRemote={false}
            />
          </DropdownMenuContent>
        </DropdownMenu>
      </MemoryRouter>
    )

    fireEvent.click(screen.getByRole('menuitem', { name: 'Change working directory' }))
    fireEvent.click(screen.getByRole('menuitem', { name: 'Copy Path' }))
    fireEvent.click(screen.getByRole('menuitem', { name: 'Reveal in File Manager' }))

    expect(changeSelect).toHaveBeenCalled()
    expect(copySelect).toHaveBeenCalled()
    expect(revealSelect).toHaveBeenCalled()
  })

  it('hides the change-working-directory row when the item is hidden upstream', () => {
    render(
      <MemoryRouter>
        <DropdownMenu open>
          <DropdownMenuTrigger asChild>
            <button type="button">chip</button>
          </DropdownMenuTrigger>
          <DropdownMenuContent>
            <WorkspaceSwitcherMenu
              busy={false}
              changeWorkspaceItem={{ hidden: true, id: 'change-workspace-path', label: 'Change working directory', onSelect: () => {} }}
              copyPathItem={{ id: 'copy-workspace-path', label: 'Copy Path', onSelect: () => {} }}
              currentCwd="/work/alpha"
              currentProjectId={null}
              onClose={onClose}
              profile="default"
              projectsCopy={p}
              revealItems={[]}
              sessionId="sess-1"
              sessionRemote={false}
            />
          </DropdownMenuContent>
        </DropdownMenu>
      </MemoryRouter>
    )

    expect(screen.queryByRole('menuitem', { name: 'Change working directory' })).toBeNull()
    expect(screen.getByRole('menuitem', { name: 'Copy Path' })).not.toBeNull()
  })
})
