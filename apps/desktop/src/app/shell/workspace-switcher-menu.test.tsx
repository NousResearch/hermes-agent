import { act, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { MemoryRouter, useLocation } from 'react-router'
import { afterEach, beforeAll, describe, expect, it, vi } from 'vitest'

import type { SidebarProjectTree } from '@/app/chat/sidebar/projects/workspace-groups'
import { Codicon } from '@/components/ui/codicon'
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
    icon: <Codicon name="folder" size="0.75rem" />,
    id: 'change-workspace-path',
    label: 'Change working directory',
    onSelect: () => {}
  }

  const copyPathItem: StatusbarMenuItem = {
    icon: <Codicon name="copy" size="0.75rem" />,
    id: 'copy-workspace-path',
    label: 'Copy Path',
    onSelect: () => {}
  }

  const revealItems: StatusbarMenuItem[] = [
    { icon: <Codicon name="folder-opened" size="0.75rem" />, id: 'reveal-workspace-finder', label: 'Reveal in File Manager', onSelect: () => {} },
    { icon: <Codicon name="files" size="0.75rem" />, id: 'reveal-workspace-sidebar', label: 'Reveal in Sidebar', onSelect: () => {} }
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
  return openNestedSub(screen.getAllByRole('menu')[0] as HTMLElement, label, 2)
}

/** Open a nested submenu carrying `label` inside `container`. */
async function openNestedSub(container: HTMLElement, label: string, minMenus = 3): Promise<HTMLElement> {
  const trigger = within(container)
    .getAllByRole('menuitem')
    .find(el => el.getAttribute('data-slot') === 'dropdown-menu-sub-trigger' && within(el).queryByText(label))

  expect(trigger).toBeDefined()

  fireEvent.pointerMove(trigger as HTMLElement, { pointerType: 'mouse' })
  fireEvent.focus(trigger as HTMLElement)

  // Radix portals each submenu into its own role="menu" surface — wait for the
  // next one and use the newest.
  await waitFor(() => expect(screen.getAllByRole('menu').length).toBeGreaterThanOrEqual(minMenus))

  const menus = screen.getAllByRole('menu')

  return menus[menus.length - 1] as HTMLElement
}

/** The wrapper row for a project node (star toggle + submenu trigger). */
function projectRow(id: string): HTMLElement {
  return screen.getByTestId(`project-row-${id}`)
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

  it('bounds the project glyph in the shared icon slot so the label keeps flex-1 space', () => {
    renderMenu()

    const alpha = screen
      .getAllByRole('menuitem')
      .find(el => el.getAttribute('data-slot') === 'dropdown-menu-sub-trigger' && within(el).queryByText('Alpha'))

    expect(alpha).toBeDefined()

    const slot = (alpha as HTMLElement).querySelector('[data-icon-slot="true"]')

    expect(slot).not.toBeNull()

    const label = within(alpha as HTMLElement).getByText('Alpha').closest('span.flex-1')

    expect(label).not.toBeNull()
    expect(label?.className).toContain('min-w-0')
    expect(label?.className).toContain('flex-1')
  })

  it('renders every project lead as a direct 0.75rem Codicon in the bounded slot — no overview-row glyph', () => {
    renderMenu({ tree: [ALPHA, HOME, proj({ id: 'auto-1', isAuto: true, label: 'Auto', path: '/work/auto' })] })

    const cases: Array<[string, string, string]> = [
      ['Alpha', 'codicon-folder-library', '#f00'],
      ['Home', 'codicon-home', ''],
      ['Auto', 'codicon-repo', '']
    ]

    for (const [label, cls, color] of cases) {
      const trigger = screen
        .getAllByRole('menuitem')
        .find(el => el.getAttribute('data-slot') === 'dropdown-menu-sub-trigger' && within(el).queryByText(label))

      expect(trigger).toBeDefined()

      const slot = (trigger as HTMLElement).querySelector('[data-icon-slot="true"]')

      expect(slot).not.toBeNull()

      const icon = slot?.querySelector('i.codicon')

      expect(icon).not.toBeNull()
      expect(icon?.classList.contains(cls)).toBe(true)
      expect((icon as HTMLElement).style.fontSize).toBe('0.75rem')

      if (color) {
        expect((icon as HTMLElement).style.color).toBe('rgb(255, 0, 0)')
      }
    }
  })

  it('tints the folder-library glyph for a color-only project instead of a bare dot', () => {
    renderMenu()

    const alpha = screen
      .getAllByRole('menuitem')
      .find(el => el.getAttribute('data-slot') === 'dropdown-menu-sub-trigger' && within(el).queryByText('Alpha'))

    const slot = (alpha as HTMLElement).querySelector('[data-icon-slot="true"]')

    // No color dot: the color tints a real folder-library Codicon.
    expect(slot?.querySelector('.size-1.rounded-full')).toBeNull()
    expect(slot?.querySelector('i.codicon-folder-library')).not.toBeNull()
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

    // No flat folder list and no silent primary-fallback move: "Switch current
    // chat" is a labelled nested submenu (a subtrigger, not a direct action).
    const flatSwitch = within(sub)
      .getAllByRole('menuitem', { name: p.switchCurrentChat })
      .find(el => el.getAttribute('data-slot') !== 'dropdown-menu-sub-trigger')

    expect(flatSwitch).toBeUndefined()
    expect(within(sub).queryByText(p.chooseFolder)).toBeNull()

    const switchSub = await openNestedSub(sub, p.switchCurrentChat)

    fireEvent.click(within(switchSub).getByRole('menuitem', { name: '/work/multi-api' }))

    await waitFor(() =>
      expect(moveSessionToProject).toHaveBeenCalledWith('sess-1', 'p_multi', 'default', '/work/multi-api')
    )
  })

  it('offers no move target for the path-less Home bucket', async () => {
    renderMenu({ tree: [HOME] })

    const sub = await openProjectSub('Home')

    expect(within(sub).queryByRole('menuitem', { name: p.switchCurrentChat })).toBeNull()
  })

  it('omits the move for the project the chat already occupies, keeping open/new and the submenu default star', async () => {
    renderMenu({ currentProjectId: 'p_alpha' })

    const sub = await openProjectSub('Alpha')

    expect(within(sub).queryByRole('menuitem', { name: p.switchCurrentChat })).toBeNull()
    expect(within(sub).getByRole('menuitem', { name: p.openProject })).not.toBeNull()
    expect(within(sub).getByRole('menuitem', { name: p.newChat })).not.toBeNull()
    // The default toggle lives at the bottom of the submenu, after a separator.
    expect(within(sub).getByRole('menuitem', { name: p.defaultForNewChats })).not.toBeNull()
    // The parent row is only a submenu trigger — no star menuitem on it.
    expect(within(projectRow('p_alpha')).queryByRole('menuitem', { name: p.defaultForNewChats })).toBeNull()
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

    const sub = await openProjectSub('Multi')

    // Each folder appears exactly once — under the labelled "New chat" nested
    // submenu — not duplicated across two flat lists.
    const newChatSub = await openNestedSub(sub, p.newChat)

    expect(within(sub).queryByText(p.chooseFolder)).toBeNull()
    expect(within(newChatSub).getAllByRole('menuitem', { name: '/work/multi-api' })).toHaveLength(1)

    fireEvent.click(within(newChatSub).getByRole('menuitem', { name: '/work/multi-api' }))

    expect(goToProject).toHaveBeenCalledWith('p_multi', { cwd: '/work/multi-api', newSession: true })
    expect(moveSessionToProject).not.toHaveBeenCalled()
  })

  it('starts a detached new chat for the path-less Home bucket', async () => {
    renderMenu({ tree: [HOME] })

    const sub = await openProjectSub('Home')

    fireEvent.click(within(sub).getByRole('menuitem', { name: p.newChat }))

    expect(goToProject).toHaveBeenCalledWith('no-project', { newSession: true })
  })

  it('sets and unsets the default project from the bottom-of-submenu star without touching the chat or scope', async () => {
    renderMenu()

    const sub = await openProjectSub('Alpha')

    fireEvent.click(within(sub).getByRole('menuitem', { name: p.defaultForNewChats }))

    await waitFor(() => expect(setDefaultProject).toHaveBeenCalledWith('p_alpha'))
    expect(goToProject).not.toHaveBeenCalled()
    expect(moveSessionToProject).not.toHaveBeenCalled()

    // Second click on the default project clears the pointer.
    act(() => $defaultProjectId.set('p_alpha'))
    const subAgain = await openProjectSub('Alpha')

    fireEvent.click(within(subAgain).getByRole('menuitem', { name: p.defaultForNewChats }))

    await waitFor(() => expect(setDefaultProject).toHaveBeenCalledWith(null))
  })

  it('marks the default project in the submenu star with an explicit pressed state and star-full glyph', async () => {
    $defaultProjectId.set('p_beta')
    renderMenu()

    const betaSub = await openProjectSub('Beta')
    const alphaSub = await openProjectSub('Alpha')

    const betaStar = within(betaSub).getByRole('menuitem', { name: p.defaultForNewChats })
    const alphaStar = within(alphaSub).getByRole('menuitem', { name: p.defaultForNewChats })

    expect(betaStar.getAttribute('aria-pressed')).toBe('true')
    expect(alphaStar.getAttribute('aria-pressed')).toBe('false')
    expect(betaStar.querySelector('i.codicon-star-full')).not.toBeNull()
    expect(alphaStar.querySelector('i.codicon-star-empty')).not.toBeNull()
    expect((betaStar.querySelector('i.codicon') as HTMLElement).style.fontSize).toBe('0.75rem')
    expect((alphaStar.querySelector('i.codicon') as HTMLElement).style.fontSize).toBe('0.75rem')
  })

  it('places the default star after a separator as the last action in the project submenu', async () => {
    renderMenu()

    const sub = await openProjectSub('Alpha')
    const defaultItem = within(sub).getByRole('menuitem', { name: p.defaultForNewChats })

    // Direct DOM order: the item directly follows a separator and is the last action.
    expect(defaultItem.previousElementSibling?.getAttribute('role')).toBe('separator')

    const actions = within(sub)
      .getAllByRole('menuitem')
      .filter(el => el.getAttribute('data-slot') !== 'dropdown-menu-sub-trigger')

    expect(actions[actions.length - 1]).toBe(defaultItem)
    // The separator exists only to set the default star apart from the project actions.
    expect(within(sub).getAllByRole('separator')).toHaveLength(1)
  })

  it('shows no separator or default item in the submenu of synthetic Home or auto-discovered projects', async () => {
    renderMenu({ tree: [HOME, proj({ id: 'auto-1', isAuto: true, label: 'Auto', path: '/work/auto' })] })

    const homeSub = await openProjectSub('Home')
    const autoSub = await openProjectSub('Auto')

    for (const sub of [homeSub, autoSub]) {
      expect(within(sub).queryByRole('menuitem', { name: p.defaultForNewChats })).toBeNull()
      expect(within(sub).queryByRole('separator')).toBeNull()
    }
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

    // The submenu default toggle is the default control; it is disabled when stale.
    const star = within(sub).getByRole('menuitem', { name: p.defaultForNewChats })

    expect(star.getAttribute('aria-disabled')).toBe('true')
    expect(star.getAttribute('title')).toBe(p.staleBackend)

    fireEvent.click(within(sub).getByRole('menuitem', { name: p.switchCurrentChat }))
    fireEvent.click(star)
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
  it('gives every action and footer row a visible label and a left-aligned leading icon slot', async () => {
    renderMenu()

    const footerLabels = [
      'Change working directory',
      'Copy Path',
      'Reveal in File Manager',
      'Reveal in Sidebar',
      en.commandCenter.openFolder,
      p.manageProjects
    ]

    for (const label of footerLabels) {
      const row = screen.getByRole('menuitem', { name: label })

      expect(within(row).queryByText(label)).not.toBeNull()
      expect(row.querySelector('[data-icon-slot]')).not.toBeNull()
    }

    // The per-project actions follow the same icon-slot contract.
    const sub = await openProjectSub('Alpha')

    for (const label of [p.switchCurrentChat, p.newChat, p.openProject]) {
      const row = within(sub).getByRole('menuitem', { name: label })

      expect(row.querySelector('[data-icon-slot]')).not.toBeNull()
    }
  })

  it('paints each middle action with its own 0.75rem Codicon in the bounded slot', () => {
    renderMenu()

    const cases: Array<[string, string]> = [
      ['Change working directory', 'codicon-folder'],
      ['Copy Path', 'codicon-copy'],
      ['Reveal in File Manager', 'codicon-folder-opened'],
      ['Reveal in Sidebar', 'codicon-files']
    ]

    for (const [label, cls] of cases) {
      const row = screen.getByRole('menuitem', { name: label })
      const icon = row.querySelector(`[data-icon-slot] i.codicon.${cls}`)

      expect(icon, label).not.toBeNull()
      expect((icon as HTMLElement).style.fontSize).toBe('0.75rem')
    }
  })

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
