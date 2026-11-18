import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import type { ReactNode } from 'react'
import { MemoryRouter } from 'react-router'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { group, split } from '@/components/pane-shell/tree/model'
import { $layoutTree, noteActiveTreeGroup } from '@/components/pane-shell/tree/store'
import { RootTooltipProvider } from '@/components/ui/tooltip'
import { en } from '@/i18n/en'
import { $sidebarOpen, setSidebarOpen } from '@/store/layout'

import { usePageSidebarOpen } from '../../overlays/page-sidebar'
import { $widthClass } from '../../redesign/breakpoints'
import { $shellSheet } from '../../shell/shell-sheet'

import { SimplifiedGlobalNav } from './simplified-global-nav'

afterEach(() => {
  cleanup()
  $layoutTree.set(null)
  noteActiveTreeGroup(null)
})

/** A page that owns a sidebar panel (Agents, Schedules, Settings) is on screen. */
function PanelPage() {
  usePageSidebarOpen()

  return null
}

function mount(route = '/', page?: ReactNode) {
  const navigate = vi.fn()
  render(
    <MemoryRouter initialEntries={[route]}>
      <RootTooltipProvider>
        <SimplifiedGlobalNav onNavigate={navigate} />
        {page}
      </RootTooltipProvider>
    </MemoryRouter>
  )

  return navigate
}

it.each([
  [en.profiles.title, '/profiles'],
  [en.sidebar.nav.capabilities, '/capabilities'],
  [en.simplifiedAgents.library, '/artifacts'],
  [en.cron.title, '/cron']
])('opens %s using its existing destination', (label, route) => {
  const navigate = mount()
  fireEvent.click(screen.getByRole('button', { name: label }))
  expect(navigate).toHaveBeenCalledWith(expect.objectContaining({ route }))
})

it.each(['skills', 'plugins', 'connectors', 'toolsets'])(
  'keeps Capabilities active for its %s tab without a dropdown',
  tab => {
    const navigate = mount(`/capabilities?tab=${tab}`)
    const button = screen.getByRole('button', { name: 'Capabilities' })
    expect(button.getAttribute('aria-current')).toBe('page')
    expect(screen.queryByRole('button', { name: 'Skills' })).toBeNull()
    fireEvent.click(button)
    expect(navigate).toHaveBeenCalledWith(expect.objectContaining({ route: '/capabilities' }))
    expect(screen.queryByRole('menu')).toBeNull()
  }
)

it.each([
  ['/settings?agentProfile=research&panel=settings&tab=config:model', en.profiles.title],
  ['/settings?tab=config:appearance', 'Settings']
])('marks the owning rail destination on %s', (route, label) => {
  mount(route)
  expect(screen.getByRole('button', { current: 'page' }).getAttribute('aria-label')).toBe(label)
})

it('opens Settings on its first section, with no menu in between', () => {
  const navigate = mount()
  fireEvent.click(screen.getByRole('button', { name: 'Settings' }))
  expect(navigate).toHaveBeenCalledWith(expect.objectContaining({ route: '/settings?tab=billing' }))
  expect(screen.queryByRole('menu')).toBeNull()
})

it('follows the focused chat tile instead of the workspace page underneath it', () => {
  $layoutTree.set(
    split('row', [
      group(['workspace'], { id: 'workspace-group' }),
      group(['session-tile:chat-one'], { id: 'chat-group' })
    ])
  )
  noteActiveTreeGroup('workspace-group')
  mount('/capabilities?tab=skills')
  const capabilities = screen.getByRole('button', { name: 'Capabilities' })
  const chats = screen.getByRole('button', { name: 'Chats' })
  expect(capabilities.getAttribute('aria-current')).toBe('page')
  act(() => noteActiveTreeGroup('chat-group'))
  expect(chats.getAttribute('aria-current')).toBe('page')
  expect(capabilities.hasAttribute('aria-current')).toBe(false)
  act(() => noteActiveTreeGroup('workspace-group'))
  expect(capabilities.getAttribute('aria-current')).toBe('page')
  expect(chats.hasAttribute('aria-current')).toBe(false)
})

describe('the open destination folds its own panel (Figma 1054:20985 / 1054:21668)', () => {
  afterEach(() => {
    act(() => setSidebarOpen(true))
    $widthClass.set('wide')
  })

  async function tooltipOf(button: HTMLElement) {
    fireEvent.pointerMove(button, { pointerType: 'mouse' })

    return (await screen.findByRole('tooltip')).textContent
  }

  it('hides and shows the chat sidebar from the open Chats button, which stays the open destination', async () => {
    $widthClass.set('wide')
    const navigate = mount('/')
    const chats = screen.getByRole('button', { name: 'Chats' })

    expect(chats.getAttribute('aria-expanded')).toBe('true')
    expect(await tooltipOf(chats)).toContain(en.simplifiedAgents.hideChats)

    fireEvent.click(chats)
    expect($sidebarOpen.get()).toBe(false)
    expect(chats.getAttribute('aria-expanded')).toBe('false')
    expect(chats.getAttribute('aria-current')).toBe('page')
    expect(navigate).not.toHaveBeenCalled()

    fireEvent.click(chats)
    expect($sidebarOpen.get()).toBe(true)
    expect(navigate).not.toHaveBeenCalled()
  })

  it('names the hidden sidebar "Show chats"', async () => {
    $widthClass.set('wide')
    act(() => setSidebarOpen(false))
    mount('/')

    expect(await tooltipOf(screen.getByRole('button', { name: 'Chats' }))).toContain(en.simplifiedAgents.showChats)
  })

  it('navigates to the chats from another destination and brings a folded sidebar back with them', () => {
    $widthClass.set('wide')
    act(() => setSidebarOpen(false))
    const navigate = mount('/artifacts')
    const chats = screen.getByRole('button', { name: 'Chats' })

    expect(chats.hasAttribute('aria-expanded')).toBe(false)
    fireEvent.click(chats)
    expect(navigate).toHaveBeenCalledWith(expect.objectContaining({ route: '/' }))
    expect($sidebarOpen.get()).toBe(true)
  })

  it('leaves a folded sidebar alone when the pick has no panel', () => {
    $widthClass.set('wide')
    act(() => setSidebarOpen(false))
    const navigate = mount('/')

    fireEvent.click(screen.getByRole('button', { name: 'Library' }))
    expect(navigate).toHaveBeenCalledWith(expect.objectContaining({ route: '/artifacts' }))
    expect($sidebarOpen.get()).toBe(false)
  })

  it('folds an open page panel from its destination, with the panel copy', async () => {
    $widthClass.set('wide')
    const navigate = mount('/profiles', <PanelPage />)
    const agents = screen.getByRole('button', { name: en.profiles.title })

    expect(agents.getAttribute('aria-expanded')).toBe('true')
    expect(await tooltipOf(agents)).toContain(en.simplifiedAgents.hidePanel)
    fireEvent.click(agents)
    expect($sidebarOpen.get()).toBe(false)
    expect(navigate).not.toHaveBeenCalled()
  })

  it('opens the page panel as a sheet below wide', () => {
    $widthClass.set('medium')
    mount('/cron', <PanelPage />)
    const schedules = screen.getByRole('button', { name: en.cron.title })

    expect(schedules.getAttribute('aria-expanded')).toBe('false')
    fireEvent.click(schedules)
    expect($shellSheet.get()).toBe('page')
    expect($sidebarOpen.get()).toBe(true)
    act(() => $shellSheet.set(null))
  })

  it('re-picks a destination without a panel', () => {
    const navigate = mount('/capabilities')
    const capabilities = screen.getByRole('button', { name: 'Capabilities' })

    expect(capabilities.hasAttribute('aria-expanded')).toBe(false)
    fireEvent.click(capabilities)
    expect(navigate).toHaveBeenCalledWith(expect.objectContaining({ route: '/capabilities' }))
  })
})
