import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { MemoryRouter, useNavigate } from 'react-router'
import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import { SimplifiedNavigationSurface } from '@/app/contrib/simplified-navigation-surface'
import { group, split } from '@/components/pane-shell/tree/model'
import { $hiddenTreePanes, $layoutTree, noteActiveTreeGroup } from '@/components/pane-shell/tree/store'
import { $routeCollapsedPanes } from '@/components/pane-shell/tree/store-route-collapse'
import { RootTooltipProvider } from '@/components/ui/tooltip'
import { $chatItemsOpen } from '@/store/chat-items'
import { $fileBrowserOpen } from '@/store/layout'
import { $rotatedSessionIds } from '@/store/preview-ownership'
import { $reviewOpen } from '@/store/review'
import { $selectedStoredSessionId } from '@/store/session'
import { $shellChoice } from '@/store/shell-variant'

import { $terminalTakeover } from '../right-sidebar/store'

import {
  $chatToolPanels,
  adoptDraftToolPanel,
  bindChatToolPanels,
  bindChatToolRouteCollapse
} from './chat-tool-panel'

const tools = { files: $fileBrowserOpen, review: $reviewOpen, terminal: $terminalTakeover, items: $chatItemsOpen }

const openTools = () =>
  Object.entries(tools)
    .filter(([, store]) => store.get())
    .map(([name]) => name)

let stops: (() => void)[] = []

function reset() {
  stops.forEach(stop => stop())
  stops = []
  act(() => {
    $selectedStoredSessionId.set(null)
    Object.values(tools).forEach(store => store.set(false))
    $chatToolPanels.set({})
    $rotatedSessionIds.set(new Map())
    $routeCollapsedPanes.set(new Set())
    $hiddenTreePanes.set(new Set())
    $layoutTree.set(null)
    noteActiveTreeGroup(null)
  })
}

beforeEach(() => {
  reset()
  $shellChoice.set(true)
})

afterEach(() => {
  cleanup()
  reset()
  $shellChoice.set(null)
})

const focus = (chat: null | string) => act(() => $selectedStoredSessionId.set(chat))

describe('each chat keeps its own tool panel (redesign)', () => {
  beforeEach(() => {
    focus('chat-a')
    stops.push(bindChatToolPanels())
  })

  it('comes back to the tools a chat had open, and a chat never opened stays closed', () => {
    act(() => $fileBrowserOpen.set(true))
    focus('chat-b')
    expect(openTools()).toEqual([])

    act(() => $terminalTakeover.set(true))
    focus('chat-a')
    expect(openTools()).toEqual(['files'])

    act(() => $fileBrowserOpen.set(false))
    focus('chat-b')
    expect(openTools()).toEqual(['terminal'])
    focus('chat-a')
    expect(openTools()).toEqual([])
  })

  it('remembers across a reload through the persisted map', () => {
    act(() => {
      $reviewOpen.set(true)
      $chatItemsOpen.set(true)
    })
    focus('chat-b')
    expect($chatToolPanels.get()).toEqual({ 'chat-a': ['review', 'items'] })
  })

  it('hands a new chat its panel on its first send, and the next new chat starts closed', () => {
    focus(null)
    expect(openTools()).toEqual([])
    act(() => $chatItemsOpen.set(true))
    adoptDraftToolPanel('chat-new')
    focus('chat-new')
    expect(openTools()).toEqual(['items'])
    focus(null)
    expect(openTools()).toEqual([])
  })

  it('keeps the panel through a compression rotation of the same conversation', () => {
    act(() => $fileBrowserOpen.set(true))
    act(() => $rotatedSessionIds.set(new Map([['chat-a', 'chat-a2']])))
    focus('chat-a2')
    expect(openTools()).toEqual(['files'])
    expect($chatToolPanels.get()).toEqual({ 'chat-a2': ['files'] })
  })

  it('leaves classic with one panel for the whole window', () => {
    act(() => $shellChoice.set(false))
    act(() => $fileBrowserOpen.set(true))
    focus('chat-b')
    expect(openTools()).toEqual(['files'])
  })
})

function Surface() {
  const navigate = useNavigate()

  return <SimplifiedNavigationSurface onNavigate={item => item.route && navigate(item.route)} />
}

function mount(route: string) {
  return render(
    <MemoryRouter initialEntries={[route]}>
      <RootTooltipProvider>
        <Surface />
      </RootTooltipProvider>
    </MemoryRouter>
  )
}

const chatWithTools = (extra: string[] = []) =>
  split('row', [group(['sessions']), group(['workspace']), group(['files', 'terminal', 'preview-tile:web', ...extra])])

describe('no page shows the chat tool panel (redesign)', () => {
  beforeEach(() => {
    stops.push(bindChatToolRouteCollapse())
    act(() => $layoutTree.set(chatWithTools()))
  })

  it('takes the tools off screen on a page without closing or unmounting them, and brings them back on the chat', () => {
    act(() => $fileBrowserOpen.set(true))
    mount('/')
    expect([...$routeCollapsedPanes.get()].filter(id => id !== 'sessions')).toEqual([])

    fireEvent.click(screen.getByRole('button', { name: 'Capabilities' }))
    expect($routeCollapsedPanes.get()).toEqual(new Set(['sessions', 'files', 'terminal', 'preview-tile:web']))
    // Not the user's hide: nothing is unmounted and the chat's choice stands.
    expect($hiddenTreePanes.get().size).toBe(0)
    expect($fileBrowserOpen.get()).toBe(true)

    // A tool that joins while the page is up (an agent's Browser tab) stays off screen too.
    act(() => $layoutTree.set(chatWithTools(['preview-tile:late'])))
    expect($routeCollapsedPanes.get().has('preview-tile:late')).toBe(true)

    fireEvent.click(screen.getByRole('button', { name: 'Chats' }))
    expect($routeCollapsedPanes.get()).toEqual(new Set())
  })

  it('leaves classic pages alone', () => {
    $shellChoice.set(false)
    mount('/capabilities')
    expect($routeCollapsedPanes.get()).toEqual(new Set())
  })
})
