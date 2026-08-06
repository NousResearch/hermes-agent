import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

const closeFocusedSessionTab = vi.fn(() => false)
const closeFocusedToolTab = vi.fn(() => false)
const nextSessionTileForWorkspace = vi.fn<() => null | string>(() => null)
const closeSessionTile = vi.fn()
const requestFreshSession = vi.fn()

vi.mock('@/components/pane-shell/tree/store', () => ({
  closeFocusedSessionTab: () => closeFocusedSessionTab(),
  closeFocusedToolTab: () => closeFocusedToolTab()
}))

vi.mock('@/store/session-states', () => ({
  closeSessionTile: (...args: unknown[]) => closeSessionTile(...args),
  nextSessionTileForWorkspace: () => nextSessionTileForWorkspace()
}))

vi.mock('@/store/profile', () => ({
  requestFreshSession: () => requestFreshSession()
}))

import { $previewTabs, closeRightRail, openPreview, type PreviewTarget } from '@/store/preview'
import { $activeSessionId, $selectedStoredSessionId } from '@/store/session'

import { $workspaceIsPage } from '../routes'

import { closeActiveTab, closeWorkspaceTab } from './close-tab'

function fileTarget(path: string): PreviewTarget {
  return {
    kind: 'file',
    label: path,
    path,
    previewKind: 'text',
    source: path,
    url: `file://${path}`
  }
}

/** Main is holding a loaded chat and nothing else is stacked with it. */
function loadedMainOnly() {
  $selectedStoredSessionId.set('stored-a')
  $activeSessionId.set('runtime-a')
}

beforeEach(() => {
  vi.stubGlobal('document', { activeElement: null })
  closeRightRail()
  window.localStorage.clear()
  $selectedStoredSessionId.set(null)
  $activeSessionId.set(null)
  $workspaceIsPage.set(false)
  closeFocusedSessionTab.mockReturnValue(false)
  closeFocusedToolTab.mockReturnValue(false)
  nextSessionTileForWorkspace.mockReturnValue(null)
  vi.clearAllMocks()
})

afterEach(() => {
  vi.unstubAllGlobals()
  closeRightRail()
  window.localStorage.clear()
})

describe('closeActiveTab', () => {
  // Preview tabs are layout-tree panes now, so ⌘W reaches them through the
  // focused-zone rungs (closeFocusedSessionTab / closeFocusedToolTab → the
  // pane's registered closer) rather than a rail-shaped special case. Open
  // previews must therefore NOT claim the key on their own.
  it('leaves ⌘W to the zone rungs even with previews open', () => {
    openPreview(fileTarget('/work/notes.md'), 'manual')
    closeFocusedToolTab.mockReturnValue(true)

    expect($previewTabs.get()).toHaveLength(1)
    expect(closeActiveTab()).toBe(true)
    // The zone closed its own tab; the rail store was never consulted.
    expect($previewTabs.get()).toHaveLength(1)
  })
})

/**
 * The main tab's own close. The workspace pane can never leave the tree, so
 * every answer here is about what FILLS it — a stacked session, or an empty
 * draft. The gesture used to dead-end whenever main was the only tab.
 */
describe('closeWorkspaceTab', () => {
  it('shifts the next stacked session into main', () => {
    loadedMainOnly()
    nextSessionTileForWorkspace.mockReturnValue('stored-b')
    const load = vi.fn()

    expect(closeWorkspaceTab(load)).toBe(true)
    expect(closeSessionTile).toHaveBeenCalledWith('stored-b')
    expect(load).toHaveBeenCalledWith('stored-b')
    // Promotion refills main — it must not ALSO blank it.
    expect(requestFreshSession).not.toHaveBeenCalled()
  })

  it('drops a lone loaded main to a fresh draft', () => {
    loadedMainOnly()

    expect(closeWorkspaceTab(vi.fn())).toBe(true)
    expect(requestFreshSession).toHaveBeenCalledTimes(1)
  })

  it('empties main even with no session loader wired', () => {
    loadedMainOnly()

    expect(closeWorkspaceTab()).toBe(true)
    expect(requestFreshSession).toHaveBeenCalledTimes(1)
  })

  it('is a no-op on a blank draft — that IS the post-close state', () => {
    expect(closeWorkspaceTab(vi.fn())).toBe(false)
    expect(requestFreshSession).not.toHaveBeenCalled()
  })

  it('is a no-op over a full-page view, which owns no chat tab', () => {
    loadedMainOnly()
    $workspaceIsPage.set(true)

    expect(closeWorkspaceTab(vi.fn())).toBe(false)
    expect(requestFreshSession).not.toHaveBeenCalled()
  })

  it('⌘W reaches it once the terminal and zone tabs pass', () => {
    loadedMainOnly()

    expect(closeActiveTab(vi.fn())).toBe(true)
    expect(requestFreshSession).toHaveBeenCalledTimes(1)
  })

  it('a focused tool panel (terminal / logs) claims ⌘W before main empties', () => {
    loadedMainOnly()
    closeFocusedToolTab.mockReturnValue(true)

    expect(closeActiveTab(vi.fn())).toBe(true)
    // The logs/terminal tab closed — main keeps its loaded chat.
    expect(requestFreshSession).not.toHaveBeenCalled()
  })
})
