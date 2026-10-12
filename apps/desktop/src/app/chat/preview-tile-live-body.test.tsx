import { act, cleanup, render } from '@testing-library/react'
import { createElement, useEffect } from 'react'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import { group } from '@/components/pane-shell/tree/model'
import { LayoutTreeRoot } from '@/components/pane-shell/tree/renderer'
import { declareDefaultTree, treePanesWithPrefix, watchContributedPanes } from '@/components/pane-shell/tree/store'
import { registry } from '@/contrib/registry'
import { $previewTabs, openPreview } from '@/store/preview'
import { $selectedStoredSessionId } from '@/store/session'
import { stubMenuDomApis } from '@/test/jsdom'

import { watchPreviewTiles } from './preview-tile'

// A native guest has a lifetime beyond React: a remount, or any detach of its
// element, destroys the Electron guest (unsaved input, page JS state). Count
// both, on a stand-in for the pane body.
const disconnected: HTMLElement[] = []
const mounts = new Map<string, number>()

class LiveGuest extends HTMLElement {
  disconnectedCallback() {
    disconnected.push(this)
  }
}

if (!customElements.get('preview-live-guest')) {
  customElements.define('preview-live-guest', LiveGuest)
}

vi.mock('./right-rail/preview', () => ({
  PreviewTilePane: ({ tabId }: { tabId: string }) => {
    useEffect(() => {
      mounts.set(tabId, (mounts.get(tabId) ?? 0) + 1)
    }, [tabId])

    return createElement('preview-live-guest', { 'data-tab': tabId }, createElement('input', { defaultValue: '' }))
  }
}))

vi.mock('./right-rail/preview-console-store', () => ({
  forgetPreviewConsole: () => undefined
}))

class ResizeObserverStub {
  observe() {}
  unobserve() {}
  disconnect() {}
}

beforeEach(() => {
  disconnected.length = 0
  mounts.clear()
  vi.stubGlobal('ResizeObserver', ResizeObserverStub)
  vi.stubGlobal('CSS', { ...globalThis.CSS, escape: (value: string) => value })
  stubMenuDomApis()
})

afterEach(() => {
  cleanup()
  vi.unstubAllGlobals()
})

// The behaviour a user sees: switching A -> B -> A shows A's Browser with the
// page they left — the same guest, never torn down in between — and B's drawer
// does not show it.
it("keeps a hidden session's live Browser body alive and shows the same one on return", () => {
  registry.register({
    area: 'panes',
    data: { placement: 'main', uncloseable: true },
    id: 'workspace',
    render: () => null,
    title: 'workspace'
  })
  declareDefaultTree(group(['workspace'], { active: 'workspace', id: 'grp-main' }))
  watchContributedPanes()
  watchPreviewTiles()

  $selectedStoredSessionId.set('sess-live-a')
  act(() => openPreview({ kind: 'url', label: 'A', source: 'https://a.example', url: 'https://a.example' }))
  const aTab = $previewTabs.get()[0]!.id

  const { container } = render(createElement(LayoutTreeRoot))

  const guestFor = (tabId: string) => container.querySelector<HTMLElement>(`preview-live-guest[data-tab="${tabId}"]`)

  const aGuest = guestFor(aTab)!
  const input = aGuest.querySelector('input')!
  input.value = 'unsaved in A'

  act(() => $selectedStoredSessionId.set('sess-live-b'))
  expect(treePanesWithPrefix('preview-tile:')).toEqual([])
  expect(aGuest.isConnected).toBe(true)
  expect(aGuest.closest('[inert]')).not.toBeNull()

  act(() => $selectedStoredSessionId.set('sess-live-a'))
  expect(treePanesWithPrefix('preview-tile:')).toEqual([`preview-tile:${aTab}`])
  expect(guestFor(aTab)).toBe(aGuest)
  expect(aGuest.closest('[inert]')).toBeNull()
  expect(input.value).toBe('unsaved in A')
  expect(disconnected).not.toContain(aGuest)
  expect(mounts.get(aTab)).toBe(1)
})
