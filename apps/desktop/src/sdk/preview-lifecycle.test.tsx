import { useStore } from '@nanostores/react'
import { act, cleanup, fireEvent, render } from '@testing-library/react'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import { PreviewTilePane } from '@/app/chat/right-rail/preview'
import { $poppedBrowserTabIds, $previewTabs, newBrowserTab, openPreview, popOutBrowserTab } from '@/store/preview'
import {
  $realProfilePromptClaim,
  $realProfilePromptDismissed,
  $realProfilePromptMuted
} from '@/store/real-profile-consent'
import { $sessionTiles } from '@/store/session-states'

import { openPluginPreview } from './preview'

const config = vi.hoisted(() => ({ data: { browser: { use_real_profile: false } }, save: vi.fn(), cache: vi.fn() }))
vi.mock('@/app/hooks/use-config-record', () => ({
  useHermesConfigRecord: () => ({ data: config.data }),
  hermesConfigCacheWriter: () => config.cache
}))
vi.mock('@/hermes', async importOriginal => ({
  ...(await importOriginal()),
  saveHermesConfigRecord: config.save
}))

const session = { runtimeSessionId: 'runtime', storedSessionId: 'stored', connectionId: 'local', profile: 'worker' }

function Previews() {
  const tabs = useStore($previewTabs)

  return tabs.map(tab => <PreviewTilePane key={tab.id} tabId={tab.id} />)
}

beforeEach(() => {
  $previewTabs.set([])
  $realProfilePromptClaim.set(null)
  $realProfilePromptDismissed.set(false)
  $realProfilePromptMuted.set(false)
  config.data = { browser: { use_real_profile: false } }
  $sessionTiles.set([
    { storedSessionId: 'stored', runtimeId: 'runtime', ownerRoute: { connectionId: 'local', profile: 'worker' } }
  ])
})
afterEach(() => {
  cleanup()
  $previewTabs.set([])
  $sessionTiles.set([])
  vi.clearAllMocks()
})

it('keeps isolated viewers separate from ordinary transient browsers and never hands their tickets to persistent popouts', async () => {
  const openBrowserWindow = vi.fn(async () => ({ ok: true }))
  const previousDesktop = window.hermesDesktop
  window.hermesDesktop = { ...previousDesktop, openBrowserWindow }

  try {
    const ordinary = {
      kind: 'url' as const,
      label: 'Browser',
      source: 'https://example.org',
      url: 'https://example.org',
      transient: true
    }

    openPreview(ordinary)
    const ordinaryTab = $previewTabs.get()[0]!
    const view = render(<Previews />)
    expect(view.getByText('Stay signed in to your sites')).toBeTruthy()
    expect(view.container.querySelector('webview')?.getAttribute('partition')).toBe('persist:hermes-preview')
    act(() => fireEvent.click(view.getByRole('button', { name: 'Not now' })))
    const url = 'http://127.0.0.1:9876/viewer?ticket=secret'
    await act(async () => {
      await openPluginPreview({ url, session })
    })
    expect($previewTabs.get()).toHaveLength(2)
    expect($previewTabs.get()[0]).toBe(ordinaryTab)
    const viewer = $previewTabs.get()[1]!
    const guest = view.container.querySelectorAll('webview')[1]!
    expect(view.queryByRole('button', { name: 'Pop out' })).toBeNull()
    act(() => popOutBrowserTab(viewer.id))
    expect(openBrowserWindow).not.toHaveBeenCalled()
    expect($poppedBrowserTabIds.get().has(viewer.id)).toBe(false)
    act(() => openPreview({ ...ordinary, url: 'https://example.org/two', transient: false }))
    expect($previewTabs.get()).toHaveLength(2)
    expect($previewTabs.get()[1]).toBe(viewer)
    expect(view.container.querySelectorAll('webview')[1]).toBe(guest)
    await act(async () => {
      await openPluginPreview({ url, session })
    })
    expect($previewTabs.get()).toHaveLength(2)
    expect(view.container.querySelectorAll('webview')[1]).toBe(guest)
    expect(window.localStorage.getItem('hermes.desktop.previewTabs.v2')).not.toContain('secret')
    expect(config.save).not.toHaveBeenCalled()
  } finally {
    window.hermesDesktop = previousDesktop
  }
})

it('first-open plugin viewer loads without profile consent and keeps its guest through unrelated consent changes', async () => {
  const view = render(<Previews />)
  const url = 'http://127.0.0.1:9876/viewer?ticket=secret'
  await act(async () => {
    expect(await openPluginPreview({ url, session })).toBe(true)
  })
  expect(view.queryByText('Stay signed in to your sites')).toBeNull()
  expect($realProfilePromptClaim.get()).toBeNull()
  const tab = $previewTabs.get()[0]!
  const guest = view.container.querySelector('webview')!
  expect(guest).not.toBeNull()
  expect(guest.getAttribute('src')).toBe(url)
  expect(guest.getAttribute('partition')).toMatch(/^hermes-viewer-/)
  expect(guest.getAttribute('preload')).toBeNull()
  expect(guest.getAttribute('webpreferences')).toBe('contextIsolation=yes,nodeIntegration=no,sandbox=yes')

  // An ordinary Browser still offers consent, even with a viewer already open.
  act(() => newBrowserTab())
  expect(view.getByText('Stay signed in to your sites')).toBeTruthy()
  act(() => fireEvent.click(view.getByRole('button', { name: 'Not now' })))
  expect(view.queryByText('Stay signed in to your sites')).toBeNull()
  expect($previewTabs.get().find(item => item.id === tab.id)).toBe(tab)
  expect(view.container.querySelector('webview')).toBe(guest)
  act(() => $realProfilePromptDismissed.set(true))
  config.data = { browser: { use_real_profile: true } }
  view.rerender(<Previews />)
  expect(view.container.querySelector('webview')).toBe(guest)
  expect(guest.getAttribute('partition')).toMatch(/^hermes-viewer-/)
  expect(window.localStorage.getItem('hermes.desktop.previewTabs.v2')).not.toContain('secret')
  expect(config.save).not.toHaveBeenCalled()
  expect(config.cache).not.toHaveBeenCalled()
})
