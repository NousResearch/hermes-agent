import { afterEach, expect, it, vi } from 'vitest'

import {
  $browserPages,
  $dockedVisiblePreviewTabs,
  $previewTabs,
  commitBrowserTabLocation,
  noteBrowserPage,
  noteViewerDocument,
  reopenViewer,
  viewerUrlSpent
} from '@/store/preview'
import { $selectedStoredSessionId } from '@/store/session'
import { $sessionTiles } from '@/store/session-states'

import { host } from './index'

// A viewer bootstraps from a one-time `#ticket=` it strips from its own
// address on load, so the capability lives only in the document.
const session = { runtimeSessionId: 'runtime-a', storedSessionId: 'A', connectionId: 'local', profile: 'default' }
const base = 'http://127.0.0.1:41751/realms/r-fixture/view'
const ticketUrl = (n: number) => `${base}#ticket=${String(n).repeat(43)}`

afterEach(() => {
  $previewTabs.set([])
  $sessionTiles.set([])
  $browserPages.set({})
  $selectedStoredSessionId.set(null)
  vi.clearAllTimers()
  vi.useRealTimers()
})

function ownSessionA() {
  $sessionTiles.set([
    { storedSessionId: 'A', runtimeId: 'runtime-a', ownerRoute: { connectionId: 'local', profile: 'default' } }
  ])
  $selectedStoredSessionId.set('A')
}

it('a viewer belongs to the session that opened it and follows it in and out of the drawer', async () => {
  ownSessionA()
  expect(await host.openPreview({ url: ticketUrl(1), label: 'Realm', session })).toBe(true)
  const tab = $previewTabs.get().at(-1)!

  expect(tab.sessionId).toBe('A')
  expect(tab.pinned).toBe(false)
  expect($dockedVisiblePreviewTabs.get().map(item => item.id)).toEqual([tab.id])

  $selectedStoredSessionId.set('B')
  expect($dockedVisiblePreviewTabs.get()).toEqual([])
  expect($previewTabs.get().map(item => item.id)).toEqual([tab.id])

  $selectedStoredSessionId.set('A')
  expect($dockedVisiblePreviewTabs.get().map(item => item.id)).toEqual([tab.id])
})

it('a viewer unmounted by a session switch comes back on a fresh capability, never its stripped address', async () => {
  vi.useFakeTimers()
  ownSessionA()

  let minted = 1
  const renewals: number[] = []

  const open = async (): Promise<boolean> => {
    const ticket = minted++

    return host.openPreview({
      url: ticketUrl(ticket),
      label: 'Realm',
      session,
      onKeepAlive: async () => void renewals.push(ticket),
      onReopen: async () => void (await open())
    })
  }

  expect(await open()).toBe(true)
  const tab = $previewTabs.get().at(-1)!
  const first = { isLive: () => true }

  // The guest loads the ticket URL, then the page strips it from its address.
  noteViewerDocument(tab.id, ticketUrl(1))
  noteBrowserPage(tab.id, { title: 'Realm', url: ticketUrl(1), document: first })
  await vi.advanceTimersByTimeAsync(0)
  noteBrowserPage(tab.id, { title: 'Realm', url: base, document: first })
  expect(renewals).toEqual([1])

  // Away to B and past the hidden-page cap: the pane unmounts and its cleanup
  // hands the live (ticketless) address back to the tab.
  $selectedStoredSessionId.set('B')
  commitBrowserTabLocation(tab.id, base, 'Realm')
  first.isLive = () => false
  noteBrowserPage(tab.id, { title: 'Realm', url: base, document: undefined })

  // The tab still holds the capability URL its document already spent.
  $selectedStoredSessionId.set('A')
  const back = $previewTabs.get().find(item => item.id === tab.id)!
  expect(back.target.url).toBe(ticketUrl(1))
  expect(viewerUrlSpent(tab.id, back.target.url)).toBe(true)

  // The remounting pane asks the opener for a fresh one: same tab, same
  // owner, a new capability, and renewal follows the new document only.
  expect(await reopenViewer(tab.id)).toBe(true)
  const fresh = $previewTabs.get().find(item => item.id === tab.id)!
  expect($previewTabs.get()).toHaveLength(1)
  expect(fresh.sessionId).toBe('A')
  expect(fresh.target.url).toBe(ticketUrl(2))
  expect(viewerUrlSpent(tab.id, fresh.target.url)).toBe(false)

  noteViewerDocument(tab.id, ticketUrl(2))
  noteBrowserPage(tab.id, { title: 'Realm', url: ticketUrl(2), document: { isLive: () => true } })
  await vi.advanceTimersByTimeAsync(60_000)
  expect(renewals).toEqual([1, 2, 2])
})

it('an opener without a re-open keeps the old behaviour: no automatic reconnect', async () => {
  ownSessionA()
  expect(await host.openPreview({ url: ticketUrl(1), label: 'Realm', session })).toBe(true)
  const tab = $previewTabs.get().at(-1)!

  noteViewerDocument(tab.id, ticketUrl(1))
  expect(await reopenViewer(tab.id)).toBe(false)
  expect($previewTabs.get().find(item => item.id === tab.id)?.target.url).toBe(ticketUrl(1))
})

it('a failing re-open leaves the tab on its address', async () => {
  ownSessionA()

  expect(
    await host.openPreview({
      url: ticketUrl(1),
      label: 'Realm',
      session,
      onReopen: async () => {
        throw new Error('realm stopped')
      }
    })
  ).toBe(true)

  const tab = $previewTabs.get().at(-1)!
  expect(await reopenViewer(tab.id)).toBe(false)
  expect($previewTabs.get().find(item => item.id === tab.id)?.target.url).toBe(ticketUrl(1))
})
