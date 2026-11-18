import { act, cleanup, fireEvent, render, screen, within } from '@testing-library/react'
import { MemoryRouter, Route } from 'react-router'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { $apiRequestScope } from '@/api/client'
import { $cronJobs } from '@/store/cron'
import { bindDesktopMetrics, resetDesktopMetricsForTests } from '@/store/desktop-metrics'
import { $firstTimeSplash } from '@/store/first-time-splash'
import { $guidanceLedger } from '@/store/guidance-ledger'
import { $itemIndex } from '@/store/item-index'
import { $newVersionReady } from '@/store/new-version'
import { $traySeen } from '@/store/notification-tray'
import { $renameSplash } from '@/store/rename-splash'
import { $sessions, $sessionsLoadError, $sessionsLoading } from '@/store/session'
import { $shellChoice } from '@/store/shell-variant'
import type { SessionInfo } from '@/types/hermes'

import { discoveryCopy } from './discovery-copy'
import { NotificationBell } from './notification-tray'
import { SplashCoveredRoutes } from './splash-layer'

const copy = discoveryCopy('en')
const chat = { id: 'chat-1', title: 'Quarterly plan' } as SessionInfo

/** The redesigned shell's pieces the splash deals with: the rail (with its bell) and the chat area. */
function renderShell() {
  return render(
    <MemoryRouter>
      <nav data-slot="global-nav">
        <button data-nav-spot="artifacts" type="button">
          Library
        </button>
        <NotificationBell />
      </nav>
      <main>
        <SplashCoveredRoutes chat={<Route element={<textarea aria-label="Message" />} index />} pages={null} />
      </main>
    </MemoryRouter>
  )
}

const splash = () => screen.queryByRole('region', { name: copy.renameSplash.title })

beforeEach(() => {
  window.localStorage.clear()
  resetDesktopMetricsForTests()
  $apiRequestScope.set({ connectionId: 'gateway', profile: 'default' })
  bindDesktopMetrics(null, 'gateway|default')
  $shellChoice.set(true)
  $guidanceLedger.set({})
  $traySeen.set([])
  $renameSplash.set(null)
  $firstTimeSplash.set(null)
  $itemIndex.set({})
  $newVersionReady.set(false)
  $cronJobs.set([])
  $sessions.set([])
  $sessionsLoadError.set(false)
  $sessionsLoading.set(true)
})

afterEach(() => {
  cleanup()
  resetDesktopMetricsForTests()
})

describe('the rename splash in the shell', () => {
  it('appears for someone with prior chats once the list settles, over an inert chat', () => {
    renderShell()
    expect(splash()).toBeNull()

    act(() => {
      $sessions.set([chat])
      $sessionsLoading.set(false)
    })

    expect(splash()).not.toBeNull()
    expect(screen.getByLabelText('Message').closest('[inert]')).not.toBeNull()
    expect(screen.getByRole('button', { name: /Sessions, now Chats/ })).toBeTruthy()
  })

  it('never appears on a new install, even once its first chat exists', () => {
    renderShell()
    act(() => $sessionsLoading.set(false))
    act(() => $sessions.set([chat]))

    expect(splash()).toBeNull()
  })

  it('closes on any nav click, lets the click through, and does not come back', () => {
    const onNav = vi.fn()
    renderShell()
    screen.getByRole('button', { name: 'Library' }).addEventListener('click', onNav)
    act(() => {
      $sessions.set([chat])
      $sessionsLoading.set(false)
    })

    fireEvent.click(screen.getByRole('button', { name: 'Library' }))

    expect(splash()).toBeNull()
    expect(onNav).toHaveBeenCalledTimes(1)
    expect($guidanceLedger.get().renameSplash).toBe('seen')
    expect(screen.getByLabelText('Message').closest('[inert]')).toBeNull()
  })

  it('closes from its own button and from Escape', () => {
    renderShell()
    act(() => {
      $sessions.set([chat])
      $sessionsLoading.set(false)
    })

    fireEvent.click(screen.getByRole('button', { name: copy.renameSplash.done }))
    expect(splash()).toBeNull()

    act(() => $renameSplash.set('reopened'))
    fireEvent.keyDown(window, { key: 'Escape' })
    expect(splash()).toBeNull()
  })

  it('never shows a Library count while the index is still reading its chats, only once it knows', () => {
    // The list was checked a moment ago (so no new check starts), but the chats' items are still being
    // read: the count so far is not the count.
    const snapshot = { checkedAt: Date.now(), chats: {}, removed: {} }
    $itemIndex.set({ local: { snapshot, refreshing: true, error: null, failures: [] } })
    renderShell()
    act(() => {
      $sessions.set([chat])
      $sessionsLoading.set(false)
    })

    expect(screen.getByRole('button', { name: /Artifacts, now Library/ }).getAttribute('aria-label')).toBe(
      copy.renameSplash.renamed(copy.classicNames.artifacts, 'Library')
    )

    act(() => $itemIndex.set({ local: { snapshot, refreshing: false, error: null, failures: [] } }))
    expect(screen.getByRole('button', { name: /Artifacts, now Library/ }).getAttribute('aria-label')).toBe(
      `${copy.renameSplash.renamed(copy.classicNames.artifacts, 'Library')} (0)`
    )
  })

  it('shows a row’s line and where it points when that row is hovered or focused', () => {
    renderShell()
    act(() => {
      $sessions.set([chat])
      $sessionsLoading.set(false)
    })

    const library = screen.getByRole('button', { name: /Artifacts, now Library/ })
    const line = screen.getByText(copy.renameSplash.lines.library)
    expect(line.hidden).toBe(true)

    fireEvent.focus(library)
    expect(library.getAttribute('aria-expanded')).toBe('true')
    expect(line.hidden).toBe(false)
  })
})

describe('the notifications tray', () => {
  it('lights the bell once the splash is seen, and reopens the splash from its entry', () => {
    renderShell()
    const bell = screen.getByRole('button', { name: copy.tray.title })
    expect(bell.getAttribute('aria-description')).toBeNull()

    act(() => $guidanceLedger.set({ renameSplash: 'seen' }))
    expect(bell.getAttribute('aria-description')).toBe(copy.tray.unseen)

    fireEvent.click(bell)
    const tray = screen.getByRole('dialog', { name: copy.tray.title })
    const entry = within(tray).getByRole('article')
    expect(within(entry).getByText(copy.renameItem.title)).toBeTruthy()
    expect(within(entry).getByText(copy.tray.newBadge)).toBeTruthy()
    expect(bell.getAttribute('aria-description')).toBeNull()

    fireEvent.click(within(entry).getByRole('button', { name: copy.renameItem.action }))
    expect($renameSplash.get()).toBe('reopened')
    expect(splash()).not.toBeNull()
  })

  it('announces a newer deployed build with a way to load it now', () => {
    renderShell()
    act(() => $newVersionReady.set(true))
    fireEvent.click(screen.getByRole('button', { name: copy.tray.title }))

    const entry = within(screen.getByRole('dialog', { name: copy.tray.title })).getByRole('article')
    expect(within(entry).getByText(copy.newVersion.title)).toBeTruthy()
    expect(within(entry).getByRole('button', { name: copy.newVersion.refresh })).toBeTruthy()
  })

  it('lists nothing for a new install and says so', () => {
    renderShell()
    fireEvent.click(screen.getByRole('button', { name: copy.tray.title }))

    expect(screen.getByText(copy.tray.empty)).toBeTruthy()
  })
})
