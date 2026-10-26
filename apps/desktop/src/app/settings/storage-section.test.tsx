import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'

import type { DesktopStorageSummary } from '@/global'
import { I18nProvider } from '@/i18n'

import { formatBytes, StorageSection } from './storage-section'

afterEach((): void => {
  cleanup()
  vi.unstubAllGlobals()
})

it('formats byte counts', (): void => {
  expect(formatBytes(0)).toBe('0 B')
  expect(formatBytes(1536)).toBe('1.5 KB')
  expect(formatBytes(5 * 1024 ** 3)).toBe('5.0 GB')
})

it('lists entries, drills into a folder with a way back, and reveals on the icon', async (): Promise<void> => {
  const home: DesktopStorageSummary = {
    root: '/test/home',
    totalBytes: 3072,
    entries: [
      { name: 'sessions', bytes: 2048, kind: 'dir' },
      { name: 'config.yaml', bytes: 1024, kind: 'file' },
      { name: 'cron', bytes: 0, kind: 'dir' }
    ],
    truncated: false
  }

  const inner: DesktopStorageSummary = {
    root: '/test/home/sessions',
    totalBytes: 2048,
    entries: [{ name: 'state.db', bytes: 2048, kind: 'file' }],
    truncated: false
  }

  const summary = vi.fn((trail: string[]) => Promise.resolve(trail.length ? inner : home))
  const reveal = vi.fn().mockResolvedValue({ ok: true })

  vi.stubGlobal('hermesDesktop', { storage: { summary, reveal } })

  render(
    <I18nProvider>
      <StorageSection />
    </I18nProvider>
  )

  await waitFor((): void => expect(screen.getByText('sessions')).toBeTruthy())
  expect(screen.getByText('3.0 KB')).toBeTruthy()
  expect(screen.queryByText('cron')).toBeNull()

  // The left icon shows where the item is, without drilling in.
  fireEvent.click(screen.getAllByRole('button', { name: 'Open file location' })[1])
  expect(reveal).toHaveBeenCalledWith([], 'config.yaml')
  expect(summary).toHaveBeenLastCalledWith([])

  // Clicking a folder shows its contents in this page.
  fireEvent.click(screen.getByText('sessions'))
  await waitFor((): void => expect(screen.getByText('state.db')).toBeTruthy())
  expect(summary).toHaveBeenLastCalledWith(['sessions'])
  expect(screen.getByText('Hermes home / sessions')).toBeTruthy()

  // A file reveals instead of drilling.
  fireEvent.click(screen.getByText('state.db'))
  expect(reveal).toHaveBeenLastCalledWith(['sessions'], 'state.db')

  // Back returns to the parent level.
  fireEvent.click(screen.getByRole('button', { name: 'Back' }))
  await waitFor((): void => expect(screen.getByText('config.yaml')).toBeTruthy())

  // The parent was already measured, so going back must not scan it again.
  expect(summary).toHaveBeenCalledTimes(2)

  // Rescan is the only thing that measures again.
  fireEvent.click(screen.getByRole('button', { name: 'Rescan' }))
  await waitFor((): void => expect(summary).toHaveBeenCalledTimes(3))
  expect(summary).toHaveBeenLastCalledWith([])
})

const HOME: DesktopStorageSummary = {
  root: '/test/home',
  totalBytes: 3072,
  entries: [
    { name: 'sessions', bytes: 2048, kind: 'dir' },
    { name: 'config.yaml', bytes: 1024, kind: 'file' }
  ],
  truncated: false
}

const SESSIONS: DesktopStorageSummary = {
  root: '/test/home/sessions',
  totalBytes: 2048,
  entries: [{ name: 'state.db', bytes: 2048, kind: 'file' }],
  truncated: false
}

it('drops a rescan that finishes after the user opened a folder', async (): Promise<void> => {
  let finishRescan: (value: DesktopStorageSummary) => void = (): void => undefined
  let calls = 0

  const summary = vi.fn((trail: string[]) => {
    calls += 1

    // The first scan and the folder scan answer at once; the rescan of the home hangs.
    return trail.length || calls === 1 ? Promise.resolve(trail.length ? SESSIONS : HOME) : new Promise<DesktopStorageSummary>(resolve => (finishRescan = resolve))
  })

  vi.stubGlobal('hermesDesktop', { storage: { summary, reveal: vi.fn() } })

  render(
    <I18nProvider>
      <StorageSection />
    </I18nProvider>
  )

  await waitFor((): void => expect(screen.getByText('sessions')).toBeTruthy())

  fireEvent.click(screen.getByRole('button', { name: 'Rescan' }))
  fireEvent.click(screen.getByText('sessions'))
  await waitFor((): void => expect(screen.getByText('state.db')).toBeTruthy())

  // The home rescan finally answers; it must not replace the folder being viewed.
  await act(async (): Promise<void> => {
    finishRescan({ ...HOME, entries: [{ name: 'stale-home-row', bytes: 1, kind: 'file' }] })
    await new Promise<void>(resolve => setTimeout(resolve, 0))
  })

  expect(screen.getByText('state.db')).toBeTruthy()
  expect(screen.queryByText('stale-home-row')).toBeNull()
})

it('lets the icon button reveal without the row opening the folder', async (): Promise<void> => {
  const summary = vi.fn((trail: string[]) => Promise.resolve(trail.length ? SESSIONS : HOME))
  const reveal = vi.fn().mockResolvedValue({ ok: true })

  vi.stubGlobal('hermesDesktop', { storage: { summary, reveal } })

  render(
    <I18nProvider>
      <StorageSection />
    </I18nProvider>
  )

  await waitFor((): void => expect(screen.getByText('sessions')).toBeTruthy())

  // Enter on the focused icon bubbles to the row; it must not drill in.
  fireEvent.keyDown(screen.getAllByRole('button', { name: 'Open file location' })[0], { key: 'Enter' })
  expect(summary).toHaveBeenCalledTimes(1)

  // Enter on the row itself still opens it.
  fireEvent.keyDown(screen.getByText('sessions').closest('[role="button"]') as HTMLElement, { key: 'Enter' })
  await waitFor((): void => expect(summary).toHaveBeenLastCalledWith(['sessions']))
})

it('renders nothing without the bridge', (): void => {
  vi.stubGlobal('hermesDesktop', {})

  const { container } = render(
    <I18nProvider>
      <StorageSection />
    </I18nProvider>
  )

  expect(container.innerHTML).toBe('')
})
