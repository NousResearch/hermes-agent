import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { PaneVisibleContext } from '@/components/pane-shell/pane-visibility'
import { I18nProvider } from '@/i18n'
import { en } from '@/i18n/en'
import {
  $backgroundStatusBySession,
  refreshBackgroundProcesses,
  resetBackgroundPollingGuard
} from '@/store/composer-status'
import { $gateway } from '@/store/gateway'

import { ComposerStatusStack } from './index'

// The stack measures itself into a surface var — jsdom has no ResizeObserver.
class ResizeObserverStub {
  observe() {}
  unobserve() {}
  disconnect() {}
}

vi.stubGlobal('ResizeObserver', ResizeObserverStub)

const S = en.statusStack
const SID = 'sess-bg-notify'
const CMD = 'curl -o test-file.bin https://example.test/1gb'

const running = { command: CMD, notify_on_complete: true, session_id: 'bg-notify', status: 'running' }
const exited = { ...running, exit_code: 0, status: 'exited' }
const failed = { ...running, exit_code: 1, status: 'exited' }

let snapshot: Record<string, unknown>[] = []

const request = vi.fn(async (method: string) => (method === 'process.list' ? { processes: snapshot } : {}))

function renderStack() {
  return render(
    <MemoryRouter>
      <I18nProvider configClient={null} initialLocale="en">
        <PaneVisibleContext.Provider value>
          <ComposerStatusStack queue={null} sessionId={SID} />
        </PaneVisibleContext.Provider>
      </I18nProvider>
    </MemoryRouter>
  )
}

const settle = () => vi.advanceTimersByTimeAsync(0)

/** The background group is collapsed by default; expand it to read the row. */
const expandBackgroundGroup = () => fireEvent.click(screen.getByRole('button', { name: /Background/ }))

// #111522: a background task can outlive the turn that started it. The chat must
// keep saying so — visibly, from the registry (not from renderer memory) — and
// must turn into the task's result instead of silently going quiet.
describe('background task delivery state', () => {
  beforeEach(() => {
    vi.useFakeTimers()
    snapshot = [running]
    request.mockClear()
    $backgroundStatusBySession.set({})
    resetBackgroundPollingGuard()
    $gateway.set({ request } as never)
  })

  afterEach(() => {
    cleanup()
    vi.useRealTimers()
    vi.restoreAllMocks()
    $gateway.set(null as never)
    $backgroundStatusBySession.set({})
    resetBackgroundPollingGuard()
  })

  it('says a result is coming, without the user expanding anything', async () => {
    renderStack()
    await settle()

    // Collapsed card: the always-visible line carries the promise.
    expect(screen.getByText(S.background(1))).toBeTruthy()
    expect(screen.getByText(S.willNotifyChat)).toBeTruthy()

    expandBackgroundGroup()
    expect(screen.getByText(CMD)).toBeTruthy()
    expect(screen.getAllByText(S.willNotifyChat).length).toBeGreaterThan(0)
  })

  it('stays gone-but-true for a plain fire-and-forget process', async () => {
    snapshot = [{ command: 'dev server', session_id: 'bg-plain', status: 'running' }]
    renderStack()
    await settle()

    // No notify contract, no delivery promise.
    expect(screen.queryByText(S.willNotifyChat)).toBeNull()
  })

  it('re-derives the running state from the registry after leaving and coming back', async () => {
    const first = renderStack()
    await settle()
    expect(screen.getByText(S.willNotifyChat)).toBeTruthy()

    // Leaving the chat: the renderer drops every row it holds…
    first.unmount()
    $backgroundStatusBySession.set({})

    // …and coming back rebuilds them from the gateway registry alone.
    renderStack()
    await settle()

    expect(screen.getByText(S.background(1))).toBeTruthy()
    expect(screen.getByText(S.willNotifyChat)).toBeTruthy()
  })

  it('turns the finished task into its result instead of vanishing', async () => {
    renderStack()
    await settle()

    // Next poll: the registry reports the same task exited successfully.
    snapshot = [exited]
    await refreshBackgroundProcesses(SID)

    // Still on screen long after the success linger window (4s) that clears a
    // fire-and-forget row: the user who returns must still see it ended.
    await vi.advanceTimersByTimeAsync(30_000)

    expect(screen.getByText(S.notifySent)).toBeTruthy()
    expect(screen.queryByText(S.willNotifyChat)).toBeNull()

    expandBackgroundGroup()
    expect(screen.getByText(CMD)).toBeTruthy()
  })

  it('shows a failed task with its exit code and the way to its output', async () => {
    renderStack()
    await settle()

    snapshot = [failed]
    await refreshBackgroundProcesses(SID)
    await vi.advanceTimersByTimeAsync(30_000)

    expect(screen.getByText(S.notifySent)).toBeTruthy()

    expandBackgroundGroup()
    expect(screen.getByText(CMD)).toBeTruthy()
    expect(screen.getByText(S.exit(1))).toBeTruthy()
    expect(screen.getByText(S.openOutput)).toBeTruthy()
  })
})
