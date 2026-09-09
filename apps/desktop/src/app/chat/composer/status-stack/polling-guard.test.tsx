import { cleanup, render } from '@testing-library/react'
import { MemoryRouter } from 'react-router'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { PaneVisibleContext } from '@/components/pane-shell/pane-visibility'
import { I18nProvider } from '@/i18n'
import { $backgroundStatusBySession, resetBackgroundPollingGuard } from '@/store/composer-status'
import { $gateway } from '@/store/gateway'

import { ComposerStatusStack } from './index'

// The stack measures itself into a surface var — jsdom has no ResizeObserver.
class ResizeObserverStub {
  observe() {}
  unobserve() {}
  disconnect() {}
}

vi.stubGlobal('ResizeObserver', ResizeObserverStub)

const SID = 'sess-dead-runtime'

function renderStack() {
  return render(
    <MemoryRouter>
      <I18nProvider configClient={null} initialLocale="en">
        <ComposerStatusStack queue={null} sessionId={SID} />
      </I18nProvider>
    </MemoryRouter>
  )
}

// #98434: a boot-restored tile can stay bound to a dead runtime id and remount
// repeatedly (no genuine rebind ever happens). The mount effect used to clear
// the gone-polling latch on every mount, so each remount re-armed the 4001
// storm against that id forever.
describe('ComposerStatusStack dead-runtime remount', () => {
  beforeEach(() => {
    resetBackgroundPollingGuard()
  })

  afterEach(() => {
    cleanup()
    $gateway.set(null as never)
    resetBackgroundPollingGuard()
  })

  it('does not re-poll process.list after a remount once the session is latched gone', async () => {
    const request = vi.fn(async (method: string) => {
      if (method === 'process.list') {
        throw new Error('session not found')
      }

      return {}
    })

    const processListCalls = () => request.mock.calls.filter(([method]) => method === 'process.list').length

    $gateway.set({ request } as never)

    const first = renderStack()
    await Promise.resolve()
    await Promise.resolve()

    expect(processListCalls()).toBe(1)

    first.unmount()

    const second = renderStack()
    await Promise.resolve()
    await Promise.resolve()

    // Before the fix: the mount effect cleared the latch, so this remount
    // re-fired process.list against the same dead id.
    expect(processListCalls()).toBe(1)

    second.unmount()
  })
})

// #73287: keep-alive keeps every ever-active tab mounted, so each background
// tile's 5s safety-net poll used to fire even while its tab was hidden — N
// sessions meant N gateway round-trips plus shared-map churn forever.
describe('ComposerStatusStack hidden-pane poll', () => {
  const SID_BG = 'sess-bg-poll'

  const runningList = vi.fn(async (method: string) =>
    method === 'process.list'
      ? { processes: [{ command: 'dev server', session_id: 'bg1', status: 'running' }] }
      : {}
  )
  const processListCalls = () => runningList.mock.calls.filter(([method]) => method === 'process.list').length

  function renderStackBg(visible: boolean) {
    return render(
      <MemoryRouter>
        <I18nProvider configClient={null} initialLocale="en">
          <PaneVisibleContext.Provider value={visible}>
            <ComposerStatusStack queue={null} sessionId={SID_BG} />
          </PaneVisibleContext.Provider>
        </I18nProvider>
      </MemoryRouter>
    )
  }

  beforeEach(() => {
    vi.useFakeTimers()
    // jsdom defaults to 'prerender', which the in-tick document check skips.
    vi.spyOn(document, 'visibilityState', 'get').mockReturnValue('visible')
    runningList.mockClear()
    $gateway.set({ request: runningList } as never)
  })

  afterEach(() => {
    cleanup()
    vi.useRealTimers()
    vi.restoreAllMocks()
    $gateway.set(null as never)
    $backgroundStatusBySession.set({})
    resetBackgroundPollingGuard()
  })

  it('hidden tile skips the 5s poll; visible tile polls; reveal resumes', async () => {
    const hidden = renderStackBg(false)
    await vi.advanceTimersByTimeAsync(0)
    await vi.advanceTimersByTimeAsync(15_000)
    // Mount seed only — the interval never armed while hidden.
    expect(processListCalls()).toBe(1)
    hidden.unmount()

    const shown = renderStackBg(true)
    await vi.advanceTimersByTimeAsync(0)
    await vi.advanceTimersByTimeAsync(15_000)
    // Mount seed + three 5s ticks.
    expect(processListCalls()).toBe(1 + 1 + 3)
    shown.unmount()
  })

  it('revealing a hidden tile arms the poll', async () => {
    const view = renderStackBg(false)
    await vi.advanceTimersByTimeAsync(0)
    await vi.advanceTimersByTimeAsync(15_000)
    expect(processListCalls()).toBe(1)

    view.rerender(
      <MemoryRouter>
        <I18nProvider configClient={null} initialLocale="en">
          <PaneVisibleContext.Provider value>
            <ComposerStatusStack queue={null} sessionId={SID_BG} />
          </PaneVisibleContext.Provider>
        </I18nProvider>
      </MemoryRouter>
    )
    await vi.advanceTimersByTimeAsync(10_000)
    expect(processListCalls()).toBeGreaterThan(1)
    view.unmount()
  })
})
