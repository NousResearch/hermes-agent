import { act, cleanup, render } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { PaneVisibleContext } from '@/components/pane-shell/pane-visibility'
import { I18nProvider } from '@/i18n'
import type { SessionControlHeartbeat } from '@/store/session-control'

import { SessionControlHeartbeatSection } from './session-control-heartbeat'

const heartbeat: SessionControlHeartbeat = {
  created_at: Math.floor(Date.now() / 1000),
  fire_count: 2,
  interval_seconds: 3600,
  last_fired_at: 0,
  prompt: 'keep going',
  status: 'active'
}

function tree(visible: boolean) {
  return (
    <I18nProvider configClient={null} initialLocale="en">
      <PaneVisibleContext.Provider value={visible}>
        <SessionControlHeartbeatSection
          heartbeat={heartbeat}
          onFeedback={() => {}}
          pendingAction={null}
          sessionId="sess-heartbeat"
        />
      </PaneVisibleContext.Provider>
    </I18nProvider>
  )
}

// #122413: keep-alive keeps every ever-active tab mounted, so a background
// tile's heartbeat countdown used to tick every second forever regardless of
// whether its tab was on screen — the same class of bug ./index.tsx's
// background-process poll and ./use-subagent-snapshot.ts already guard against.
describe('SessionControlHeartbeatSection hidden-pane clock', () => {
  beforeEach(() => {
    vi.useFakeTimers()
    vi.spyOn(document, 'visibilityState', 'get').mockReturnValue('visible')
  })

  afterEach(() => {
    cleanup()
    vi.useRealTimers()
    vi.restoreAllMocks()
  })

  const headerText = (container: HTMLElement) => container.querySelector('.status-section-trigger')?.textContent ?? ''

  it('a mounted card pauses while its pane is hidden and catches up the moment it is shown', async () => {
    const view = render(tree(true))
    const label = () => headerText(view.container)

    const tickFor = (ms: number) =>
      act(async () => {
        await vi.advanceTimersByTimeAsync(ms)
      })

    const first = label()
    await tickFor(3_000)
    const shown = label()
    expect(shown).not.toBe(first)

    view.rerender(tree(false))
    await tickFor(5_000)
    expect(label()).toBe(shown) // hidden: no re-render, the label is frozen

    view.rerender(tree(true))
    await act(async () => {}) // re-shown: the clock is re-read at once, before the next tick
    expect(label()).not.toBe(shown)
    view.unmount()
  })
})
