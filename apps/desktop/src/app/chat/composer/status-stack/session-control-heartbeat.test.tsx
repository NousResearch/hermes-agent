import { cleanup, render } from '@testing-library/react'
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

  it('hidden tile freezes the countdown', async () => {
    const hidden = render(tree(false))
    const initial = headerText(hidden.container)

    await vi.advanceTimersByTimeAsync(5_000)
    expect(headerText(hidden.container)).toBe(initial)
    hidden.unmount()
  })

  it('visible tile keeps the countdown ticking', async () => {
    const shown = render(tree(true))
    const initial = headerText(shown.container)

    await vi.advanceTimersByTimeAsync(3_000)
    expect(headerText(shown.container)).not.toBe(initial)
    shown.unmount()
  })
})
