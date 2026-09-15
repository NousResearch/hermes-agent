import { cleanup, render } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { PaneVisibleContext } from '@/components/pane-shell/pane-visibility'
import { I18nProvider } from '@/i18n'
import * as gateway from '@/store/gateway'
import { _resetSessionOwnerHintsForTests, setSessionOwnerHint } from '@/store/session'
import { $subagentsBySession } from '@/store/subagents'

import { SubagentTranscript } from './subagent-transcript'
import { useSubagentSnapshot } from './use-subagent-snapshot'

const SID = 'sess-hidden-subagent-poll'

function SnapshotHarness({ sessionId }: { sessionId: string }) {
  useSubagentSnapshot(sessionId)
  return <div data-testid="snapshot-harness" />
}

function renderSnapshot(visible: boolean) {
  return render(
    <PaneVisibleContext.Provider value={visible}>
      <SnapshotHarness sessionId={SID} />
    </PaneVisibleContext.Provider>
  )
}

function renderTranscript(visible: boolean) {
  return render(
    <I18nProvider configClient={null} initialLocale="en">
      <PaneVisibleContext.Provider value={visible}>
        <SubagentTranscript sessionId={SID} subagentId="worker" />
      </PaneVisibleContext.Provider>
    </I18nProvider>
  )
}

// #106686: keep-alive hidden tiles stay mounted with document.visibilityState
// === 'visible'. document.hidden alone does not stop subagent.list (5s) or
// subagent.tail (2s); both must gate the interval on usePaneVisible().
describe('hidden-pane subagent polls', () => {
  const request = vi.fn(async (_c: string, _p: string, method: string) => {
    if (method === 'subagent.list') {
      return { subagents: [] }
    }

    if (method === 'subagent.tail') {
      return { available: true, text: 'tail', truncated: false }
    }

    return {}
  })

  const listCalls = () => request.mock.calls.filter(([, , method]) => method === 'subagent.list').length
  const tailCalls = () => request.mock.calls.filter(([, , method]) => method === 'subagent.tail').length

  beforeEach(() => {
    vi.useFakeTimers()
    vi.spyOn(document, 'visibilityState', 'get').mockReturnValue('visible')
    request.mockClear()
    vi.spyOn(gateway, 'requestGatewayForAgent').mockImplementation(request as never)
    setSessionOwnerHint(SID, { connectionId: 'remote-owner', profile: 'research' })
  })

  afterEach(() => {
    cleanup()
    vi.useRealTimers()
    vi.restoreAllMocks()
    $subagentsBySession.set({})
    _resetSessionOwnerHintsForTests()
  })

  it('hidden tile seeds subagent.list once; visible tile polls; reveal resumes', async () => {
    const hidden = renderSnapshot(false)
    await vi.advanceTimersByTimeAsync(0)
    await vi.advanceTimersByTimeAsync(15_000)
    expect(listCalls()).toBe(1)
    hidden.unmount()

    request.mockClear()
    const shown = renderSnapshot(true)
    await vi.advanceTimersByTimeAsync(0)
    await vi.advanceTimersByTimeAsync(15_000)
    // Mount seed + three 5s ticks.
    expect(listCalls()).toBe(1 + 3)
    shown.unmount()
  })

  it('revealing a hidden snapshot tile arms the 5s poll', async () => {
    const view = renderSnapshot(false)
    await vi.advanceTimersByTimeAsync(0)
    await vi.advanceTimersByTimeAsync(15_000)
    expect(listCalls()).toBe(1)

    view.rerender(
      <PaneVisibleContext.Provider value>
        <SnapshotHarness sessionId={SID} />
      </PaneVisibleContext.Provider>
    )
    await vi.advanceTimersByTimeAsync(10_000)
    expect(listCalls()).toBeGreaterThan(1)
    view.unmount()
  })

  it('hidden tile seeds subagent.tail once; visible tile polls; reveal resumes', async () => {
    const hidden = renderTranscript(false)
    await vi.advanceTimersByTimeAsync(0)
    await vi.advanceTimersByTimeAsync(6_000)
    expect(tailCalls()).toBe(1)
    hidden.unmount()

    request.mockClear()
    const shown = renderTranscript(true)
    await vi.advanceTimersByTimeAsync(0)
    await vi.advanceTimersByTimeAsync(6_000)
    // Mount seed + three 2s ticks.
    expect(tailCalls()).toBe(1 + 3)
    shown.unmount()
  })

  it('revealing a hidden transcript tile arms the 2s poll', async () => {
    const view = renderTranscript(false)
    await vi.advanceTimersByTimeAsync(0)
    await vi.advanceTimersByTimeAsync(6_000)
    expect(tailCalls()).toBe(1)

    view.rerender(
      <I18nProvider configClient={null} initialLocale="en">
        <PaneVisibleContext.Provider value>
          <SubagentTranscript sessionId={SID} subagentId="worker" />
        </PaneVisibleContext.Provider>
      </I18nProvider>
    )
    await vi.advanceTimersByTimeAsync(4_000)
    expect(tailCalls()).toBeGreaterThan(1)
    view.unmount()
  })

  it('visible snapshot tile skips in-tick polls when the document is hidden', async () => {
    vi.spyOn(document, 'visibilityState', 'get').mockReturnValue('hidden')
    const shown = renderSnapshot(true)
    await vi.advanceTimersByTimeAsync(0)
    await vi.advanceTimersByTimeAsync(15_000)
    expect(listCalls()).toBe(1)
    shown.unmount()
  })
})
