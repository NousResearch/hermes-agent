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

function snapshotTree(visible: boolean) {
  return (
    <PaneVisibleContext.Provider value={visible}>
      <SnapshotHarness sessionId={SID} />
    </PaneVisibleContext.Provider>
  )
}

function transcriptTree(visible: boolean) {
  return (
    <I18nProvider configClient={null} initialLocale="en">
      <PaneVisibleContext.Provider value={visible}>
        <SubagentTranscript sessionId={SID} subagentId="worker" />
      </PaneVisibleContext.Provider>
    </I18nProvider>
  )
}

// #106686: keep-alive hidden tiles stay mounted with document.visibilityState
// === 'visible', so document.hidden alone never stops subagent.list (5s) or
// subagent.tail (2s); both effects gate on usePaneVisible() and re-run on reveal.
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

  it('hidden snapshot tile issues no subagent.list; reveal seeds once and polls every 5s; hiding again stops it', async () => {
    const view = render(snapshotTree(false))
    await vi.advanceTimersByTimeAsync(15_000)
    expect(listCalls()).toBe(0)

    view.rerender(snapshotTree(true))
    await vi.advanceTimersByTimeAsync(0)
    expect(listCalls()).toBe(1)
    await vi.advanceTimersByTimeAsync(15_100)
    expect(listCalls()).toBe(1 + 3)

    view.rerender(snapshotTree(false))
    await vi.advanceTimersByTimeAsync(15_000)
    expect(listCalls()).toBe(1 + 3)
  })

  it('hidden transcript tile issues no subagent.tail; reveal seeds once and polls every 2s; hiding again stops it', async () => {
    const view = render(transcriptTree(false))
    await vi.advanceTimersByTimeAsync(6_000)
    expect(tailCalls()).toBe(0)

    view.rerender(transcriptTree(true))
    await vi.advanceTimersByTimeAsync(0)
    expect(tailCalls()).toBe(1)
    await vi.advanceTimersByTimeAsync(6_100)
    expect(tailCalls()).toBe(1 + 3)

    view.rerender(transcriptTree(false))
    await vi.advanceTimersByTimeAsync(6_000)
    expect(tailCalls()).toBe(1 + 3)
  })

  it('visible snapshot tile skips in-tick polls while the document is hidden', async () => {
    vi.spyOn(document, 'visibilityState', 'get').mockReturnValue('hidden')
    render(snapshotTree(true))
    await vi.advanceTimersByTimeAsync(0)
    await vi.advanceTimersByTimeAsync(15_000)
    expect(listCalls()).toBe(1)
  })
})
