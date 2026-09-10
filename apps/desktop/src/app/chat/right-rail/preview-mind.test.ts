import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

const target = (url: string) => ({ kind: 'url' as const, label: url, source: url, url })

describe('preview mind', () => {
  beforeEach(() => {
    vi.resetModules()
  })

  afterEach(async () => {
    const { setBusy } = await import('@/store/session')
    const { closeRightRail } = await import('@/store/preview')

    setBusy(false)
    closeRightRail()
  })

  it('rests the overlay that started thinking when another browser tab becomes active', async () => {
    const { $rightRailActiveTabId, selectRightRailTab } = await import('@/store/layout')
    const { newBrowserTab, openPreview } = await import('@/store/preview')
    const { setBusy } = await import('@/store/session')
    const { registerPreviewScriptRunner } = await import('./preview-script-runner')

    openPreview(target('https://example.com'))

    const firstTabId = $rightRailActiveTabId.get()!
    const firstRunner = vi.fn(async (_code: string) => 'ok')
    const unregisterFirst = registerPreviewScriptRunner(firstTabId, firstRunner)

    newBrowserTab()

    const secondTabId = $rightRailActiveTabId.get()!
    const secondRunner = vi.fn(async (_code: string) => 'ok')
    const unregisterSecond = registerPreviewScriptRunner(secondTabId, secondRunner)

    selectRightRailTab(firstTabId)
    await import('./preview-mind')
    setBusy(true)

    expect(firstRunner.mock.calls.at(-1)?.[0]).toContain('"think"')
    expect(secondRunner).not.toHaveBeenCalled()

    selectRightRailTab(secondTabId)
    setBusy(false)

    expect(secondRunner.mock.calls.at(-1)?.[0]).toContain('"rest"')
    expect(firstRunner.mock.calls.at(-1)?.[0]).toContain('"rest"')

    unregisterSecond()
    unregisterFirst()
  })
})
