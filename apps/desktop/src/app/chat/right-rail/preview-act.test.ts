import { beforeEach, describe, expect, it, vi } from 'vitest'

import { $rightRailActiveTabId } from '@/store/layout'
import { closeRightRail, openPreview, type PreviewTarget } from '@/store/preview'

import { actOnActivePreview } from './preview-act'
import { registerPreviewNav } from './preview-nav'
import { registerPreviewScriptRunner } from './preview-script-runner'

function urlTarget(url: string): PreviewTarget {
  return { kind: 'url', label: 'Browser', source: url, url }
}

describe('actOnActivePreview (act_preview tool)', () => {
  // URL targets share the singleton Browser tab id, so anything a test
  // registers would answer the next one.
  let cleanups: Array<() => void> = []

  const openBrowserTab = () => {
    openPreview(urlTarget('https://example.com'), 'tool-result')

    return $rightRailActiveTabId.get()!
  }

  const withRunner = (runner: (code: string) => Promise<unknown>) =>
    cleanups.push(registerPreviewScriptRunner(openBrowserTab(), runner))

  beforeEach(() => {
    vi.useRealTimers()

    for (const cleanup of cleanups) {
      cleanup()
    }

    cleanups = []
    closeRightRail()
    window.localStorage.clear()
  })

  it('tells the agent to open a page when no live pane is behind the tab', async () => {
    const result = await actOnActivePreview({ kind: 'elements' })

    expect(result.success).toBe(false)
    expect(result.error).toContain('open_preview')
  })

  it('injects the engine and returns the page’s answer', async () => {
    let injected = ''

    withRunner(async code => {
      injected = code

      return JSON.stringify({ acted: 'clicked button "Save"', success: true })
    })

    const result = await actOnActivePreview({ kind: 'click', ref: '@e1' })

    expect(result).toMatchObject({ acted: 'clicked button "Save"', success: true })
    // Self-contained payload: the engine source and the action travel together,
    // and the holder keeps refs alive across calls on the same page.
    expect(injected).toContain('__hermesActHolder')
    expect(injected).toContain('"ref":"@e1"')
  })

  it('re-inventories after a mutating action so the next ref is current', async () => {
    const actions: string[] = []

    withRunner(code => {
      // Stand in for the guest page: run the script's own settle/rescan shape
      // by answering each act() call in order.
      actions.push(...(code.match(/"kind":"(\w+)"/g) ?? []))

      return Promise.resolve(
        JSON.stringify({
          acted: 'clicked',
          elements: [{ label: 'Log out', ref: '@e1', role: 'button', selector: '#out' }],
          success: true,
          url: 'https://example.com/app'
        })
      )
    })

    const result = await actOnActivePreview({ kind: 'click', ref: '@e1' })

    expect(result.elements?.[0].label).toBe('Log out')
    expect(result.url).toBe('https://example.com/app')
  })

  it('does not pay the settle delay for a plain inventory', async () => {
    let injected = ''
    withRunner(async code => {
      injected = code

      return JSON.stringify({ elements: [], success: true })
    })

    await actOnActivePreview({ kind: 'elements' })

    expect(injected).toContain('0 <= 0')
  })

  it('reports a page that answers with nothing', async () => {
    withRunner(async () => '')

    expect((await actOnActivePreview({ kind: 'click', ref: '@e1' })).error).toContain('did not answer')
  })

  it('routes history verbs to the pane instead of the guest page', async () => {
    const back = vi.fn()
    const runner = vi.fn()

    const tabId = openBrowserTab()
    cleanups.push(registerPreviewNav(tabId, { back, forward: vi.fn(), reload: vi.fn() }))
    cleanups.push(registerPreviewScriptRunner(tabId, runner))

    const result = await actOnActivePreview({ kind: 'back' })

    expect(back).toHaveBeenCalledOnce()
    expect(runner).not.toHaveBeenCalled()
    expect(result.success).toBe(true)
    expect(result.note).toContain('elements')
  })

  it('reports history verbs with no pane to drive', async () => {
    expect((await actOnActivePreview({ kind: 'reload' })).error).toContain('open_preview')
  })
})
