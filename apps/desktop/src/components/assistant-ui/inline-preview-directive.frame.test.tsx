import { act, cleanup, render, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type * as FocusModule from '@/app/chat/composer/focus'
import { ComposerScopeProvider, MAIN_COMPOSER_SCOPE } from '@/app/chat/composer/scope'

import { InlinePreviewDirective, MAX_FRAME_WIDTH } from './inline-preview-directive'

const submit = vi.hoisted(() => vi.fn(() => true))

vi.mock('@/app/chat/composer/focus', async importOriginal => ({
  ...(await importOriginal<typeof FocusModule>()),
  requestComposerSubmit: submit
}))

const original = window.hermesDesktop

// The frame reads its file through the desktop bridge, as it does in the app.
beforeEach(() => {
  submit.mockClear()
  window.hermesDesktop = {
    ...original,
    readFileText: async (path: string) => ({
      path,
      text: '<html><body><div style="width:300px">card</div></body></html>'
    })
  }
})

afterEach(() => {
  cleanup()
  window.hermesDesktop = original
})

async function mountFrame(target?: string) {
  const directive = <InlinePreviewDirective attrs={{ file: '/work/card.html' }} streaming={false} />
  const { container } = render(
    target ? (
      <ComposerScopeProvider value={{ ...MAIN_COMPOSER_SCOPE, target }}>{directive}</ComposerScopeProvider>
    ) : (
      directive
    )
  )

  const iframe = await waitFor(() => {
    const node = container.querySelector('iframe')

    if (!node?.parentElement) {
      throw new Error('preview frame not mounted yet')
    }

    return node
  })

  const token = /var t="([^"]+)"/.exec(iframe.getAttribute('srcdoc') ?? '')?.[1]
  expect(token).toBeDefined()

  const report = (size: { height: number; width?: number }) =>
    act(() => {
      window.dispatchEvent(new MessageEvent('message', { data: { type: 'hermes-inline-preview-size', token, ...size } }))
    })

  return { box: iframe.parentElement, report }
}

// No pixel width taken from the content: the block spans the transcript
// column up to the frame cap, and CSS follows the column on resize.
function expectColumnWide(box: HTMLElement | null) {
  // Height is the only inline size the frame takes from the content.
  expect(box?.style.cssText).toMatch(/^height: \d+px;$/)
  expect(box?.style.width).toBe('')
  expect(box?.classList.contains('w-full')).toBe(true)
  expect(box?.closest('.max-w-160')).toBeNull()
  expect(box?.parentElement?.style.maxWidth).toBe(`${MAX_FRAME_WIDTH}px`)
}

describe('inline preview widget intents', () => {
  it("sends the widget's prompt to its own tile composer, not the active one", async () => {
    const { box } = await mountFrame('tile:split-1')
    const iframe = box!.querySelector('iframe')!
    const token = /var t="([^"]+)"/.exec(iframe.getAttribute('srcdoc') ?? '')?.[1]

    await act(async () => {
      window.dispatchEvent(
        new MessageEvent('message', {
          data: { id: 1, prompt: 'refresh the chart', token, type: 'hermes-inline-preview-intent' }
        })
      )
    })

    expect(submit).toHaveBeenCalledWith('refresh the chart', { displayKind: 'hidden', target: 'tile:split-1' })
  })
})

describe('inline preview frame width', () => {
  it('fills the transcript column whatever width the content reports', async () => {
    const { box, report } = await mountFrame()

    await report({ height: 400, width: 300 })

    expect(box?.style.height).toBe('400px')
    expectColumnWide(box)
  })

  it('keeps filling the column when a later report is wider', async () => {
    const { box, report } = await mountFrame()

    await report({ height: 400, width: 300 })
    await report({ height: 400, width: 900 })

    expectColumnWide(box)
  })
})
