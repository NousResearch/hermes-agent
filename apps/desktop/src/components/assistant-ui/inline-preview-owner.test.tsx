import { act, cleanup, render, waitFor } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'

import { ComposerScopeProvider, ComposerSurfaceProvider, MAIN_COMPOSER_SCOPE } from '@/app/chat/composer/scope'
import { PaneVisibleContext } from '@/components/pane-shell/pane-visibility'

import { InlinePreviewDirective } from './inline-preview-directive'

vi.mock('@/lib/desktop-fs', () => ({
  isReadFileErrorResult: () => false,
  readDesktopFileText: async () => ({ text: '<html><body>synthetic widget</body></html>' })
}))
vi.mock('@/hooks/use-theme-epoch', () => ({ useThemeEpoch: () => 0 }))

afterEach(() => {
  cleanup()
})

it.each([true, false])('binds widget submission to its originating visible surface: %s', async visible => {
  const admitted: unknown[] = []
  const onSubmit = (event: Event) => admitted.push((event as CustomEvent).detail)
  window.addEventListener('hermes:composer-submit', onSubmit)

  try {
    const { container } = render(
      <>
        <div data-composer-surface-id="other-surface" data-composer-target="main" />
        <div data-pane-hidden={visible ? undefined : 'true'}>
          <div data-composer-surface-id="origin-surface" data-composer-target="tile:origin" />
          <PaneVisibleContext.Provider value={visible}>
            <ComposerScopeProvider value={{ ...MAIN_COMPOSER_SCOPE, target: 'tile:origin' }}>
              <ComposerSurfaceProvider value="origin-surface">
                <InlinePreviewDirective attrs={{ file: '/synthetic/widget.html' }} streaming={false} />
              </ComposerSurfaceProvider>
            </ComposerScopeProvider>
          </PaneVisibleContext.Provider>
        </div>
      </>
    )

    await waitFor(() => expect(container.querySelector('iframe')).not.toBeNull())
    const frame = container.querySelector('iframe')!
    const token = JSON.parse(frame.srcdoc.match(/var t=("[^"]+")/)![1])
    act(() => window.dispatchEvent(new MessageEvent('message', {
      data: { type: 'hermes-inline-preview-intent', token, id: 1, prompt: 'harmless marker' },
      source: frame.contentWindow
    })))
    expect(admitted).toEqual(visible ? [{
      target: 'tile:origin', surfaceId: 'origin-surface', text: 'harmless marker', displayKind: 'hidden'
    }] : [])
  } finally {
    window.removeEventListener('hermes:composer-submit', onSubmit)
  }
})
