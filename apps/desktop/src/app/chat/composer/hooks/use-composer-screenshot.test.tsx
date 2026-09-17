import { act, cleanup, renderHook, waitFor } from '@testing-library/react'
import type { PropsWithChildren } from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { I18nProvider } from '@/i18n'
import { createComposerAttachmentScope } from '@/store/composer'

import type { ScreenshotApi } from '../../../../../electron/command-screenshot-types'
import { markActiveComposer } from '../focus'
import { ComposerScopeProvider, ComposerSurfaceProvider, MAIN_COMPOSER_SCOPE } from '../scope'

import { useComposerScreenshot } from './use-composer-screenshot'

const listeners = new Set<(id: string) => void>()
const capture = vi.fn<ScreenshotApi['capture']>()
const onAttach = vi.fn(async (_blob: Blob, isCurrent?: () => boolean) => isCurrent?.() ?? true)

function mount(target: string, surfaceId: string, key = 'draft-a') {
  const scope = { ...MAIN_COMPOSER_SCOPE, target, attachments: createComposerAttachmentScope() }
  const Wrapper = ({ children }: PropsWithChildren) => (
    <I18nProvider configClient={null}>
      <ComposerScopeProvider value={scope}>
        <ComposerSurfaceProvider value={surfaceId}>
          <div data-composer-surface-id={surfaceId} data-composer-target={target}>{children}</div>
        </ComposerSurfaceProvider>
      </ComposerScopeProvider>
    </I18nProvider>
  )

  return renderHook(({ sessionKey }) => useComposerScreenshot({ sessionKey, onAttachImageBlob: onAttach }), {
    initialProps: { sessionKey: key }, wrapper: Wrapper
  })
}

function bridge() {
  window.hermesDesktop = { ...window.hermesDesktop, screenshot: {
    getSettings: vi.fn(), setEnabled: vi.fn(), openPermissionSettings: vi.fn(), onStatus: () => () => undefined,
    capture,
    onRequest: callback => { listeners.add(callback); return () => listeners.delete(callback) }
  } as ScreenshotApi }
}

afterEach(() => {
  cleanup()
  listeners.clear()
  vi.clearAllMocks()
  delete window.hermesDesktop.screenshot
  markActiveComposer('main')
})

describe('screenshot composer routing', () => {
  it('attaches once to the active split composer even without document focus, preserving its draft', async () => {
    bridge()
    capture.mockResolvedValue({ ok: true, png: new Uint8Array([1, 2, 3]) })
    mount('main', 'primary')
    mount('tile:second', 'second')
    markActiveComposer('tile:second')
    expect(document.hasFocus()).toBe(false)
    act(() => listeners.forEach(listener => listener('gesture')))
    await waitFor(() => expect(onAttach).toHaveBeenCalledTimes(1))
    expect(capture).toHaveBeenCalledExactlyOnceWith('gesture')
    expect(onAttach.mock.calls[0]![0].type).toBe('image/png')
  })

  it('rejects a late screenshot after a draft round trip and invalidates the image-write continuation', async () => {
    bridge()
    let finish!: (value: Awaited<ReturnType<ScreenshotApi['capture']>>) => void
    capture.mockImplementation(() => new Promise(resolve => { finish = resolve }))
    const hook = mount('main', 'primary')
    act(() => listeners.forEach(listener => listener('first')))
    hook.rerender({ sessionKey: 'draft-b' })
    hook.rerender({ sessionKey: 'draft-a' })
    await act(async () => finish({ ok: true, png: new Uint8Array([1]) }))
    expect(onAttach).not.toHaveBeenCalled()

    capture.mockResolvedValue({ ok: true, png: new Uint8Array([1]) })
    let isCurrent!: () => boolean
    onAttach.mockImplementation(async (_blob, guard) => { isCurrent = guard!; return true })
    act(() => listeners.forEach(listener => listener('second')))
    await waitFor(() => expect(onAttach).toHaveBeenCalledOnce())
    expect(isCurrent()).toBe(true)
    hook.rerender({ sessionKey: 'draft-b' })
    expect(isCurrent()).toBe(false)
  })
})
