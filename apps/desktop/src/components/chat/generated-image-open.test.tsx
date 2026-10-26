import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import type { HermesConnection } from '@/global'
import { I18nProvider } from '@/i18n'
import { $connection } from '@/store/session'

import { GeneratedImage } from './generated-image-result'

// The image never loads, so the card collapses to its "Open image" link.
vi.mock('@/hooks/use-media-image', () => ({
  useMediaImage: () => ({ failed: true, frameStyle: {}, key: 'k', loaded: false, src: '' })
}))

const openExternal = vi.fn()
const saveGatewayFile = vi.fn().mockResolvedValue({ ok: true })

beforeEach(() => {
  // A native Electron renderer: the preload's bridge, no browser-host marker.
  ;(window as unknown as { hermesDesktop: unknown }).hermesDesktop = { openExternal, saveGatewayFile }
})

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
  $connection.set(null)
  delete (window as unknown as { hermesDesktop?: unknown }).hermesDesktop
})

function openImageLink(image: string) {
  render(
    <I18nProvider configClient={null} initialLocale="en">
      <GeneratedImage result={{ image }} />
    </I18nProvider>
  )

  return screen.getByRole('link', { name: /Open image/ })
}

// A remote gateway's file URL carries the connection token. Handed to the
// system browser it would sit in that browser's history, so native Desktop
// saves the file through the bridge's save dialog instead.
it('saves a remote gateway image through the save dialog in native Desktop, never opening its tokened URL', async () => {
  $connection.set({
    baseUrl: 'https://gateway.example',
    connectionId: 'remote-1',
    mode: 'remote',
    profile: 'default',
    token: 'secret-token'
  } as unknown as HermesConnection)

  const link = openImageLink('/srv/hermes/images/cat.png')
  await act(async () => {
    fireEvent.click(link)
  })

  expect(saveGatewayFile).toHaveBeenCalledWith(
    expect.objectContaining({ connectionId: 'remote-1', path: '/srv/hermes/images/cat.png', suggestedName: 'cat.png' })
  )
  expect(JSON.stringify(saveGatewayFile.mock.calls)).not.toContain('secret-token')
  expect(openExternal).not.toHaveBeenCalled()
})

// Anything else opens as on main: a web image's own URL.
it('opens a web image\'s URL with the system handler in native Desktop', async () => {
  $connection.set({ baseUrl: 'https://gateway.example', mode: 'remote', token: 'secret-token' } as unknown as HermesConnection)

  const link = openImageLink('https://images.example/cat.png')
  await act(async () => {
    fireEvent.click(link)
  })

  expect(openExternal).toHaveBeenCalledWith('https://images.example/cat.png')
  expect(saveGatewayFile).not.toHaveBeenCalled()
})
