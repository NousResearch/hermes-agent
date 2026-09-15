import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { IS_MAC } from '@/lib/keybinds/combo'
import { $previewTabs, closeRightRail } from '@/store/preview'

import { DirectiveContent } from './directive-text'

const desktopWindow = window as unknown as { hermesDesktop?: Window['hermesDesktop'] }

const PR_URL = 'https://github.com/NousResearch/hermes-agent/pull/107950'

function installDesktopBridge() {
  const openExternal = vi.fn().mockResolvedValue(undefined)

  desktopWindow.hermesDesktop = {
    fetchLinkTitle: vi.fn().mockResolvedValue(''),
    openExternal
  } as unknown as Window['hermesDesktop']

  return openExternal
}

afterEach(() => {
  closeRightRail()
  vi.restoreAllMocks()
  cleanup()
  delete desktopWindow.hermesDesktop
  document.body.replaceChildren()
})

// The wire form of a url chip in a SENT user message: a `<button>` had none of
// the gestures a real link gets, which is exactly what these tests pin.
describe('sent-message url chip', () => {
  it('opens the in-app preview pane on a plain click', async () => {
    const openExternal = installDesktopBridge()

    render(<DirectiveContent text={`@url:\`${PR_URL}\``} />)

    fireEvent.click(screen.getByRole('link'))

    expect(openExternal).not.toHaveBeenCalled()
    await waitFor(() => expect($previewTabs.get().at(-1)?.target.url).toBe(PR_URL))
  })

  // Platform-specific on purpose (same rule as lib/external-link.test.tsx):
  // ⌘ on macOS, Ctrl elsewhere. The suite runs as non-mac.
  it('escapes to the system browser on the platform open-elsewhere modifier', () => {
    const openExternal = installDesktopBridge()

    render(<DirectiveContent text={`@url:\`${PR_URL}\``} />)

    fireEvent.click(screen.getByRole('link'), IS_MAC ? { metaKey: true } : { ctrlKey: true })

    expect(openExternal).toHaveBeenCalledWith(PR_URL)
    expect($previewTabs.get()).toHaveLength(0)
  })

  it('escapes to the system browser on middle-click', () => {
    const openExternal = installDesktopBridge()

    render(<DirectiveContent text={`@url:\`${PR_URL}\``} />)

    // No fireEvent.auxClick in this Testing Library version — dispatch the
    // real `auxclick` event React's onAuxClick listens for.
    fireEvent(screen.getByRole('link'), new MouseEvent('auxclick', { bubbles: true, button: 1 }))

    expect(openExternal).toHaveBeenCalledWith(PR_URL)
    expect($previewTabs.get()).toHaveLength(0)
  })
})
