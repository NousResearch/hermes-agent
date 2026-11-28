// The open lightbox steps through every image in the chat it was opened from
// (port of block/buzz#6705): ←/→ and the toolbar arrows move to the
// neighbouring transcript image; images in another chat surface are not part
// of the gallery.
import { act, cleanup, fireEvent, render, screen, within } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'

import { I18nProvider } from '@/i18n/context'
import { resetTranscriptLightbox } from '@/store/transcript-lightbox'

import { ZoomableImage } from './zoomable-image'

const config = { getConfig: async () => ({}), saveConfig: async () => ({ ok: true }) }

async function renderChats() {
  await act(async () => {
    render(
      <I18nProvider configClient={config}>
        <div data-chat-surface="">
          <ZoomableImage alt="first" src="https://e.x/a.png" />
          <ZoomableImage alt="second" src="https://e.x/b.png" />
          <ZoomableImage alt="third" src="https://e.x/c.png" />
        </div>
        <div data-chat-surface="">
          <ZoomableImage alt="other chat" src="https://e.x/z.png" />
        </div>
      </I18nProvider>
    )
  })
}

const shown = () => (within(screen.getByRole('dialog')).getByRole('img') as HTMLImageElement).src

describe('transcript lightbox gallery', () => {
  afterEach(() => {
    cleanup()
    resetTranscriptLightbox()
  })

  it('steps to neighbouring images in the same chat with arrow keys and buttons', async () => {
    await renderChats()
    await act(async () => {
      fireEvent.click(screen.getByAltText('first').closest('button')!)
    })
    expect(shown()).toBe('https://e.x/a.png')
    expect((screen.getByRole('button', { name: 'Previous image' }) as HTMLButtonElement).disabled).toBe(true)

    await act(async () => {
      fireEvent.keyDown(screen.getByRole('dialog'), { key: 'ArrowRight' })
    })
    expect(shown()).toBe('https://e.x/b.png')

    // The gallery ends at this chat's last image; the other chat's image is not in it.
    await act(async () => {
      fireEvent.click(screen.getByRole('button', { name: 'Next image' }))
    })
    expect(shown()).toBe('https://e.x/c.png')
    expect((screen.getByRole('button', { name: 'Next image' }) as HTMLButtonElement).disabled).toBe(true)

    await act(async () => {
      fireEvent.keyDown(screen.getByRole('dialog'), { key: 'ArrowLeft' })
    })
    expect(shown()).toBe('https://e.x/b.png')
  })

  it('shows no step arrows for an image that is alone in its chat', async () => {
    await renderChats()
    await act(async () => {
      fireEvent.click(screen.getByAltText('other chat').closest('button')!)
    })
    expect(shown()).toBe('https://e.x/z.png')
    expect(screen.queryByRole('button', { name: 'Next image' })).toBeNull()
  })
})
