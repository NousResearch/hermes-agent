import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { MarkdownTextContent } from './markdown-text'

const desktopWindow = window as unknown as { hermesDesktop?: Window['hermesDesktop'] }
const initialHermesDesktop = desktopWindow.hermesDesktop

function installDesktopBridge(partial: Partial<Window['hermesDesktop']> = {}) {
  desktopWindow.hermesDesktop = {
    fetchLinkTitle: vi.fn().mockResolvedValue(''),
    openExternal: vi.fn().mockResolvedValue(undefined),
    ...partial
  } as unknown as Window['hermesDesktop']
}

afterEach(() => {
  vi.restoreAllMocks()
  cleanup()

  if (initialHermesDesktop) {
    desktopWindow.hermesDesktop = initialHermesDesktop
  } else {
    delete desktopWindow.hermesDesktop
  }
})

describe('MarkdownTextContent', () => {
  it.each([
    ['inline', '[Bug 8797](https://dev.azure.com/electronicdevice/Apps/_workitems/edit/8797)'],
    [
      'table cell',
      '| Task |\n|---|\n| [Bug 8797](https://dev.azure.com/electronicdevice/Apps/_workitems/edit/8797): Site search |'
    ]
  ])('keeps explicit markdown link labels in %s instead of replacing them with fetched titles', async (_case, text) => {
    const fetchLinkTitle = vi.fn().mockResolvedValue('Azure DevOps Services | Sign In')
    installDesktopBridge({ fetchLinkTitle: fetchLinkTitle as unknown as Window['hermesDesktop']['fetchLinkTitle'] })

    render(<MarkdownTextContent isRunning={false} text={text} />)

    const link = await screen.findByRole('link', { name: 'Bug 8797' })
    expect(link.getAttribute('href')).toBe('https://dev.azure.com/electronicdevice/Apps/_workitems/edit/8797')

    await new Promise(resolve => setTimeout(resolve, 0))
    expect(fetchLinkTitle).not.toHaveBeenCalled()
    expect(link.textContent).toBe('Bug 8797')
  })
})
