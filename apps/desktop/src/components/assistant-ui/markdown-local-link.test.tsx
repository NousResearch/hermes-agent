import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'

import { MarkdownTextContent } from './markdown-text'

const desktopWindow = window as unknown as { hermesDesktop?: Window['hermesDesktop'] }
const initialHermesDesktop = desktopWindow.hermesDesktop

afterEach(() => {
  cleanup()
  vi.restoreAllMocks()

  if (initialHermesDesktop) {
    desktopWindow.hermesDesktop = initialHermesDesktop
  } else {
    delete desktopWindow.hermesDesktop
  }
})

it.each([
  ['local file', 'file:///C:/Users/example/My%20Note.md'],
  ['Obsidian note', 'obsidian://open?vault=Personal&file=00%20Inbox%2FMy%20Note.md']
])('opens a %s response link through the controlled desktop bridge', async (_label, href) => {
  const openExternal = vi.fn().mockResolvedValue(undefined)
  const openResponseLink = vi.fn().mockResolvedValue(undefined)

  desktopWindow.hermesDesktop = { openExternal, openResponseLink } as unknown as Window['hermesDesktop']

  render(<MarkdownTextContent isRunning={false} text={`[Open note](${href})`} />)

  const link = await screen.findByRole('link', { name: 'Open note' })
  fireEvent.click(link)

  await waitFor(() => expect(openResponseLink).toHaveBeenCalledWith(href))
  expect(openExternal).not.toHaveBeenCalled()
})
