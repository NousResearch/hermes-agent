import { afterEach, expect, it, vi } from 'vitest'

import { writeClipboardText } from './copy-button'

afterEach(() => {
  vi.restoreAllMocks()
  window.hermesDesktop = {} as Window['hermesDesktop']
})

it('uses the renderer copy path after the Electron bridge on Wayland', async () => {
  const writeClipboard = vi.fn().mockResolvedValue(true)
  const execCommand = vi.fn().mockReturnValue(true)
  Object.defineProperty(document, 'execCommand', { configurable: true, value: execCommand })
  window.hermesDesktop = { writeClipboard } as unknown as Window['hermesDesktop']

  await writeClipboardText('reply text')

  expect(writeClipboard).toHaveBeenCalledWith('reply text')
  expect(execCommand).toHaveBeenCalledWith('copy')
  expect(document.querySelector('textarea')).toBeNull()
})
