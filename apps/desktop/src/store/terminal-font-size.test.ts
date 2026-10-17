import { beforeEach, expect, it, vi } from 'vitest'

const KEY = 'hermes.desktop.terminal-font-size.v1'

beforeEach(() => {
  vi.resetModules()
  localStorage.clear()
})

it('defaults to 11px and persists a selected preset across reloads', async () => {
  const { $terminalFontSize, setTerminalFontSize } = await import('./terminal-font-size')
  expect($terminalFontSize.get()).toBe(11)

  setTerminalFontSize(14)
  expect($terminalFontSize.get()).toBe(14)
  expect(localStorage.getItem(KEY)).toBe('14')

  vi.resetModules()
  expect((await import('./terminal-font-size')).$terminalFontSize.get()).toBe(14)
})

it('falls back to the default for invalid input and clears storage at the default', async () => {
  localStorage.setItem(KEY, 'huge')
  const { $terminalFontSize, setTerminalFontSize } = await import('./terminal-font-size')
  expect($terminalFontSize.get()).toBe(11)

  setTerminalFontSize(13)
  setTerminalFontSize(99)
  expect($terminalFontSize.get()).toBe(11)
  expect(localStorage.getItem(KEY)).toBeNull()
})
