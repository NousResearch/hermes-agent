import { beforeEach, describe, expect, it, vi } from 'vitest'

const STORAGE_KEY = 'hermes.desktop.fileBrowser.showHiddenFiles'

beforeEach(() => {
  window.localStorage.clear()
  vi.resetModules()
})

describe('file browser preferences', () => {
  it('shows hidden files by default for backward compatibility', async () => {
    const { $showHiddenFiles } = await import('./file-browser')

    expect($showHiddenFiles.get()).toBe(true)
  })

  it('restores and persists the device-local preference', async () => {
    window.localStorage.setItem(STORAGE_KEY, 'false')

    const { $showHiddenFiles, setShowHiddenFiles, toggleShowHiddenFiles } = await import('./file-browser')

    expect($showHiddenFiles.get()).toBe(false)

    toggleShowHiddenFiles()
    expect($showHiddenFiles.get()).toBe(true)
    expect(window.localStorage.getItem(STORAGE_KEY)).toBe('true')

    setShowHiddenFiles(false)
    expect(window.localStorage.getItem(STORAGE_KEY)).toBe('false')
  })

  it('recognizes dotfiles without treating ordinary dotted names as hidden', async () => {
    const { isHiddenFileName } = await import('./file-browser')

    expect(isHiddenFileName('.env')).toBe(true)
    expect(isHiddenFileName('.github')).toBe(true)
    expect(isHiddenFileName('component.test.ts')).toBe(false)
    expect(isHiddenFileName('.')).toBe(false)
  })
})
