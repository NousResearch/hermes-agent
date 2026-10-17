import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { HermesReadDirResult } from '@/global'
import type * as WindowsModule from '@/store/windows'

import { $diskPluginsScanPending, watchRuntimePlugins } from './runtime-loader'

// The HUD window is a query-param fact of the real renderer; the module-level
// window-kind caches make overriding location.search mid-suite unreliable, so
// the window module is mocked instead.
vi.mock('@/store/windows', async importActual => ({
  ...(await importActual<typeof WindowsModule>()),
  isHudWindow: () => true
}))

const desktopPluginsRoot = vi.fn<() => Promise<string>>()
const readDir = vi.fn<(path: string) => Promise<HermesReadDirResult>>()
const watchDirectory = vi.fn<(path: string) => Promise<{ id: string }>>()
const onPreviewFileChanged = vi.fn()

beforeEach(() => {
  desktopPluginsRoot.mockReset()
  readDir.mockReset()
  watchDirectory.mockReset()
  onPreviewFileChanged.mockReset()
  ;(window as unknown as { hermesDesktop: unknown }).hermesDesktop = {
    desktopPluginsRoot,
    onPreviewFileChanged,
    readDir,
    watchDirectory
  }
})

afterEach(() => {
  delete (window as unknown as { hermesDesktop?: unknown }).hermesDesktop
})

describe('watchRuntimePlugins HUD gate (#138663)', () => {
  it('starts no disk scan, watch or poll in the HUD window', async () => {
    desktopPluginsRoot.mockResolvedValue('/local/.hermes/desktop-plugins')

    watchRuntimePlugins()

    // A skipped door settles synchronously: no pending scan, nothing watched,
    // and the watcher registration itself never runs.
    expect($diskPluginsScanPending.get()).toBe(false)
    expect(onPreviewFileChanged).not.toHaveBeenCalled()
    expect(watchDirectory).not.toHaveBeenCalled()
    expect(readDir).not.toHaveBeenCalled()

    // The poll fallback is also never armed — give any mis-armed timer a beat
    // to fire and assert it had nothing to do.
    await new Promise(resolve => setTimeout(resolve, 20))

    expect(readDir).not.toHaveBeenCalled()
    expect(desktopPluginsRoot).not.toHaveBeenCalled()
  })
})
