// @vitest-environment jsdom
import { beforeEach, describe, expect, it, vi } from 'vitest'

const notifyErrorMock = vi.fn()

vi.mock('@/store/notifications', () => ({
  notifyError: (...args: unknown[]) => notifyErrorMock(...args)
}))

import { $ideWorkspaceRoot, IDE_WORKSPACE_STORAGE_KEY } from './state'
import { openIdeFolder } from './workspace'

const desktopWindow = window as unknown as { hermesDesktop?: unknown }
const initialBridge = desktopWindow.hermesDesktop

function installBridge(selectPaths?: (options?: unknown) => Promise<string[]>) {
  desktopWindow.hermesDesktop = {
    ...(selectPaths ? { selectPaths } : {})
  }
}

beforeEach(() => {
  window.localStorage.clear()
  $ideWorkspaceRoot.set(null)
  notifyErrorMock.mockReset()
  installBridge(vi.fn().mockResolvedValue([]))
})

describe('openIdeFolder', () => {
  it('adopts the picked folder as the IDE workspace and persists it', async () => {
    installBridge(vi.fn().mockResolvedValue(['D:\\repo\\two']))

    const chosen = await openIdeFolder('could not open')

    expect(chosen).toBe('D:\\repo\\two')
    expect($ideWorkspaceRoot.get()).toBe('D:\\repo\\two')
    expect(window.localStorage.getItem(IDE_WORKSPACE_STORAGE_KEY)).toContain('D:\\repo\\two')
  })

  it('leaves the workspace untouched when the dialog is cancelled', async () => {
    $ideWorkspaceRoot.set('/repo/one')
    installBridge(vi.fn().mockResolvedValue([]))

    expect(await openIdeFolder('could not open')).toBe(null)
    expect($ideWorkspaceRoot.get()).toBe('/repo/one')
    expect(notifyErrorMock).not.toHaveBeenCalled()
  })

  it('no-ops outside Electron', async () => {
    delete desktopWindow.hermesDesktop

    expect(await openIdeFolder('could not open')).toBe(null)
    expect(notifyErrorMock).not.toHaveBeenCalled()
  })

  it('notifies on a dialog failure', async () => {
    installBridge(vi.fn().mockRejectedValue(new Error('dialog boom')))

    expect(await openIdeFolder('could not open')).toBe(null)
    expect(notifyErrorMock).toHaveBeenCalledTimes(1)
  })
})
