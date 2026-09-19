import { expect, type Mock, test, vi } from 'vitest'

import { registerDesktopUninstallIpc } from './desktop-uninstall'
import type { DesktopUninstallIpcDeps, DesktopUninstallSummary } from './desktop-uninstall'
import type { InstallStamp } from './install-stamp'

const fallbackSummary: Omit<DesktopUninstallSummary, 'code_removal_allowed'> = {
  hermes_home: '/test/.hermes',
  agent_installed: true,
  gui_installed: true,
  source_built_artifacts: [],
  packaged_app_paths: [],
  userdata_dir: '/test/desktop',
  userdata_exists: true,
  platform: process.platform,
  probe: 'fallback'
}

interface CapturedIpc {
  probeSummary: Mock<() => Promise<DesktopUninstallSummary>>
  runUninstall: Mock<DesktopUninstallIpcDeps['runUninstall']>
  localSummary: Mock<DesktopUninstallIpcDeps['fallbackSummary']>
  invoke: (channel: string, payload?: unknown) => Promise<unknown>
}

function captureIpc(stamp: DesktopUninstallIpcDeps['stamp']): CapturedIpc {
  type Handler = Parameters<DesktopUninstallIpcDeps['ipcMain']['handle']>[1]

  const handlers: Map<string, Handler> = new Map()

  const probeSummary: CapturedIpc['probeSummary'] = vi.fn(async (): Promise<DesktopUninstallSummary> => ({
    ...fallbackSummary,
    probe: 'python',
    code_removal_allowed: true
  }))

  const runUninstall: CapturedIpc['runUninstall'] = vi.fn<DesktopUninstallIpcDeps['runUninstall']>(
    async (mode: string): ReturnType<DesktopUninstallIpcDeps['runUninstall']> => ({ ok: true, mode })
  )

  const localSummary: CapturedIpc['localSummary'] = vi.fn(
    (): ReturnType<DesktopUninstallIpcDeps['fallbackSummary']> => fallbackSummary
  )

  const ipcMain: DesktopUninstallIpcDeps['ipcMain'] = {
    handle: (channel: string, handler: Handler): void => {
      handlers.set(channel, handler)
    }
  }

  registerDesktopUninstallIpc({ ipcMain, stamp, fallbackSummary: localSummary, probeSummary, runUninstall })

  return {
    probeSummary,
    runUninstall,
    localSummary,
    invoke: async (channel: string, payload?: unknown): Promise<unknown> => {
      const handler: Handler | undefined = handlers.get(channel)

      if (!handler) {
        throw new Error(`Missing IPC handler: ${channel}`)
      }

      return handler(undefined, payload)
    }
  }
}

test('excluded artifact owners block every uninstall IPC before a Python probe or destructive callback', async (): Promise<void> => {
  const stamps: Array<Partial<InstallStamp>> = [
    { distribution: 'nix' },
    { source: 'nix' },
    { payload: 'bundled', updateMechanism: 'electron-updater' },
    { payload: 'light', updateMechanism: 'app-installer' },
    { payload: 'bootstrap', distribution: 'package-manager', updateMechanism: 'external' }
  ]

  for (const stamp of stamps) {
    const ipc: CapturedIpc = captureIpc(stamp)

    expect(await ipc.invoke('hermes:uninstall:summary')).toEqual({
      ...fallbackSummary,
      code_removal_allowed: false
    })
    expect(ipc.probeSummary).not.toHaveBeenCalled()

    for (const mode of ['gui', 'lite', 'full', 'data']) {
      for (const payload of [mode, { mode }]) {
        expect(await ipc.invoke('hermes:uninstall:run', payload)).toMatchObject({
          ok: false,
          error: 'externally-managed'
        })
      }
    }

    expect(ipc.runUninstall).not.toHaveBeenCalled()
  }
})

test('self-managed installs retain summary and uninstall IPC behavior under Electron policy', async (): Promise<void> => {
  for (const stamp of [null, { payload: 'bootstrap', updateMechanism: 'self' }] as const) {
    const ipc: CapturedIpc = captureIpc(stamp)

    ipc.probeSummary.mockResolvedValue({ ...fallbackSummary, probe: 'python', code_removal_allowed: false })
    expect(await ipc.invoke('hermes:uninstall:summary')).toEqual({
      ...fallbackSummary,
      probe: 'python',
      code_removal_allowed: true
    })
    expect(ipc.probeSummary).toHaveBeenCalledOnce()
    expect(ipc.localSummary).not.toHaveBeenCalled()

    ipc.probeSummary.mockResolvedValue({ ...fallbackSummary, code_removal_allowed: false })
    expect(await ipc.invoke('hermes:uninstall:summary')).toMatchObject({
      probe: 'fallback',
      code_removal_allowed: true
    })

    for (const mode of ['gui', 'lite', 'full']) {
      for (const payload of [mode, { mode }]) {
        expect(await ipc.invoke('hermes:uninstall:run', payload)).toEqual({ ok: true, mode })
        expect(ipc.runUninstall).toHaveBeenLastCalledWith(mode)
      }
    }

    ipc.runUninstall.mockClear()

    for (const payload of ['unknown', 'data', undefined, { mode: 'unknown' }]) {
      expect(await ipc.invoke('hermes:uninstall:run', payload)).toMatchObject({ ok: false, error: 'invalid-mode' })
    }

    expect(ipc.runUninstall).not.toHaveBeenCalled()
  }
})
