import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { expect, test, vi } from 'vitest'

import { installDesktopPlatformPreflightRuntime } from './desktop-platform-preflight-runtime'
import { readSandboxMarker, WINDOWS_SANDBOX_BREAKPOINT_EXIT } from './windows-sandbox-fallback'

test('pre-ready Windows policy keeps the sandbox recovery marker and relaunch single-use', async () => {
  const userData = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-preflight-'))
  const switches: Array<[string, string | undefined]> = []

  let gpuGone: ((event: unknown, details: unknown) => void) | undefined
  let remoteDisplayHandler: (() => string | null) | undefined
  const relaunch = vi.fn()
  const exitAfterBackendShutdown = vi.fn(async () => undefined)

  const app = {
    commandLine: { appendSwitch: (name: string, value?: string) => switches.push([name, value]) },
    disableHardwareAcceleration: vi.fn(),
    getPath: () => userData,
    getVersion: () => '1.2.3',
    on: (event: string, listener: (event: unknown, details: unknown) => void) => {
      if (event === 'child-process-gone') {
        gpuGone = listener
      }
    },
    relaunch
  }

  const ipcMain = {
    handle: (name: string, handler: () => string | null) => {
      if (name === 'hermes:get-remote-display-reason') {
        remoteDisplayHandler = handler
      }
    }
  }

  try {
    const preflight = installDesktopPlatformPreflightRuntime({
      app: app as any,
      ipcMain: ipcMain as any,
      devServer: undefined,
      env: {},
      argv: ['Hermes.exe'],
      execPath: 'C:/Hermes/Hermes.exe',
      exitAfterBackendShutdown,
      isPackaged: true,
      isWindows: true,
      isWsl: false,
      platform: 'win32'
    })

    expect(remoteDisplayHandler?.()).toBeNull()
    expect(preflight.sandboxState.fallbackActive).toBe(false)
    expect(readSandboxMarker(userData)).toEqual({ state: 'booting' })
    expect(switches).toContainEqual(['disable-renderer-backgrounding', undefined])

    gpuGone?.(null, { type: 'GPU', exitCode: WINDOWS_SANDBOX_BREAKPOINT_EXIT })
    gpuGone?.(null, { type: 'GPU', exitCode: WINDOWS_SANDBOX_BREAKPOINT_EXIT })

    expect(preflight.sandboxState.fallbackActive).toBe(true)
    expect(preflight.sandboxState.fallbackSticky).toBe(true)
    expect(preflight.sandboxState.fallbackReason).toBe('gpu-breakpoint')
    expect(readSandboxMarker(userData)).toEqual({ state: 'fallback', reason: 'gpu-breakpoint', version: '1.2.3' })
    expect(relaunch).toHaveBeenCalledOnce()
    expect(exitAfterBackendShutdown).toHaveBeenCalledExactlyOnceWith(0)
  } finally {
    fs.rmSync(userData, { recursive: true, force: true })
  }
})
