import * as fs from 'node:fs'
import * as os from 'node:os'

import { app, ipcMain } from 'electron'

export interface MachineProfile {
  ageDays: number | null
  arch: string
  locale: string
  model: string
  nvidia: boolean
  platform: NodeJS.Platform
  release: string
  username: string
}

export function registerMachineProfile(): void {
  ipcMain.handle('hermes:machine:profile', readMachineProfile)
}

async function readMachineProfile(): Promise<MachineProfile> {
  let ageDays: number | null = null
  let username = ''

  // Home birthtime approximates account age. Unsupported filesystems leave it unknown.
  try {
    const { birthtimeMs } = fs.statSync(os.homedir())

    if (birthtimeMs > 0) {
      ageDays = Math.max(0, Math.floor((Date.now() - birthtimeMs) / 86_400_000))
    }
  } catch {
    // Unknown age must not make a daily-use machine look newly set up.
  }

  try {
    username = os.userInfo().username
  } catch {
    // The guide can ask for a name without suggesting an OS login.
  }

  return {
    ageDays,
    arch: process.arch,
    locale: app.getLocale() || '',
    model: readHardwareModel(),
    nvidia: await hasNvidiaGpu(),
    platform: process.platform,
    release: os.release(),
    username
  }
}

function readHardwareModel(): string {
  try {
    // ARM firmware supplies model names such as NVIDIA_DGX_Spark here.
    return fs.readFileSync('/proc/device-tree/model', 'utf8').replace(/\0/g, '').trim()
  } catch {
    return ''
  }
}

async function hasNvidiaGpu(): Promise<boolean> {
  try {
    // Chromium already enumerates GPUs; no vendor tools or subprocess are needed.
    const info = await app.getGPUInfo('basic') as { gpuDevice?: { vendorId?: number }[] }

    return (info.gpuDevice ?? []).some((device): boolean => device.vendorId === 0x10de)
  } catch {
    return false
  }
}
