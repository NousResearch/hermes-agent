import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { DesktopMachineProfile } from '@/global'

import {
  $machine,
  loadMachineProfile,
  machineDescription,
  machineIsSpark,
  machineKind,
  machineLooksNew,
  machineSetupLeads,
  machineUserName
} from './machine'
import { forkFallbackOptions, forkOptions, machineForkOption } from './onboarding-script'

const { invoke } = vi.hoisted(() => ({ invoke: vi.fn() }))

vi.mock('electron', () => ({
  contextBridge: {
    exposeInMainWorld: (key: string, api: Window['hermesDesktop']) => vi.stubGlobal(key, api)
  },
  ipcRenderer: { invoke, sendSync: () => ({ guestOnboarding: true }) },
  webFrame: {},
  webUtils: {}
}))

afterEach(() => vi.unstubAllGlobals())

const profile = (patch: Partial<DesktopMachineProfile>): DesktopMachineProfile => ({
  ageDays: 900,
  arch: 'x64',
  locale: 'en-US',
  model: '',
  nvidia: false,
  platform: 'darwin',
  release: '24.6.0',
  username: '',
  ...patch
})

describe('machine profile', () => {
  beforeEach(() => {
    $machine.set(null)
  })

  it('knows an RTX Spark by its shape — Windows, Arm, NVIDIA — whatever the OEM badge says', () => {
    $machine.set(profile({ ageDays: 900, arch: 'arm64', nvidia: true, platform: 'win32' }))

    expect(machineIsSpark()).toBe(true)
    expect(machineKind()).toBe('Spark')
    expect(machineSetupLeads()).toBe(true)
  })

  it('does not mistake the neighbours of that shape for one', () => {
    const neighbours: Partial<DesktopMachineProfile>[] = [
      { arch: 'arm64', platform: 'win32' }, // Snapdragon Windows-on-Arm
      { arch: 'x64', nvidia: true, platform: 'win32' }, // an NVIDIA gaming tower
      { arch: 'arm64', platform: 'darwin' } // an Apple silicon Mac
    ]

    for (const neighbour of neighbours) {
      $machine.set(profile(neighbour))
      expect(machineIsSpark()).toBe(false)
    }
  })

  it('reads a DGX Spark off the device tree, underscores and all', () => {
    // What a real unit reports.
    $machine.set(profile({ ageDays: 900, arch: 'arm64', model: 'NVIDIA_DGX_Spark', nvidia: true, platform: 'linux' }))

    expect(machineIsSpark()).toBe(true)
    expect(machineKind()).toBe('Spark')
  })

  it('knows the GB10 box under a partner badge', () => {
    for (const model of ['NVIDIA DGX Spark', 'ASUS Ascent GX10 (GB10)', 'NVIDIA GB10']) {
      $machine.set(profile({ model, platform: 'linux' }))
      expect(machineIsSpark()).toBe(true)
    }
  })

  it('does not read an ordinary box as one', () => {
    for (const model of ['', 'MacBookPro18,3', 'Raspberry Pi 5 Model B']) {
      $machine.set(profile({ model }))
      expect(machineIsSpark()).toBe(false)
    }
  })

  it('leads on a machine set up days ago, and stands down on a lived-in one', () => {
    $machine.set(profile({ ageDays: 3 }))
    expect(machineSetupLeads()).toBe(true)

    $machine.set(profile({ ageDays: 400 }))
    expect(machineLooksNew()).toBe(false)
    expect(machineSetupLeads()).toBe(false)
  })

  it('treats an unknown age as not-new rather than guessing', () => {
    $machine.set(profile({ ageDays: null }))

    expect(machineSetupLeads()).toBe(false)
  })

  it('leads the brief with how fresh the machine is, since that is what changes the work', () => {
    $machine.set(profile({ ageDays: 0, arch: 'arm64', nvidia: true, platform: 'win32' }))
    expect(machineDescription()).toMatch(/^set up today, an NVIDIA Spark, win32/)

    $machine.set(profile({ ageDays: 4 }))
    expect(machineDescription()).toMatch(/^set up 4 days ago, darwin/)

    $machine.set(profile({ ageDays: 900 }))
    expect(machineDescription()).toMatch(/^darwin/)
  })

  it('names the machine the way the user would', () => {
    $machine.set(profile({ platform: 'darwin' }))
    expect(machineForkOption()).toBe('Help me set up this Mac')

    $machine.set(profile({ platform: 'win32' }))
    expect(machineForkOption()).toBe('Help me set up this PC')

    $machine.set(null)
    expect(machineForkOption()).toBe('Help me set up this computer')
  })

  it('offers a login name as a suggestion only when it looks like a name', () => {
    $machine.set(profile({ username: 'alex' }))
    expect(machineUserName()).toBe('alex')

    $machine.set(profile({ username: 'Austin.Pickett' }))
    expect(machineUserName()).toBe('Austin.Pickett')
  })

  it('holds back login handles that are not a name', () => {
    for (const username of ['user', 'admin', 'administrator', 'default', 'guest', 'me', 'owner', 'root', 'test']) {
      $machine.set(profile({ username }))
      expect(machineUserName(), `"${username}" must not be suggested`).toBeNull()
    }
  })

  it('holds back an absent or unusable account name', () => {
    $machine.set(profile({ username: '' }))
    expect(machineUserName()).toBeNull()

    $machine.set(profile({ username: 'x' }))
    expect(machineUserName()).toBeNull()

    $machine.set(profile({ username: 'u' }))
    expect(machineUserName()).toBeNull()

    $machine.set(null)
    expect(machineUserName()).toBeNull()
  })
})

describe('the fork', () => {
  beforeEach(() => {
    $machine.set(null)
  })

  it('offers the new machine one job and folds the rest behind one tap', () => {
    $machine.set(profile({ ageDays: 2 }))

    expect(forkOptions()).toEqual(['Help me set up this Mac', 'Something else'])
    expect(forkFallbackOptions()).toHaveLength(4)
  })

  it('lists everything up front on a machine that is already someone’s', () => {
    $machine.set(profile({ ageDays: 400 }))

    const options = forkOptions()

    expect(options).toHaveLength(5)
    expect(options[0]).toBe('I have something in mind')
    expect(options).toContain('Help me set up this Mac')
    expect(forkFallbackOptions()).toEqual([])
  })

  it('never drops an option between the two tiers', () => {
    $machine.set(profile({ ageDays: 2 }))
    const twoTier = [...forkOptions().filter(option => option !== 'Something else'), ...forkFallbackOptions()]

    $machine.set(profile({ ageDays: 400 }))

    expect([...twoTier].sort()).toEqual([...forkOptions()].sort())
  })
})

it('loads machine facts through the real preload bridge and reuses them for the guide', async () => {
  const host = profile({ username: 'Alex', ageDays: 2 })

  $machine.set(null)
  invoke.mockResolvedValue(host)
  // Preload is typechecked with the Electron project's compiler options.
  await vi.importActual('../../electron/preload')
  expect(invoke).not.toHaveBeenCalled()

  await loadMachineProfile()
  await loadMachineProfile()

  expect(invoke).toHaveBeenCalledExactlyOnceWith('hermes:machine:profile')
  expect($machine.get()).toBe(host)
  expect(machineUserName()).toBe(host.username)
  expect(machineSetupLeads()).toBe(true)
  expect(machineForkOption()).toBe('Help me set up this Mac')
})
