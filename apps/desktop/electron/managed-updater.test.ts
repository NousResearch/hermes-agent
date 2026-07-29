import assert from 'node:assert/strict'
import { EventEmitter } from 'node:events'

import { afterEach, describe, test, vi } from 'vitest'

import { createManagedUpdater, type ManagedUpdateSnapshot, shouldEnableManagedUpdates } from './managed-updater'

class FakeAppUpdater extends EventEmitter {
  allowDowngrade = true
  autoDownload = false
  autoInstallOnAppQuit = false
  checkForUpdates = vi.fn(async () => null)
}

const services: Array<ReturnType<typeof createManagedUpdater>> = []

function setup(enabled = true, shouldAcceptVersion?: (version: string) => boolean) {
  const updater = new FakeAppUpdater()
  const service = createManagedUpdater({ enabled, now: () => 1_234, shouldAcceptVersion, updater })

  services.push(service)

  return { service, updater }
}

afterEach(() => {
  for (const service of services.splice(0)) {
    service.dispose()
  }
})

describe('managed packaged updater', () => {
  test('enables only packaged Windows builds with generated update configuration', () => {
    assert.equal(shouldEnableManagedUpdates({ isPackaged: true, platform: 'win32', updateConfigExists: true }), true)
    assert.equal(shouldEnableManagedUpdates({ isPackaged: true, platform: 'darwin', updateConfigExists: true }), false)
    assert.equal(shouldEnableManagedUpdates({ isPackaged: true, platform: 'linux', updateConfigExists: true }), false)
    assert.equal(shouldEnableManagedUpdates({ isPackaged: false, platform: 'win32', updateConfigExists: true }), false)
    assert.equal(shouldEnableManagedUpdates({ isPackaged: true, platform: 'win32', updateConfigExists: false }), false)
  })

  test('enables zero-click background download and install-on-normal-quit', async () => {
    const { service, updater } = setup()

    await service.start()

    assert.equal(updater.autoDownload, true)
    assert.equal(updater.autoInstallOnAppQuit, true)
    assert.equal(updater.allowDowngrade, false)
    assert.equal(updater.checkForUpdates.mock.calls.length, 1)
    assert.equal(service.getSnapshot().stage, 'checking')
  })

  test('publishes package progress and the staged automatic-install state', async () => {
    const { service, updater } = setup()
    const snapshots: ManagedUpdateSnapshot[] = []

    service.subscribe(snapshot => snapshots.push(snapshot))
    await service.start()
    updater.emit('update-available', { version: '0.18.0' })
    updater.emit('download-progress', {
      bytesPerSecond: 2_048,
      percent: 42.34,
      total: 1_000,
      transferred: 423
    })
    updater.emit('update-downloaded', { version: '0.18.0' })

    assert.deepEqual(snapshots.at(-1), {
      checkedAt: 1_234,
      percent: 100,
      stage: 'downloaded',
      version: '0.18.0'
    })
    assert.equal(
      snapshots.some(snapshot => snapshot.stage === 'downloading' && snapshot.percent === 42.3),
      true
    )
  })

  test('returns to idle when no packaged release is available', async () => {
    const { service, updater } = setup()

    await service.start()
    updater.emit('update-not-available', { version: '0.17.0' })

    assert.deepEqual(service.getSnapshot(), {
      checkedAt: 1_234,
      percent: null,
      stage: 'idle',
      version: '0.17.0'
    })
  })

  test('does not redownload an exact version that startup health rejected', async () => {
    const { service, updater } = setup(true, version => version !== '0.17.201')

    await service.start()
    updater.emit('update-available', { version: '0.17.201' })

    assert.equal(updater.autoDownload, false)
    assert.deepEqual(service.getSnapshot(), {
      checkedAt: 1_234,
      percent: null,
      stage: 'idle',
      version: '0.17.201'
    })

    updater.emit('update-available', { version: '0.17.202' })

    assert.equal(updater.autoDownload, true)
    assert.equal(service.getSnapshot().stage, 'available')
  })

  test('surfaces updater failures without throwing into the desktop lifecycle', async () => {
    const { service, updater } = setup()

    await service.start()
    updater.emit('error', new Error('release metadata unavailable'))

    assert.deepEqual(service.getSnapshot(), {
      checkedAt: 1_234,
      error: 'release metadata unavailable',
      percent: null,
      stage: 'error'
    })
  })

  test('does not run the packaged updater in an unpackaged development build', async () => {
    const { service, updater } = setup(false)

    await service.start()

    assert.equal(updater.checkForUpdates.mock.calls.length, 0)
    assert.equal(updater.autoDownload, false)
    assert.equal(updater.autoInstallOnAppQuit, false)
    assert.equal(service.getSnapshot().stage, 'disabled')
  })

  test('coalesces overlapping checks', async () => {
    const { service, updater } = setup()
    let finish: (() => void) | undefined

    updater.checkForUpdates.mockImplementation(
      () =>
        new Promise(resolve => {
          finish = () => resolve(null)
        })
    )

    const first = service.check()
    const second = service.check()

    assert.equal(updater.checkForUpdates.mock.calls.length, 1)
    finish?.()
    await Promise.all([first, second])
  })
})
