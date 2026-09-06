import { describe, expect, it, vi } from 'vitest'

import { createPreviewUblockController } from './preview-ublock'

function fakeSession() {
  const extensions = new Map<string, any>()
  let nextId = 0

  return {
    extensions: {
      getAllExtensions: () => [...extensions.values()],
      getExtension: (id: string) => extensions.get(id) ?? null,
      loadExtension: async (extensionPath: string) => {
        const extension = {
          id: `ublock-${++nextId}`,
          manifest: { name: 'uBlock Origin Lite', version: '2026.825.1619' },
          path: extensionPath,
          url: `chrome-extension://ublock-${nextId}`
        }

        extensions.set(extension.id, extension)

        return extension
      },
      removeExtension: (id: string) => extensions.delete(id)
    }
  }
}

const cached = { path: '/user-data/preview-ublock/versions/pinned', version: '2026.825.1619' }

function candidate() {
  return {
    finalPath: '/user-data/preview-ublock/versions/pinned-next',
    installFinal: vi.fn(),
    commitActive: vi.fn(),
    discard: vi.fn(),
    stagedPath: '/user-data/preview-ublock/.staging-next',
    version: '2026.825.1619'
  }
}

function installer(resolve = vi.fn().mockResolvedValue(cached)) {
  return { resolve }
}

describe('preview uBlock lifecycle', () => {
  it('does not resolve or load an extension on default startup', async () => {
    const resolve = vi.fn()
    const previewSession = fakeSession()
    const controller = createPreviewUblockController({ installer: installer(resolve), session: previewSession })

    await expect(controller.initialize()).resolves.toMatchObject({
      enabled: false,
      available: false,
      operation: { phase: 'idle' }
    })
    expect(resolve).not.toHaveBeenCalled()
    expect(previewSession.extensions.getAllExtensions()).toEqual([])
  })

  it('loads a cached extension and publishes ready with a version', async () => {
    const previewSession = fakeSession()
    const controller = createPreviewUblockController({ enabled: true, installer: installer(), session: previewSession })

    await expect(controller.initialize()).resolves.toMatchObject({
      enabled: true,
      available: true,
      dashboardUrl: 'chrome-extension://ublock-1/dashboard.html',
      operation: { phase: 'ready', operationId: expect.any(String) },
      rulesetsReady: true,
      version: '2026.825.1619'
    })
  })

  it('publishes the validated popup URL only after activation succeeds', async () => {
    const previewSession = fakeSession()
    const controller = createPreviewUblockController({
      enabled: true,
      installer: installer(vi.fn().mockResolvedValue({ ...cached, popupPath: 'popup.html' })),
      session: previewSession
    })

    await expect(controller.initialize()).resolves.toMatchObject({
      available: true,
      popupUrl: 'chrome-extension://ublock-1/popup.html',
      rulesetsReady: true
    })
  })

  it('clears the popup URL when uBlock is disabled or activation fails', async () => {
    const previewSession = fakeSession()
    const controller = createPreviewUblockController({
      installer: installer(vi.fn().mockResolvedValue({ ...cached, popupPath: 'popup.html' })),
      session: previewSession
    })

    await controller.setEnabled(true)
    await expect(controller.setEnabled(false)).resolves.toMatchObject({ popupUrl: null, enabled: false })

    const failed = createPreviewUblockController({
      enabled: true,
      installer: installer(vi.fn().mockRejectedValue(new Error('broken'))),
      session: fakeSession()
    })
    await expect(failed.initialize()).resolves.toMatchObject({ popupUrl: null, enabled: false })
  })

  it('uses the pinned intent on explicit enable and unloads without deleting the cache', async () => {
    const previewSession = fakeSession()
    const resolve = vi.fn().mockResolvedValue(cached)
    const controller = createPreviewUblockController({ installer: installer(resolve), session: previewSession })

    await expect(controller.setEnabled(true)).resolves.toMatchObject({ enabled: true, available: true })
    await expect(controller.setEnabled(false)).resolves.toMatchObject({
      enabled: false,
      available: false,
      operation: { phase: 'idle' }
    })
    expect(resolve).toHaveBeenCalledWith('pinned', expect.any(Object))
    expect(previewSession.extensions.getAllExtensions()).toEqual([])
  })

  it('returns a categorized failed state and leaves the preference disabled', async () => {
    const previewSession = fakeSession()
    const resolve = vi.fn().mockRejectedValue(Object.assign(new Error('offline'), { code: 'network' }))
    const controller = createPreviewUblockController({ installer: installer(resolve), session: previewSession })

    await expect(controller.setEnabled(true)).resolves.toMatchObject({
      enabled: false,
      available: false,
      operation: { phase: 'failed', failure: { code: 'network', message: 'offline' } }
    })
    expect(previewSession.extensions.getAllExtensions()).toEqual([])
  })

  it('does not commit a candidate when final-path validation fails', async () => {
    const next = candidate()
    let activePointer = 'previous-release'
    next.commitActive.mockImplementation(() => {
      activePointer = 'candidate-release'
    })
    const resolve = vi.fn().mockResolvedValue(next)
    const bootstrap = vi.fn().mockResolvedValueOnce(true).mockResolvedValueOnce(false)
    const controller = createPreviewUblockController({
      bootstrap,
      installer: installer(resolve),
      session: fakeSession()
    })

    await expect(controller.setEnabled(true)).resolves.toMatchObject({
      enabled: false,
      operation: { phase: 'failed', failure: { code: 'activation' } }
    })
    expect(next.installFinal).toHaveBeenCalledOnce()
    expect(next.commitActive).not.toHaveBeenCalled()
    expect(next.discard).toHaveBeenCalledOnce()
    expect(activePointer).toBe('previous-release')
  })

  it('cannot become ready when the functional match probe reports no rule', async () => {
    const resolve = vi.fn().mockResolvedValue(cached)
    const bootstrap = vi.fn().mockResolvedValue(false)
    const controller = createPreviewUblockController({
      bootstrap,
      installer: installer(resolve),
      session: fakeSession()
    })

    await expect(controller.setEnabled(true)).resolves.toMatchObject({
      enabled: false,
      available: false,
      rulesetsReady: false,
      operation: { phase: 'failed', failure: { code: 'activation' } }
    })
    expect(bootstrap).toHaveBeenCalledOnce()
  })

  it('coalesces concurrent enable requests into one installation', async () => {
    const previewSession = fakeSession()
    let release!: () => void

    const resolve = vi.fn().mockImplementation(
      () =>
        new Promise(resolvePromise => {
          release = () => resolvePromise(cached)
        })
    )

    const controller = createPreviewUblockController({ installer: installer(resolve), session: previewSession })

    const first = controller.setEnabled(true)
    const second = controller.setEnabled(true)
    expect(second).toBe(first)
    await Promise.resolve()
    release()
    await expect(first).resolves.toMatchObject({ enabled: true })
    expect(resolve).toHaveBeenCalledOnce()
  })

  it('publishes every operation transition to subscribers', async () => {
    const states: string[] = []
    const controller = createPreviewUblockController({ installer: installer(), session: fakeSession() })
    controller.subscribe(next => states.push(next.operation.phase))
    await controller.setEnabled(true)
    expect(states).toContain('checking-cache')
    expect(states).toContain('loading')
    expect(states).toContain('validating')
    expect(states.at(-1)).toBe('ready')
  })

  it('isolates a throwing subscriber from the install outcome and other subscribers', async () => {
    const healthyStates: string[] = []
    const controller = createPreviewUblockController({ installer: installer(), session: fakeSession() })

    expect(() =>
      controller.subscribe(() => {
        throw new Error('window closed')
      })
    ).not.toThrow()
    controller.subscribe(next => healthyStates.push(next.operation.phase))

    await expect(controller.setEnabled(true)).resolves.toMatchObject({ enabled: true, available: true })
    expect(healthyStates.at(-1)).toBe('ready')
  })
})
