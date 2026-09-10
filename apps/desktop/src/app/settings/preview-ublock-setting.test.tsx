// @vitest-environment jsdom
import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { $previewUblock } from '@/store/preview-ublock'

import { PreviewUblockSetting } from './preview-ublock-setting'

const copy = {
  description:
    'Downloads uBlock Origin Lite from its official GitHub release, blocks ads and trackers only in Preview, and keeps it locally for later use.',
  downloading: 'Downloading and enabling uBlock Origin Lite…',
  failure: 'Could not update the Preview content-blocking setting.',
  label: 'Enable uBlock Origin Lite in Preview',
  progress: {
    checkingCache: 'Checking cached release…',
    downloading: 'Downloading uBlock Origin Lite…',
    extracting: 'Extracting release…',
    loading: 'Loading Preview extension…',
    preparing: 'Preparing Electron compatibility…',
    validating: 'Testing content blocking…',
    verifying: 'Verifying official release…',
    enabled: (version: string) => `Enabled · ${version}`,
    transferred: (amount: string) => `${amount} transferred`,
    downloadProgress: 'uBlock Origin Lite download progress',
    retry: 'Retry'
  }
}

function state(enabled: boolean, operation?: Record<string, unknown>) {
  const defaultOperation = {
    failure: null,
    operationId: null,
    phase: enabled ? ('ready' as const) : ('idle' as const),
    receivedBytes: 0,
    totalBytes: null
  }

  return {
    enabled,
    available: enabled,
    dashboardUrl: enabled ? 'chrome-extension://ublock/dashboard.html' : null,
    extensionId: enabled ? 'ublock' : null,
    operation: { ...defaultOperation, ...operation },
    popupUrl: enabled ? 'chrome-extension://ublock/popup.html' : null,
    rulesetsReady: enabled,
    version: enabled ? '2026.825.1619' : null
  }
}

vi.mock('@/i18n', () => ({
  useI18n: () => ({
    t: {
      settings: {
        config: {
          previewUblockDescription: copy.description,
          previewUblockDownloading: copy.downloading,
          previewUblockFailure: copy.failure,
          previewUblockTitle: copy.label,
          previewUblock: copy.progress
        }
      }
    }
  })
}))

describe('PreviewUblockSetting', () => {
  beforeEach(() => {
    $previewUblock.set(state(false))
    window.hermesDesktop = {
      previewUblock: {
        getState: vi.fn().mockResolvedValue(state(false)),
        setEnabled: vi.fn().mockResolvedValue(state(false))
      }
    } as any
  })

  afterEach(() => {
    cleanup()
    vi.clearAllMocks()
  })

  it('renders disabled by default', () => {
    render(<PreviewUblockSetting />)

    expect(screen.getByRole('switch', { name: copy.label }).getAttribute('data-state')).toBe('unchecked')
    expect(screen.getByText(copy.description)).toBeTruthy()
  })

  it('shows the download status while enabling', async () => {
    let resolve: ((value: ReturnType<typeof state>) => void) | undefined
    $previewUblock.set(state(false))
    window.hermesDesktop.previewUblock.setEnabled = vi.fn().mockImplementation(
      () =>
        new Promise<ReturnType<typeof state>>(nextResolve => {
          resolve = nextResolve
        })
    )

    render(<PreviewUblockSetting />)
    const toggle = screen.getByRole('switch', { name: copy.label })
    fireEvent.click(toggle)

    expect(toggle.hasAttribute('disabled')).toBe(true)
    expect(screen.getByText('Checking cached release…')).toBeTruthy()
    expect(toggle.getAttribute('data-state')).toBe('unchecked')

    await act(async () => resolve?.(state(true)))
    expect(toggle.getAttribute('data-state')).toBe('checked')
  })

  it.each([
    ['checking-cache', copy.progress.checkingCache],
    ['downloading', copy.progress.downloading],
    ['extracting', copy.progress.extracting],
    ['loading', copy.progress.loading],
    ['preparing', copy.progress.preparing],
    ['validating', copy.progress.validating],
    ['verifying', copy.progress.verifying]
  ] as const)('renders the %s phase from localized copy', async (phase, label) => {
    const current = state(false, { phase })
    $previewUblock.set(current)
    window.hermesDesktop.previewUblock.getState = vi.fn().mockResolvedValue(current)

    render(<PreviewUblockSetting />)

    await act(async () => {
      await Promise.resolve()
    })

    expect(screen.getByText(label)).toBeTruthy()
  })

  it('supports determinate and indeterminate download progress', async () => {
    const indeterminate = state(false, { phase: 'downloading', receivedBytes: 12, totalBytes: null })
    $previewUblock.set(indeterminate)
    window.hermesDesktop.previewUblock.getState = vi.fn().mockResolvedValue(indeterminate)
    const { unmount } = render(<PreviewUblockSetting />)

    await act(async () => {
      await Promise.resolve()
    })

    expect(
      screen.getByRole('progressbar', { name: copy.progress.downloadProgress }).getAttribute('aria-valuenow')
    ).toBe(null)
    expect(screen.getByText('12 B transferred')).toBeTruthy()

    unmount()

    const determinate = state(false, { phase: 'downloading', receivedBytes: 12, totalBytes: 100 })
    $previewUblock.set(determinate)
    window.hermesDesktop.previewUblock.getState = vi.fn().mockResolvedValue(determinate)
    render(<PreviewUblockSetting />)

    await act(async () => {
      await Promise.resolve()
    })

    expect(
      screen.getByRole('progressbar', { name: copy.progress.downloadProgress }).getAttribute('aria-valuenow')
    ).toBe('12')
    expect(screen.getByText('12 B / 100 B')).toBeTruthy()
  })

  it('disables the switch while turning uBlock off and mirrors the result', async () => {
    let resolve: ((value: ReturnType<typeof state>) => void) | undefined

    const setEnabled = vi.fn().mockImplementation(
      () =>
        new Promise<ReturnType<typeof state>>(nextResolve => {
          resolve = nextResolve
        })
    )

    $previewUblock.set(state(true))
    window.hermesDesktop.previewUblock.getState = vi.fn().mockResolvedValue(state(true))
    window.hermesDesktop.previewUblock.setEnabled = setEnabled

    render(<PreviewUblockSetting />)
    const toggle = screen.getByRole('switch', { name: copy.label })

    fireEvent.click(toggle)
    expect(setEnabled).toHaveBeenCalledWith(false)
    expect(toggle.hasAttribute('disabled')).toBe(true)

    await act(async () => {
      resolve?.(state(false))
    })

    expect(toggle.getAttribute('data-state')).toBe('unchecked')
    expect(toggle.hasAttribute('disabled')).toBe(false)
    expect($previewUblock.get()).toMatchObject({ enabled: false, available: false })
  })

  it('keeps a failed enable inline and offers retry without a toast', async () => {
    const failed = {
      ...state(false),
      operation: {
        failure: { code: 'network', message: 'offline' },
        operationId: 'operation-1',
        phase: 'failed',
        receivedBytes: 0,
        totalBytes: null
      }
    }

    const setEnabled = vi.fn().mockResolvedValueOnce(failed).mockResolvedValueOnce(state(true))
    window.hermesDesktop.previewUblock.setEnabled = setEnabled

    render(<PreviewUblockSetting />)
    fireEvent.click(screen.getByRole('switch', { name: copy.label }))

    await act(async () => {
      await Promise.resolve()
    })

    expect(screen.getByRole('switch', { name: copy.label }).getAttribute('data-state')).toBe('unchecked')
    expect(screen.getByRole('alert').textContent).toContain('offline')
    fireEvent.click(screen.getByRole('button', { name: 'Retry' }))

    await act(async () => {
      await Promise.resolve()
    })

    expect(setEnabled).toHaveBeenNthCalledWith(2, true)
    expect(screen.getByRole('switch', { name: copy.label }).getAttribute('data-state')).toBe('checked')
  })

  it('shows a handled transport failure, re-enables the switch, and clears it on authoritative update', async () => {
    let publishState: ((next: ReturnType<typeof state>) => void) | undefined
    const setEnabled = vi
      .fn()
      .mockRejectedValueOnce(new Error('private transport details'))
      .mockResolvedValueOnce(state(true))
    window.hermesDesktop.previewUblock.setEnabled = setEnabled
    window.hermesDesktop.previewUblock.onState = vi.fn(callback => {
      publishState = callback

      return vi.fn()
    })

    render(<PreviewUblockSetting />)
    const toggle = screen.getByRole('switch', { name: copy.label })
    fireEvent.click(toggle)

    await act(async () => {
      await Promise.resolve()
    })

    expect(toggle.hasAttribute('disabled')).toBe(false)
    expect(screen.getByRole('alert').textContent).toContain(copy.failure)
    expect(screen.getByRole('alert').textContent).not.toContain('private transport details')

    fireEvent.click(screen.getByRole('button', { name: copy.progress.retry }))
    await act(async () => {
      await Promise.resolve()
    })

    expect(setEnabled).toHaveBeenNthCalledWith(2, true)
    expect(toggle.getAttribute('data-state')).toBe('checked')

    await act(async () => {
      publishState?.(state(false, { phase: 'idle' }))
    })

    expect(screen.queryByRole('alert')).toBeNull()
  })
})
