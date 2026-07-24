import { beforeEach, describe, expect, it, vi } from 'vitest'

const getHermesConfigRecord = vi.fn()
const saveHermesConfig = vi.fn()

vi.mock('@/hermes', () => ({
  getHermesConfigRecord: () => getHermesConfigRecord(),
  saveHermesConfig: (config: unknown) => saveHermesConfig(config)
}))

import {
  $desktopStatusbarMode,
  applyDesktopStatusbarFromConfig,
  normalizeDesktopStatusbarMode,
  persistDesktopStatusbarMode
} from './desktop-statusbar'

describe('desktop status bar preference', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    $desktopStatusbarMode.set('on')
  })

  it('normalizes unknown and missing values to the backward-compatible default', () => {
    expect(normalizeDesktopStatusbarMode(undefined)).toBe('on')
    expect(normalizeDesktopStatusbarMode('sometimes')).toBe('on')
    expect(normalizeDesktopStatusbarMode('auto-hide')).toBe('auto-hide')
  })

  it('applies a saved profile preference', () => {
    applyDesktopStatusbarFromConfig({ display: { desktop_statusbar: 'off' } })

    expect($desktopStatusbarMode.get()).toBe('off')
  })

  it('preserves sibling config fields while persisting', async () => {
    getHermesConfigRecord.mockResolvedValue({ display: { language: 'zh', skin: 'slate' }, terminal: { cwd: '/work' } })
    saveHermesConfig.mockResolvedValue({ ok: true })

    const saved = await persistDesktopStatusbarMode('auto-hide')

    expect($desktopStatusbarMode.get()).toBe('auto-hide')
    expect(saved).toEqual({
      display: { desktop_statusbar: 'auto-hide', language: 'zh', skin: 'slate' },
      terminal: { cwd: '/work' }
    })
    expect(saveHermesConfig).toHaveBeenCalledWith(saved)
  })

  it('rolls back the optimistic preference when persistence fails', async () => {
    getHermesConfigRecord.mockRejectedValue(new Error('offline'))

    await expect(persistDesktopStatusbarMode('off')).rejects.toThrow('offline')
    expect($desktopStatusbarMode.get()).toBe('on')
  })
})
