import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import {
  $dataUrlReadMaxMb,
  clampDataUrlReadMaxMb,
  DATA_URL_READ_DEFAULT_MAX_MB,
  refreshDataUrlReadMaxMb,
  setDataUrlReadMaxMb
} from './data-url-read-max'

const desktopWindow = window as unknown as { hermesDesktop?: Window['hermesDesktop'] }
const initialHermesDesktop = desktopWindow.hermesDesktop

const get = vi.fn(async () => ({ defaultMaxMb: 16, maxBytes: 16 * 1024 * 1024, maxMb: 16 }))

const set = vi.fn(async (maxMb: number) => ({
  defaultMaxMb: 16,
  maxBytes: maxMb * 1024 * 1024,
  maxMb
}))

beforeEach(() => {
  desktopWindow.hermesDesktop = { dataUrlReadMax: { get, set } } as unknown as Window['hermesDesktop']
  $dataUrlReadMaxMb.set(DATA_URL_READ_DEFAULT_MAX_MB)
  get.mockClear()
  set.mockClear()
})

afterEach(() => {
  desktopWindow.hermesDesktop = initialHermesDesktop
})

describe('clampDataUrlReadMaxMb', () => {
  it('defaults invalid input and clamps only the safety range', () => {
    expect(clampDataUrlReadMaxMb(undefined)).toBe(16)
    expect(clampDataUrlReadMaxMb(0)).toBe(1)
    expect(clampDataUrlReadMaxMb(100)).toBe(100)
    expect(clampDataUrlReadMaxMb(99999)).toBe(4096)
    expect(clampDataUrlReadMaxMb(32.4)).toBe(32)
  })
})

describe('data-url-read-max store', () => {
  it('loads the main-process value into the atom', async () => {
    get.mockResolvedValueOnce({ defaultMaxMb: 16, maxBytes: 32 * 1024 * 1024, maxMb: 32 })

    await expect(refreshDataUrlReadMaxMb()).resolves.toBe(32)
    expect($dataUrlReadMaxMb.get()).toBe(32)
  })

  it('writes through the bridge and updates the atom', async () => {
    await expect(setDataUrlReadMaxMb(48)).resolves.toBe(48)
    expect(set).toHaveBeenCalledWith(48)
    expect($dataUrlReadMaxMb.get()).toBe(48)
  })

  it('keeps the last known-good value when the bridge write fails', async () => {
    set.mockRejectedValueOnce(new Error('boom'))

    await expect(setDataUrlReadMaxMb(48)).resolves.toBe(16)
    expect($dataUrlReadMaxMb.get()).toBe(16)
  })
})
