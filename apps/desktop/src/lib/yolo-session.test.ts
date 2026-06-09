import { beforeEach, describe, expect, it, vi } from 'vitest'

import {
  $desktopYoloDefault,
  $yoloActive,
  setDesktopYoloDefaultActive,
  setYoloActive
} from '@/store/session'

import { desktopYoloDefaultFromConfig, setDesktopYoloDefault } from './yolo-session'

describe('desktop YOLO preference', () => {
  beforeEach(() => {
    setDesktopYoloDefaultActive(true)
    setYoloActive(true)
  })

  it('defaults desktop YOLO to on when no preference is saved', () => {
    expect(desktopYoloDefaultFromConfig({})).toBe(true)
    expect(desktopYoloDefaultFromConfig({ desktop: {} })).toBe(true)
  })

  it('honors an explicit saved desktop YOLO preference', () => {
    expect(desktopYoloDefaultFromConfig({ desktop: { yolo_default: false } })).toBe(false)
    expect(desktopYoloDefaultFromConfig({ desktop: { yolo_default: true } })).toBe(true)
  })

  it('persists the desktop YOLO default without dropping other desktop config', async () => {
    const api = vi.fn(async request => {
      if (request.path === '/api/config' && !request.method) {
        return {
          desktop: { frame_rate: 60 },
          display: { skin: 'mono' }
        }
      }

      if (request.path === '/api/config' && request.method === 'PUT') {
        return { ok: true }
      }

      throw new Error(`unexpected request: ${request.path}`)
    })

    Object.defineProperty(window, 'hermesDesktop', {
      configurable: true,
      value: { api }
    })

    await setDesktopYoloDefault(false)

    expect(api).toHaveBeenLastCalledWith({
      path: '/api/config',
      method: 'PUT',
      body: {
        config: {
          desktop: { frame_rate: 60, yolo_default: false },
          display: { skin: 'mono' }
        }
      }
    })
    expect($desktopYoloDefault.get()).toBe(false)
    expect($yoloActive.get()).toBe(true)
  })
})
