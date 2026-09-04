import { atom } from 'nanostores'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { DEFAULT_WALLPAPER_PREFERENCES, wallpaperStorageKey } from '@/lib/wallpaper'
import { deferred } from '@/test/deferred'

const wallpaperAsset = { url: 'hermes-wallpaper://asset/37a8eec1ce19687d132fe290?v=1&token=secret', version: '1' }

let boot = atom({ phase: 'renderer.init', running: true })
let profile = atom('default')
let frameCallbacks: FrameRequestCallback[] = []
let idleCallbacks: IdleRequestCallback[] = []
let resolveDecode: () => void = () => undefined
let decodePromise = Promise.resolve()
let hudWindow = false
let profileResetCallback: ((profile: string) => void) | null = null

const getWallpaper = vi.fn(async () => wallpaperAsset)
const selectWallpaper = vi.fn()
const removeWallpaper = vi.fn()
const decodeImage = vi.fn(() => decodePromise)
const extractPalette = vi.fn(async () => ({ accent: '#e63658', dominant: '#84888e' }))

function flushFrame(): void {
  const callbacks = frameCallbacks.splice(0)

  for (const callback of callbacks) {
    callback(performance.now())
  }
}

function flushIdle(): void {
  const callbacks = idleCallbacks.splice(0)

  for (const callback of callbacks) {
    callback({ didTimeout: false, timeRemaining: () => 50 })
  }
}

async function loadWallpaperStore(enabled: boolean, preferences: Partial<typeof DEFAULT_WALLPAPER_PREFERENCES> = {}) {
  window.localStorage.setItem(
    wallpaperStorageKey('default'),
    JSON.stringify({ ...DEFAULT_WALLPAPER_PREFERENCES, ...preferences, enabled })
  )

  vi.doMock('@/store/boot', () => ({ $desktopBoot: boot }))
  vi.doMock('@/store/profile', () => ({
    $activeGatewayProfile: profile,
    normalizeProfileKey: (value: null | string | undefined) => value?.trim() || 'default'
  }))
  vi.doMock('@/store/windows', () => ({ isHudWindow: () => hudWindow }))
  Object.defineProperty(window, 'hermesDesktop', {
    configurable: true,
    value: {
      wallpaper: {
        get: getWallpaper,
        onProfileReset: (callback: (profile: string) => void) => {
          profileResetCallback = callback

          return () => {
            profileResetCallback = null
          }
        },
        palette: extractPalette,
        remove: removeWallpaper,
        select: selectWallpaper
      }
    }
  })

  const store = await import('./wallpaper')
  const actions = await import('./wallpaper-actions')

  return { ...actions, ...store }
}

beforeEach(() => {
  vi.resetModules()
  window.localStorage.clear()

  boot = atom({ phase: 'renderer.init', running: true })
  profile = atom('default')
  frameCallbacks = []
  idleCallbacks = []
  hudWindow = false
  profileResetCallback = null
  decodePromise = new Promise<void>(resolve => {
    resolveDecode = resolve
  })

  getWallpaper.mockClear()
  selectWallpaper.mockReset()
  removeWallpaper.mockReset()
  decodeImage.mockClear()
  extractPalette.mockClear()

  vi.stubGlobal(
    'Image',
    class {
      decode = decodeImage
      decoding = 'auto'
      onerror: null | (() => void) = null
      onload: null | (() => void) = null
      src = ''

      set crossOrigin(_value: string) {
        throw new Error('Backdrop decoding must not opt into CORS mode.')
      }
    }
  )
  vi.stubGlobal('requestAnimationFrame', (callback: FrameRequestCallback) => {
    frameCallbacks.push(callback)

    return frameCallbacks.length
  })
  vi.stubGlobal('cancelAnimationFrame', vi.fn())
  vi.stubGlobal('requestIdleCallback', (callback: IdleRequestCallback) => {
    idleCallbacks.push(callback)

    return idleCallbacks.length
  })
  vi.stubGlobal('cancelIdleCallback', vi.fn())
})

afterEach(() => {
  vi.unstubAllGlobals()
  vi.doUnmock('@/store/boot')
  vi.doUnmock('@/store/profile')
  vi.doUnmock('@/store/windows')
})

describe('wallpaper startup loading', () => {
  it('does no wallpaper IPC when the persisted feature is disabled', async () => {
    const { $wallpaper } = await loadWallpaperStore(false)

    boot.set({ phase: 'renderer.ready', running: false })
    flushFrame()
    flushFrame()
    flushIdle()
    await Promise.resolve()

    expect(getWallpaper).not.toHaveBeenCalled()
    expect(decodeImage).not.toHaveBeenCalled()
    expect($wallpaper.get()).toMatchObject({ asset: null, status: 'idle' })
  })

  it('does not load or decode wallpaper images in the HUD window', async () => {
    hudWindow = true

    const palette = { accent: '#e63658', dominant: '#84888e' }

    const { $wallpaper, $wallpaperThemePalette } = await loadWallpaperStore(true, {
      adaptiveTheme: true,
      palette,
      paletteSource: wallpaperAsset.version
    })

    boot.set({ phase: 'renderer.ready', running: false })
    flushFrame()
    flushFrame()
    flushIdle()
    await Promise.resolve()

    expect(getWallpaper).not.toHaveBeenCalled()
    expect(decodeImage).not.toHaveBeenCalled()
    expect($wallpaper.get()).toMatchObject({ asset: null, status: 'idle' })
    expect($wallpaperThemePalette.get()).toEqual(palette)
  })

  it('waits for boot, two paints, idle time, and image decode before showing the wallpaper', async () => {
    const { $wallpaper } = await loadWallpaperStore(true)

    expect(getWallpaper).not.toHaveBeenCalled()

    boot.set({ phase: 'renderer.ready', running: false })
    expect(getWallpaper).not.toHaveBeenCalled()

    flushFrame()
    flushFrame()
    expect(getWallpaper).not.toHaveBeenCalled()

    flushIdle()
    await Promise.resolve()
    await Promise.resolve()

    expect(getWallpaper).toHaveBeenCalledOnce()
    expect(decodeImage).toHaveBeenCalledOnce()
    expect($wallpaper.get()).toMatchObject({ asset: null, status: 'loading' })

    resolveDecode()
    await decodePromise
    await Promise.resolve()

    expect($wallpaper.get()).toMatchObject({ asset: wallpaperAsset, status: 'ready' })
  })

  it('loads on demand from settings even when startup loading was disabled', async () => {
    const { $wallpaper, ensureWallpaperLoaded } = await loadWallpaperStore(false)

    const loading = ensureWallpaperLoaded()

    await Promise.resolve()
    expect(getWallpaper).toHaveBeenCalledOnce()
    expect($wallpaper.get()).toMatchObject({ asset: null, status: 'loading' })

    resolveDecode()
    await loading

    expect($wallpaper.get()).toMatchObject({ asset: wallpaperAsset, status: 'ready' })
  })

  it('extracts an adaptive palette once after decode and reuses the persisted cache', async () => {
    const first = await loadWallpaperStore(true, { adaptiveTheme: true })

    boot.set({ phase: 'renderer.ready', running: false })
    flushFrame()
    flushFrame()
    flushIdle()
    await Promise.resolve()
    await Promise.resolve()

    expect(extractPalette).not.toHaveBeenCalled()

    resolveDecode()
    await decodePromise
    await vi.waitFor(() => {
      expect(extractPalette).toHaveBeenCalledOnce()
      expect(first.$wallpaper.get()).toMatchObject({
        paletteStatus: 'ready',
        preferences: {
          adaptiveTheme: true,
          palette: { accent: '#e63658', dominant: '#84888e' },
          paletteSource: wallpaperAsset.version
        }
      })
    })
    expect(first.$wallpaperThemePalette.get()).toEqual({ accent: '#e63658', dominant: '#84888e' })

    const persisted = JSON.parse(window.localStorage.getItem(wallpaperStorageKey('default')) ?? '{}')

    expect(persisted.paletteSource).toBe(wallpaperAsset.version)
  })

  it('does not resample a cached palette and removes it from the live theme when disabled', async () => {
    const palette = { accent: '#e63658', dominant: '#84888e' }

    const store = await loadWallpaperStore(true, {
      adaptiveTheme: true,
      palette,
      paletteSource: wallpaperAsset.version
    })

    boot.set({ phase: 'renderer.ready', running: false })
    flushFrame()
    flushFrame()
    flushIdle()
    await Promise.resolve()
    await Promise.resolve()
    resolveDecode()
    await decodePromise
    await Promise.resolve()

    expect(extractPalette).not.toHaveBeenCalled()
    expect(store.$wallpaperThemePalette.get()).toEqual(palette)

    const stablePalette = store.$wallpaperThemePalette.get()

    store.setWallpaperPreferences({ opacity: 64 })
    expect(store.$wallpaperThemePalette.get()).toBe(stablePalette)

    await store.setWallpaperAdaptiveTheme(false)

    expect(store.$wallpaperThemePalette.get()).toBeNull()
  })

  it('preserves visual state identity when only non-rendering status changes', async () => {
    const store = await loadWallpaperStore(false)
    const visualState = store.$wallpaperVisual.get()

    store.$wallpaper.set({ ...store.$wallpaper.get(), paletteStatus: 'loading', status: 'loading' })

    expect(store.$wallpaperVisual.get()).toBe(visualState)
  })

  it('switches between cached automatic colors and editable manual colors without resampling', async () => {
    const automatic = { accent: '#e63658', dominant: '#84888e' }
    const manual = { accent: '#2468ac', dominant: '#465768' }

    const store = await loadWallpaperStore(true, {
      adaptiveTheme: true,
      palette: automatic,
      paletteSource: wallpaperAsset.version
    })

    boot.set({ phase: 'renderer.ready', running: false })
    flushFrame()
    flushFrame()
    flushIdle()
    await Promise.resolve()
    await Promise.resolve()
    resolveDecode()
    await decodePromise
    await Promise.resolve()

    await store.setWallpaperPaletteMode('manual')

    expect(extractPalette).not.toHaveBeenCalled()
    expect(store.$wallpaper.get()).toMatchObject({
      paletteStatus: 'ready',
      preferences: { manualPalette: automatic, paletteMode: 'manual' }
    })
    expect(store.$wallpaperThemePalette.get()).toEqual(automatic)

    store.setWallpaperPreferences({ manualPalette: manual })

    expect(store.$wallpaperThemePalette.get()).toEqual(manual)

    await store.setWallpaperPaletteMode('auto')

    expect(extractPalette).not.toHaveBeenCalled()
    expect(store.$wallpaper.get()).toMatchObject({
      paletteStatus: 'ready',
      preferences: { manualPalette: manual, paletteMode: 'auto' }
    })
    expect(store.$wallpaperThemePalette.get()).toEqual(automatic)
  })

  it('keeps a selected wallpaper ready when adaptive palette analysis fails', async () => {
    selectWallpaper.mockResolvedValue({ asset: wallpaperAsset, canceled: false })
    extractPalette.mockRejectedValueOnce(new Error('Palette sampling failed.'))

    const store = await loadWallpaperStore(false, { adaptiveTheme: true })
    const selecting = store.selectWallpaper()

    await Promise.resolve()
    expect(store.$wallpaper.get().status).toBe('selecting')

    resolveDecode()
    await selecting

    expect(store.$wallpaper.get()).toMatchObject({
      asset: wallpaperAsset,
      error: false,
      paletteStatus: 'error',
      preferences: { adaptiveTheme: true, enabled: true },
      status: 'ready'
    })
  })

  it('reuses the palette returned by import instead of decoding the saved image again', async () => {
    const palette = { accent: '#e63658', dominant: '#84888e' }

    selectWallpaper.mockResolvedValue({ asset: wallpaperAsset, canceled: false, palette })

    const store = await loadWallpaperStore(false, { adaptiveTheme: true })
    const selecting = store.selectWallpaper()

    await Promise.resolve()
    resolveDecode()
    await selecting

    expect(extractPalette).not.toHaveBeenCalled()
    expect(store.$wallpaper.get()).toMatchObject({
      asset: wallpaperAsset,
      paletteStatus: 'ready',
      preferences: {
        palette,
        paletteSource: wallpaperAsset.version
      },
      status: 'ready'
    })
  })

  it.each(['refresh', 'select'] as const)('stops obsolete %s work at the next async boundary', async operation => {
    const store = await loadWallpaperStore(true, { adaptiveTheme: true })
    const run = () => (operation === 'refresh' ? store.refreshWallpaper() : store.selectWallpaper())
    const importing = { asset: wallpaperAsset, canceled: false }
    const read = deferred<typeof wallpaperAsset>()
    const selected = deferred<typeof importing>()

    getWallpaper.mockReturnValueOnce(read.promise)
    selectWallpaper.mockReturnValueOnce(selected.promise)
    const reading = run()

    profile.set('another-profile')
    const inactivePreferences = window.localStorage.getItem(wallpaperStorageKey('default'))

    read.resolve(wallpaperAsset)
    selected.resolve(importing)
    resolveDecode()
    await reading

    expect(decodeImage).not.toHaveBeenCalled()
    expect(extractPalette).not.toHaveBeenCalled()
    expect(window.localStorage.getItem(wallpaperStorageKey('default'))).toBe(inactivePreferences)

    profile.set('default')
    const decoding = deferred()

    decodeImage.mockReturnValueOnce(decoding.promise)
    selectWallpaper.mockResolvedValueOnce(importing)
    const loading = run()

    await vi.waitFor(() => expect(decodeImage).toHaveBeenCalledOnce())
    profile.set('another-profile')
    decoding.resolve()
    await loading

    expect(extractPalette).not.toHaveBeenCalled()
    expect(store.$wallpaper.get()).toMatchObject({ asset: null, profile: 'another-profile' })
    expect(window.localStorage.getItem(wallpaperStorageKey('default'))).toBe(inactivePreferences)
  })

  it.each(['refresh', 'select'] as const)(
    'merges %s palette results without reverting newer controls',
    async operation => {
      const store = await loadWallpaperStore(true, { adaptiveTheme: true })
      const analysis = deferred<{ accent: string; dominant: string }>()
      const palette = { accent: '#e63658', dominant: '#84888e' }
      const manual = { accent: '#2468ac', dominant: '#465768' }

      extractPalette.mockReturnValueOnce(analysis.promise)
      selectWallpaper.mockResolvedValueOnce({ asset: wallpaperAsset, canceled: false })
      resolveDecode()
      const loading = operation === 'refresh' ? store.refreshWallpaper() : store.selectWallpaper()

      await vi.waitFor(() => expect(extractPalette).toHaveBeenCalledOnce())
      store.setWallpaperPreferences({ opacity: 72 })
      await store.setWallpaperPaletteMode('manual')
      store.setWallpaperPreferences({ manualPalette: manual })
      analysis.resolve(palette)
      await loading

      expect(store.$wallpaper.get()).toMatchObject({
        asset: wallpaperAsset,
        paletteStatus: 'ready',
        preferences: { manualPalette: manual, opacity: 72, palette, paletteMode: 'manual' },
        status: 'ready'
      })
      expect(store.$wallpaperThemePalette.get()).toEqual(manual)
      expect(JSON.parse(window.localStorage.getItem(wallpaperStorageKey('default'))!)).toMatchObject({
        manualPalette: manual,
        opacity: 72,
        paletteMode: 'manual'
      })
    }
  )

  it('does not restore a reset profile when its old palette request finishes', async () => {
    const store = await loadWallpaperStore(true, { adaptiveTheme: true })
    const analysis = deferred<{ accent: string; dominant: string }>()

    extractPalette.mockReturnValueOnce(analysis.promise)
    resolveDecode()
    const loading = store.refreshWallpaper()

    await vi.waitFor(() => expect(extractPalette).toHaveBeenCalledOnce())
    profileResetCallback?.('default')
    analysis.resolve({ accent: '#e63658', dominant: '#84888e' })
    await loading

    expect(window.localStorage.getItem(wallpaperStorageKey('default'))).toBeNull()
    expect(store.$wallpaper.get()).toMatchObject({ asset: null, preferences: DEFAULT_WALLPAPER_PREFERENCES })
  })

  it('caches import colors even before adaptive theming is enabled', async () => {
    const palette = { accent: '#e63658', dominant: '#84888e' }

    selectWallpaper.mockResolvedValueOnce({ asset: wallpaperAsset, canceled: false, palette })
    const store = await loadWallpaperStore(false)

    resolveDecode()
    await store.selectWallpaper()
    expect(store.$wallpaperThemePalette.get()).toBeNull()
    await store.setWallpaperAdaptiveTheme(true)

    expect(extractPalette).not.toHaveBeenCalled()
    expect(store.$wallpaperThemePalette.get()).toEqual(palette)
    store.cancelScheduledPreferences('default')
  })

  it('keeps coalesced edits when revisiting a profile before the storage write', async () => {
    const store = await loadWallpaperStore(true)

    store.setWallpaperPreferences({ opacity: 72 })
    profile.set('another-profile')
    profile.set('default')

    expect(store.$wallpaper.get().preferences.opacity).toBe(72)
    expect(store.readWallpaperPreferences('default').opacity).toBe(72)
    store.cancelScheduledPreferences('default')
  })

  it('removes local preferences when main resets a deleted or newly recreated profile', async () => {
    const store = await loadWallpaperStore(true, { adaptiveTheme: true })

    store.$wallpaper.set({ ...store.$wallpaper.get(), asset: wallpaperAsset, status: 'ready' })
    expect(window.localStorage.getItem(wallpaperStorageKey('default'))).not.toBeNull()

    profileResetCallback?.('default')

    expect(window.localStorage.getItem(wallpaperStorageKey('default'))).toBeNull()
    expect(store.$wallpaper.get()).toMatchObject({
      asset: null,
      paletteStatus: 'idle',
      preferences: DEFAULT_WALLPAPER_PREFERENCES,
      status: 'ready'
    })
  })
})
