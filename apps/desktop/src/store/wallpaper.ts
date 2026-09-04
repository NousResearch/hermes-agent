import { atom, computed } from 'nanostores'

import type { DesktopWallpaperAsset } from '@/global'
import { readJson, writeJson } from '@/lib/storage'
import {
  DEFAULT_MANUAL_WALLPAPER_PALETTE,
  DEFAULT_WALLPAPER_PREFERENCES,
  sanitizeWallpaperPreferences,
  type WallpaperPreferences,
  wallpaperStorageKey
} from '@/lib/wallpaper'
import type { WallpaperPalette } from '@/lib/wallpaper-palette'
import { $desktopBoot } from '@/store/boot'
import { $activeGatewayProfile, normalizeProfileKey } from '@/store/profile'
import { isHudWindow } from '@/store/windows'

export interface WallpaperState {
  asset: DesktopWallpaperAsset | null
  error: boolean
  paletteStatus: 'error' | 'idle' | 'loading' | 'ready'
  preferences: WallpaperPreferences
  profile: string
  status: 'error' | 'idle' | 'loading' | 'ready' | 'removing' | 'selecting'
  supported: boolean
}

const IDLE_LOAD_TIMEOUT_MS = 3_000
const NO_IDLE_CALLBACK_DELAY_MS = 400
const PERSIST_DELAY_MS = 160
const persistTimers = new Map<string, ReturnType<typeof setTimeout>>()
const pendingPreferences = new Map<string, WallpaperPreferences>()
let requestGeneration = 0
let paletteRequestGeneration = 0
let cancelDeferredRefresh: (() => void) | null = null

export function readWallpaperPreferences(profile: string): WallpaperPreferences {
  const key = normalizeProfileKey(profile)

  return pendingPreferences.get(key) ?? sanitizeWallpaperPreferences(readJson(wallpaperStorageKey(key)))
}

export function persistPreferences(profile: string, preferences: WallpaperPreferences): void {
  writeJson(wallpaperStorageKey(normalizeProfileKey(profile)), preferences)
}

export function schedulePreferences(profile: string, preferences: WallpaperPreferences): void {
  const prior = persistTimers.get(profile)

  if (prior) {
    clearTimeout(prior)
  }

  pendingPreferences.set(profile, preferences)
  persistTimers.set(
    profile,
    setTimeout(() => {
      persistTimers.delete(profile)
      pendingPreferences.delete(profile)
      persistPreferences(profile, preferences)
    }, PERSIST_DELAY_MS)
  )
}

export function cancelScheduledPreferences(profile: string): void {
  const prior = persistTimers.get(profile)

  if (prior) {
    clearTimeout(prior)
  }

  persistTimers.delete(profile)
  pendingPreferences.delete(profile)
}

function flushScheduledPreferences(): void {
  for (const [profile, preferences] of pendingPreferences) {
    cancelScheduledPreferences(profile)
    persistPreferences(profile, preferences)
  }
}

const initialProfile = normalizeProfileKey($activeGatewayProfile.get())
const wallpaperSupported = Boolean(window.hermesDesktop?.wallpaper)

export const $wallpaper = atom<WallpaperState>({
  asset: null,
  error: false,
  paletteStatus: 'idle',
  preferences: readWallpaperPreferences(initialProfile),
  profile: initialProfile,
  status: wallpaperSupported ? 'idle' : 'ready',
  supported: wallpaperSupported
})

export const $wallpaperActive = computed(
  $wallpaper,
  state => Boolean(state.asset && state.preferences.enabled) && state.status !== 'removing'
)

export interface WallpaperVisualState {
  asset: DesktopWallpaperAsset | null
  preferences: WallpaperPreferences
}

let wallpaperVisualState: WallpaperVisualState = {
  asset: $wallpaper.get().asset,
  preferences: $wallpaper.get().preferences
}

export const $wallpaperVisual = computed($wallpaper, state => {
  if (wallpaperVisualState.asset === state.asset && wallpaperVisualState.preferences === state.preferences) {
    return wallpaperVisualState
  }

  wallpaperVisualState = { asset: state.asset, preferences: state.preferences }

  return wallpaperVisualState
})

let themePaletteCacheKey = ''
let themePaletteCache: WallpaperPalette | null = null

function selectedThemePalette(state: WallpaperState): WallpaperPalette | null {
  const { asset, preferences } = state

  if (asset) {
    return preferences.paletteMode === 'manual' ? preferences.manualPalette : cachedPalette(asset, preferences)
  }

  // HUD intentionally skips loading the bitmap. Its persisted palette is
  // already profile-scoped and lets the compact overlay keep the same theme as
  // the main window without fetching or decoding an image it will never draw.
  if (!isHudWindow()) {
    return null
  }

  return preferences.paletteMode === 'manual'
    ? preferences.manualPalette
    : preferences.paletteSource
      ? preferences.palette
      : null
}

export const $wallpaperThemePalette = computed($wallpaper, state => {
  const { preferences } = state
  const palette = selectedThemePalette(state)

  const active = Boolean(preferences.adaptiveTheme && preferences.enabled && palette && state.status !== 'removing')

  const key = active ? `${preferences.paletteMode}\u0000${palette?.accent}\u0000${palette?.dominant}` : ''

  if (key === themePaletteCacheKey) {
    return themePaletteCache
  }

  themePaletteCacheKey = key
  themePaletteCache = active ? palette : null

  return themePaletteCache
})

export const $wallpaperThemePaletteMode = computed($wallpaper, state => {
  const { preferences } = state
  const palette = selectedThemePalette(state)

  return preferences.adaptiveTheme && preferences.enabled && palette && state.status !== 'removing'
    ? preferences.paletteMode
    : null
})

export function cancelDeferredWallpaperRefresh(): void {
  cancelDeferredRefresh?.()
  cancelDeferredRefresh = null
}

function scheduleAfterPaintAndIdle(run: () => void): () => void {
  let canceled = false
  let firstFrame: number | null = null
  let secondFrame: number | null = null
  let idleCallback: number | null = null
  let fallbackTimeout: number | null = null

  const runUnlessCanceled = () => {
    if (!canceled) {
      run()
    }
  }

  const scheduleIdle = () => {
    if (canceled) {
      return
    }

    if (typeof window.requestIdleCallback === 'function') {
      idleCallback = window.requestIdleCallback(runUnlessCanceled, { timeout: IDLE_LOAD_TIMEOUT_MS })

      return
    }

    fallbackTimeout = window.setTimeout(runUnlessCanceled, NO_IDLE_CALLBACK_DELAY_MS)
  }

  if (typeof window.requestAnimationFrame === 'function') {
    firstFrame = window.requestAnimationFrame(() => {
      firstFrame = null
      secondFrame = window.requestAnimationFrame(() => {
        secondFrame = null
        scheduleIdle()
      })
    })
  } else {
    fallbackTimeout = window.setTimeout(scheduleIdle, NO_IDLE_CALLBACK_DELAY_MS)
  }

  return () => {
    canceled = true

    if (firstFrame !== null) {
      window.cancelAnimationFrame(firstFrame)
    }

    if (secondFrame !== null) {
      window.cancelAnimationFrame(secondFrame)
    }

    if (idleCallback !== null) {
      window.cancelIdleCallback(idleCallback)
    }

    if (fallbackTimeout !== null) {
      window.clearTimeout(fallbackTimeout)
    }
  }
}

export async function decodeWallpaperAsset(asset: DesktopWallpaperAsset): Promise<HTMLImageElement> {
  const image = new Image()

  image.decoding = 'async'

  if (typeof image.decode === 'function') {
    image.src = asset.url
    await image.decode()

    return image
  }

  await new Promise<void>((resolve, reject) => {
    image.onload = () => resolve()
    image.onerror = () => reject(new Error('Wallpaper could not be decoded.'))
    image.src = asset.url
  })

  return image
}

export function cachedPalette(
  asset: DesktopWallpaperAsset,
  preferences: WallpaperPreferences
): WallpaperPalette | null {
  return preferences.paletteSource === (asset.version || asset.url) ? preferences.palette : null
}

export function manualPalette(
  asset: DesktopWallpaperAsset | null,
  preferences: WallpaperPreferences
): WallpaperPalette {
  return (
    preferences.manualPalette ?? (asset ? cachedPalette(asset, preferences) : null) ?? DEFAULT_MANUAL_WALLPAPER_PALETTE
  )
}

export async function preferencesWithPalette(
  profile: string,
  preferences: WallpaperPreferences,
  asset: DesktopWallpaperAsset
): Promise<WallpaperPreferences | null> {
  const palette = await window.hermesDesktop?.wallpaper?.palette?.(profile)

  return palette
    ? sanitizeWallpaperPreferences({ ...preferences, palette, paletteSource: asset.version || asset.url })
    : null
}

function scheduleDeferredWallpaperRefresh(profile: string): void {
  cancelDeferredWallpaperRefresh()

  const state = $wallpaper.get()

  // HUD deliberately has no image backdrop: it floats over another app and
  // only its compact transcript glass should occupy the transparent window.
  // Do not fetch/decode a bitmap that this renderer will never paint.
  if (isHudWindow() || !state.supported || state.profile !== profile || !state.preferences.enabled) {
    return
  }

  let canceled = false
  let cancelPaintAndIdle: (() => void) | null = null
  let stopBootListener: (() => void) | null = null

  const schedule = () => {
    if (canceled) {
      return
    }

    stopBootListener?.()
    stopBootListener = null
    cancelPaintAndIdle = scheduleAfterPaintAndIdle(() => {
      cancelDeferredRefresh = null

      const current = $wallpaper.get()

      if (current.profile === profile && current.preferences.enabled) {
        void refreshWallpaper(profile)
      }
    })
  }

  const boot = $desktopBoot.get()

  if (boot.phase === 'renderer.ready' && !boot.running) {
    schedule()
  } else {
    stopBootListener = $desktopBoot.listen(next => {
      if (next.phase === 'renderer.ready' && !next.running) {
        schedule()
      }
    })
  }

  cancelDeferredRefresh = () => {
    canceled = true
    stopBootListener?.()
    cancelPaintAndIdle?.()
  }
}

interface WallpaperRefreshOptions {
  loadWhenDisabled?: boolean
}

export function beginWallpaperRequest(): number {
  cancelDeferredWallpaperRefresh()

  return ++requestGeneration
}

export function invalidateWallpaperRequests(): void {
  cancelDeferredWallpaperRefresh()
  requestGeneration += 1
}

export function wallpaperRequestIsCurrent(generation: number, profile: string): boolean {
  return generation === requestGeneration && $wallpaper.get().profile === profile
}

export function beginWallpaperPaletteRequest(): number {
  return ++paletteRequestGeneration
}

export function invalidateWallpaperPaletteRequests(): void {
  paletteRequestGeneration += 1
}

export function wallpaperPaletteRequestIsCurrent(generation: number): boolean {
  return generation === paletteRequestGeneration
}

export async function refreshWallpaper(
  profile = normalizeProfileKey($activeGatewayProfile.get()),
  options: WallpaperRefreshOptions = {}
): Promise<void> {
  const normalizedProfile = normalizeProfileKey(profile)
  const bridge = window.hermesDesktop?.wallpaper
  const current = $wallpaper.get()
  const preferences = current.profile === normalizedProfile ? current.preferences : readWallpaperPreferences(profile)

  if (!bridge) {
    if (normalizedProfile === normalizeProfileKey($activeGatewayProfile.get())) {
      $wallpaper.set({
        asset: null,
        error: false,
        paletteStatus: 'idle',
        preferences,
        profile: normalizedProfile,
        status: 'ready',
        supported: false
      })
    }

    return
  }

  if (!options.loadWhenDisabled && !preferences.enabled) {
    if (normalizedProfile === normalizeProfileKey($activeGatewayProfile.get())) {
      $wallpaper.set({
        asset: null,
        error: false,
        paletteStatus: 'idle',
        preferences,
        profile: normalizedProfile,
        status: 'idle',
        supported: true
      })
    }

    return
  }

  const generation = ++requestGeneration

  $wallpaper.set({
    asset: null,
    error: false,
    paletteStatus: 'idle',
    preferences,
    profile: normalizedProfile,
    status: 'loading',
    supported: true
  })

  try {
    const asset = await bridge.get(normalizedProfile)

    if (!wallpaperRequestIsCurrent(generation, normalizedProfile)) {
      return
    }

    if (asset) {
      await decodeWallpaperAsset(asset)

      if (!wallpaperRequestIsCurrent(generation, normalizedProfile)) {
        return
      }
    }

    const loadedPreferences = $wallpaper.get().preferences
    let withPalette: WallpaperPreferences | null = null

    if (
      asset &&
      loadedPreferences.adaptiveTheme &&
      loadedPreferences.paletteMode === 'auto' &&
      !cachedPalette(asset, loadedPreferences)
    ) {
      withPalette = await preferencesWithPalette(normalizedProfile, loadedPreferences, asset).catch(() => null)

      if (!wallpaperRequestIsCurrent(generation, normalizedProfile)) {
        return
      }
    }

    // Only the palette belongs to this request. Sliders, mode switches and
    // profile resets may have changed while native work was in flight.
    const latest = $wallpaper.get()
    let nextPreferences = latest.preferences
    let paletteStatus: WallpaperState['paletteStatus'] = 'idle'

    if (withPalette?.palette) {
      nextPreferences = sanitizeWallpaperPreferences({
        ...nextPreferences,
        palette: withPalette.palette,
        paletteSource: withPalette.paletteSource
      })
    }

    if (asset && nextPreferences.adaptiveTheme) {
      if (nextPreferences.paletteMode === 'manual') {
        if (!nextPreferences.manualPalette) {
          nextPreferences = sanitizeWallpaperPreferences({
            ...nextPreferences,
            manualPalette: manualPalette(asset, nextPreferences)
          })
        }

        paletteStatus = 'ready'
      } else {
        paletteStatus = cachedPalette(asset, nextPreferences) ? 'ready' : 'error'
      }
    }

    if (nextPreferences !== latest.preferences) {
      cancelScheduledPreferences(normalizedProfile)
      persistPreferences(normalizedProfile, nextPreferences)
    }

    $wallpaper.set({ ...latest, asset, paletteStatus, preferences: nextPreferences, status: 'ready' })
  } catch {
    if (wallpaperRequestIsCurrent(generation, normalizedProfile)) {
      $wallpaper.set({ ...$wallpaper.get(), error: true, status: 'error' })
    }
  }
}

export function setWallpaperPreferences(patch: Partial<WallpaperPreferences>): void {
  const state = $wallpaper.get()
  const preferences = sanitizeWallpaperPreferences({ ...state.preferences, ...patch })
  const disabling = state.preferences.enabled && !preferences.enabled
  const enabling = !state.preferences.enabled && preferences.enabled

  if (disabling) {
    cancelDeferredWallpaperRefresh()
    requestGeneration += 1
  }

  $wallpaper.set({
    ...state,
    error: false,
    preferences,
    status: disabling && state.status === 'loading' ? 'idle' : state.status
  })
  schedulePreferences(state.profile, preferences)

  if (enabling && !state.asset) {
    void refreshWallpaper(state.profile, { loadWhenDisabled: true })
  }
}

let profileSubscriptionReady = false

function activateWallpaperProfile(profile: string): void {
  cancelDeferredWallpaperRefresh()
  requestGeneration += 1
  paletteRequestGeneration += 1

  const preferences = readWallpaperPreferences(profile)

  $wallpaper.set({
    asset: null,
    error: false,
    paletteStatus: 'idle',
    preferences,
    profile,
    status: wallpaperSupported ? 'idle' : 'ready',
    supported: wallpaperSupported
  })

  if (preferences.enabled) {
    scheduleDeferredWallpaperRefresh(profile)
  }
}

$activeGatewayProfile.subscribe(profile => {
  const normalizedProfile = normalizeProfileKey(profile)

  if (!profileSubscriptionReady) {
    profileSubscriptionReady = true
    scheduleDeferredWallpaperRefresh(normalizedProfile)

    return
  }

  activateWallpaperProfile(normalizedProfile)
})

window.addEventListener('pagehide', () => {
  cancelDeferredWallpaperRefresh()
  flushScheduledPreferences()
})

window.hermesDesktop?.wallpaper?.onProfileReset?.(profile => {
  const normalizedProfile = normalizeProfileKey(profile)

  cancelScheduledPreferences(normalizedProfile)
  writeJson(wallpaperStorageKey(normalizedProfile), null)

  const state = $wallpaper.get()

  if (state.profile === normalizedProfile) {
    requestGeneration += 1
    paletteRequestGeneration += 1
    $wallpaper.set({
      ...state,
      asset: null,
      error: false,
      paletteStatus: 'idle',
      preferences: { ...DEFAULT_WALLPAPER_PREFERENCES },
      status: 'ready'
    })
  }
})
