import { atom, computed } from 'nanostores'

import type { DesktopWallpaperAsset } from '@/global'
import { readJson, writeJson } from '@/lib/storage'
import {
  DEFAULT_MANUAL_WALLPAPER_PALETTE,
  DEFAULT_WALLPAPER_PREFERENCES,
  sanitizeWallpaperPreferences,
  type WallpaperPaletteMode,
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
  return sanitizeWallpaperPreferences(readJson(wallpaperStorageKey(normalizeProfileKey(profile))))
}

function persistPreferences(profile: string, preferences: WallpaperPreferences): void {
  writeJson(wallpaperStorageKey(normalizeProfileKey(profile)), preferences)
}

function schedulePreferences(profile: string, preferences: WallpaperPreferences): void {
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

function cancelScheduledPreferences(profile: string): void {
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

function cancelDeferredWallpaperRefresh(): void {
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

async function decodeWallpaperAsset(asset: DesktopWallpaperAsset): Promise<HTMLImageElement> {
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

function cachedPalette(asset: DesktopWallpaperAsset, preferences: WallpaperPreferences): WallpaperPalette | null {
  return preferences.paletteSource === (asset.version || asset.url) ? preferences.palette : null
}

function manualPalette(asset: DesktopWallpaperAsset | null, preferences: WallpaperPreferences): WallpaperPalette {
  return (
    preferences.manualPalette ?? (asset ? cachedPalette(asset, preferences) : null) ?? DEFAULT_MANUAL_WALLPAPER_PALETTE
  )
}

async function preferencesWithPalette(
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
    let nextPreferences = preferences
    let paletteStatus: WallpaperState['paletteStatus'] = 'idle'

    if (asset) {
      await decodeWallpaperAsset(asset)

      if (preferences.adaptiveTheme) {
        if (preferences.paletteMode === 'manual') {
          if (preferences.manualPalette) {
            paletteStatus = 'ready'
          } else {
            nextPreferences = sanitizeWallpaperPreferences({
              ...preferences,
              manualPalette: manualPalette(asset, preferences)
            })
            cancelScheduledPreferences(normalizedProfile)
            persistPreferences(normalizedProfile, nextPreferences)
            paletteStatus = 'ready'
          }
        } else if (cachedPalette(asset, preferences)) {
          paletteStatus = 'ready'
        } else {
          try {
            const withPalette = await preferencesWithPalette(normalizedProfile, preferences, asset)

            if (withPalette) {
              nextPreferences = withPalette
              cancelScheduledPreferences(normalizedProfile)
              persistPreferences(normalizedProfile, withPalette)
              paletteStatus = 'ready'
            } else {
              paletteStatus = 'error'
            }
          } catch {
            paletteStatus = 'error'
          }
        }
      }
    }

    if (generation === requestGeneration && normalizedProfile === normalizeProfileKey($activeGatewayProfile.get())) {
      $wallpaper.set({ ...$wallpaper.get(), asset, paletteStatus, preferences: nextPreferences, status: 'ready' })
    }
  } catch {
    if (generation === requestGeneration && normalizedProfile === normalizeProfileKey($activeGatewayProfile.get())) {
      $wallpaper.set({ ...$wallpaper.get(), error: true, status: 'error' })
    }
  }
}

export async function ensureWallpaperLoaded(): Promise<void> {
  const state = $wallpaper.get()

  if (
    !state.supported ||
    state.status === 'loading' ||
    state.status === 'ready' ||
    state.status === 'removing' ||
    state.status === 'selecting'
  ) {
    return
  }

  cancelDeferredWallpaperRefresh()
  await refreshWallpaper(state.profile, { loadWhenDisabled: true })
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

async function analyzeAndApplyWallpaperPalette(
  profile: string,
  asset: DesktopWallpaperAsset,
  generation: number,
  preferences: WallpaperPreferences
): Promise<void> {
  try {
    const withPalette = await preferencesWithPalette(profile, preferences, asset)
    const current = $wallpaper.get()

    if (
      generation !== paletteRequestGeneration ||
      current.profile !== profile ||
      current.asset?.url !== asset.url ||
      !current.preferences.adaptiveTheme ||
      current.preferences.paletteMode !== 'auto'
    ) {
      return
    }

    if (!withPalette) {
      $wallpaper.set({ ...current, paletteStatus: 'error' })

      return
    }

    const nextPreferences = sanitizeWallpaperPreferences({
      ...current.preferences,
      palette: withPalette.palette,
      paletteSource: withPalette.paletteSource
    })

    cancelScheduledPreferences(profile)
    persistPreferences(profile, nextPreferences)
    $wallpaper.set({ ...current, paletteStatus: 'ready', preferences: nextPreferences })
  } catch {
    const current = $wallpaper.get()

    if (
      generation === paletteRequestGeneration &&
      current.profile === profile &&
      current.asset?.url === asset.url &&
      current.preferences.adaptiveTheme &&
      current.preferences.paletteMode === 'auto'
    ) {
      $wallpaper.set({ ...current, paletteStatus: 'error' })
    }
  }
}

export async function setWallpaperAdaptiveTheme(enabled: boolean): Promise<void> {
  const state = $wallpaper.get()
  const generation = ++paletteRequestGeneration
  let preferences = sanitizeWallpaperPreferences({ ...state.preferences, adaptiveTheme: enabled })

  if (enabled && state.asset && preferences.paletteMode === 'manual' && !preferences.manualPalette) {
    preferences = sanitizeWallpaperPreferences({
      ...preferences,
      manualPalette: manualPalette(state.asset, preferences)
    })
  }

  schedulePreferences(state.profile, preferences)
  $wallpaper.set({
    ...state,
    error: false,
    paletteStatus: enabled && state.asset ? (preferences.paletteMode === 'manual' ? 'ready' : 'loading') : 'idle',
    preferences
  })

  if (!enabled || !state.asset) {
    return
  }

  if (preferences.paletteMode === 'manual') {
    return
  }

  if (cachedPalette(state.asset, preferences)) {
    $wallpaper.set({ ...$wallpaper.get(), paletteStatus: 'ready' })

    return
  }

  await analyzeAndApplyWallpaperPalette(state.profile, state.asset, generation, preferences)
}

export async function setWallpaperPaletteMode(mode: WallpaperPaletteMode): Promise<void> {
  const state = $wallpaper.get()
  const generation = ++paletteRequestGeneration
  let preferences = sanitizeWallpaperPreferences({ ...state.preferences, paletteMode: mode })

  if (mode === 'manual' && state.asset && !preferences.manualPalette) {
    preferences = sanitizeWallpaperPreferences({
      ...preferences,
      manualPalette: manualPalette(state.asset, preferences)
    })
  }

  const shouldApply = Boolean(state.asset && preferences.adaptiveTheme)

  const paletteStatus: WallpaperState['paletteStatus'] = shouldApply
    ? mode === 'manual'
      ? 'ready'
      : 'loading'
    : 'idle'

  schedulePreferences(state.profile, preferences)
  $wallpaper.set({ ...state, error: false, paletteStatus, preferences })

  if (!shouldApply || !state.asset || mode === 'manual') {
    return
  }

  if (cachedPalette(state.asset, preferences)) {
    $wallpaper.set({ ...$wallpaper.get(), paletteStatus: 'ready' })

    return
  }

  await analyzeAndApplyWallpaperPalette(state.profile, state.asset, generation, preferences)
}

export function resetWallpaperPreferences(): void {
  const state = $wallpaper.get()

  const preferences = {
    ...DEFAULT_WALLPAPER_PREFERENCES,
    enabled: state.preferences.enabled,
    palette: state.preferences.palette,
    paletteSource: state.preferences.paletteSource
  }

  paletteRequestGeneration += 1
  cancelScheduledPreferences(state.profile)
  persistPreferences(state.profile, preferences)
  $wallpaper.set({ ...state, error: false, paletteStatus: 'idle', preferences })
}

export async function selectWallpaper(): Promise<void> {
  const state = $wallpaper.get()
  const bridge = window.hermesDesktop?.wallpaper

  if (!bridge) {
    $wallpaper.set({ ...state, error: true, status: 'error', supported: false })

    return
  }

  const profile = state.profile
  const priorStatus = state.status
  const generation = ++requestGeneration

  cancelDeferredWallpaperRefresh()
  $wallpaper.set({ ...state, error: false, status: 'selecting' })

  try {
    const result = await bridge.select(profile)

    if (result.canceled) {
      if (generation === requestGeneration && $wallpaper.get().profile === profile) {
        $wallpaper.set({ ...$wallpaper.get(), status: priorStatus })
      }

      return
    }

    if (!result.asset) {
      throw new Error('Wallpaper import returned no image.')
    }

    await decodeWallpaperAsset(result.asset)

    const current = $wallpaper.get()
    const priorPreferences = current.profile === profile ? current.preferences : readWallpaperPreferences(profile)

    let preferences = sanitizeWallpaperPreferences({
      ...priorPreferences,
      enabled: true,
      palette: null,
      paletteSource: ''
    })

    let paletteStatus: WallpaperState['paletteStatus'] = 'idle'

    if (preferences.adaptiveTheme) {
      if (preferences.paletteMode === 'manual') {
        preferences = sanitizeWallpaperPreferences({
          ...preferences,
          manualPalette: manualPalette(result.asset, preferences)
        })
        paletteStatus = 'ready'
      } else {
        try {
          preferences = (await preferencesWithPalette(profile, preferences, result.asset)) ?? preferences
          paletteStatus = preferences.palette ? 'ready' : 'error'
        } catch {
          paletteStatus = 'error'
        }
      }
    }

    if (generation !== requestGeneration || current.profile !== profile) {
      return
    }

    cancelScheduledPreferences(profile)
    persistPreferences(profile, preferences)

    $wallpaper.set({
      ...current,
      asset: result.asset,
      error: false,
      paletteStatus,
      preferences,
      status: 'ready'
    })
  } catch {
    if (generation === requestGeneration && $wallpaper.get().profile === profile) {
      $wallpaper.set({ ...$wallpaper.get(), error: true, status: 'error' })
    }
  }
}

export async function removeWallpaper(): Promise<void> {
  const state = $wallpaper.get()
  const bridge = window.hermesDesktop?.wallpaper

  if (!bridge) {
    $wallpaper.set({ ...state, error: true, status: 'error', supported: false })

    return
  }

  const profile = state.profile

  cancelDeferredWallpaperRefresh()
  requestGeneration += 1
  paletteRequestGeneration += 1
  cancelScheduledPreferences(profile)
  $wallpaper.set({ ...state, error: false, status: 'removing' })

  try {
    await bridge.remove(profile)
    writeJson(wallpaperStorageKey(profile), null)

    if ($wallpaper.get().profile === profile) {
      $wallpaper.set({
        ...$wallpaper.get(),
        asset: null,
        error: false,
        paletteStatus: 'idle',
        preferences: { ...DEFAULT_WALLPAPER_PREFERENCES },
        status: 'ready'
      })
    }
  } catch {
    if ($wallpaper.get().profile === profile) {
      persistPreferences(profile, state.preferences)
      $wallpaper.set({ ...$wallpaper.get(), error: true, status: 'error' })
    }
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
