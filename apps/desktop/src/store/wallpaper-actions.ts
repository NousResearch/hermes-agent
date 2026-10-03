import type { DesktopWallpaperAsset } from '@/global'
import { writeJson } from '@/lib/storage'
import {
  DEFAULT_WALLPAPER_PREFERENCES,
  sanitizeWallpaperPreferences,
  type WallpaperPaletteMode,
  type WallpaperPreferences,
  wallpaperStorageKey
} from '@/lib/wallpaper'

import {
  $wallpaper,
  beginWallpaperPaletteRequest,
  beginWallpaperRequest,
  cachedPalette,
  cancelDeferredWallpaperRefresh,
  cancelScheduledPreferences,
  decodeWallpaperAsset,
  invalidateWallpaperPaletteRequests,
  invalidateWallpaperRequests,
  manualPalette,
  persistPreferences,
  preferencesWithPalette,
  refreshWallpaper,
  schedulePreferences,
  wallpaperPaletteRequestIsCurrent,
  wallpaperRequestIsCurrent
} from './wallpaper'

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
      !wallpaperPaletteRequestIsCurrent(generation) ||
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
      wallpaperPaletteRequestIsCurrent(generation) &&
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
  const generation = beginWallpaperPaletteRequest()
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

  if (!enabled || !state.asset || preferences.paletteMode === 'manual') {
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
  const generation = beginWallpaperPaletteRequest()
  let preferences = sanitizeWallpaperPreferences({ ...state.preferences, paletteMode: mode })

  if (mode === 'manual' && state.asset && !preferences.manualPalette) {
    preferences = sanitizeWallpaperPreferences({
      ...preferences,
      manualPalette: manualPalette(state.asset, preferences)
    })
  }

  const shouldApply = Boolean(state.asset && preferences.adaptiveTheme)

  const paletteStatus = shouldApply ? (mode === 'manual' ? 'ready' : 'loading') : 'idle'

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

  invalidateWallpaperPaletteRequests()
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
  const generation = beginWallpaperRequest()

  $wallpaper.set({ ...state, error: false, status: 'selecting' })

  try {
    const result = await bridge.select(profile)

    if (!wallpaperRequestIsCurrent(generation, profile)) {
      return
    }

    if (result.canceled) {
      $wallpaper.set({ ...$wallpaper.get(), status: priorStatus })

      return
    }

    if (!result.asset) {
      throw new Error('Wallpaper import returned no image.')
    }

    await decodeWallpaperAsset(result.asset)

    if (!wallpaperRequestIsCurrent(generation, profile)) {
      return
    }

    // Import already sampled the image in main. Keep that cache even when
    // adaptive colors are off, so enabling them later needs no second read.
    let imported = sanitizeWallpaperPreferences({
      ...$wallpaper.get().preferences,
      enabled: true,
      palette: result.palette ?? null,
      paletteSource: result.asset.version || result.asset.url
    })

    if (imported.adaptiveTheme && imported.paletteMode === 'auto' && !imported.palette) {
      imported = (await preferencesWithPalette(profile, imported, result.asset).catch(() => null)) ?? imported

      if (!wallpaperRequestIsCurrent(generation, profile)) {
        return
      }
    }

    let preferences = sanitizeWallpaperPreferences({
      ...$wallpaper.get().preferences,
      enabled: true,
      palette: imported.palette,
      paletteSource: imported.palette ? imported.paletteSource : ''
    })

    let paletteStatus: 'error' | 'idle' | 'ready' = 'idle'

    if (preferences.adaptiveTheme) {
      if (preferences.paletteMode === 'manual') {
        preferences = sanitizeWallpaperPreferences({
          ...preferences,
          manualPalette: manualPalette(result.asset, preferences)
        })
        paletteStatus = 'ready'
      } else {
        paletteStatus = preferences.palette ? 'ready' : 'error'
      }
    }

    cancelScheduledPreferences(profile)
    persistPreferences(profile, preferences)

    $wallpaper.set({
      ...$wallpaper.get(),
      asset: result.asset,
      error: false,
      paletteStatus,
      preferences,
      status: 'ready'
    })
  } catch {
    if (wallpaperRequestIsCurrent(generation, profile)) {
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

  invalidateWallpaperRequests()
  invalidateWallpaperPaletteRequests()
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
