import { atom } from 'nanostores'

import { isMissingRpcMethod } from '@/lib/gateway-rpc'
import { normalize } from '@/lib/text'
import {
  $companionInfo,
  type CompanionInfo,
  type CompanionInfoMeta,
  companionProfile,
  hasCompanionSpriteForMeta,
  mergeCompanionInfoMeta,
  setCompanionInfo
} from '@/store/companion'

/**
 * Feature store for the companiondex gallery picker (Cmd+K "Companions…" + Settings).
 *
 * Why this exists: `companion.gallery` does a *network* manifest fetch on the gateway,
 * so re-pulling it after every adopt/toggle made the picker feel laggy and made
 * two components (palette + settings) each carry their own copy of the same
 * fetch / thumb-cache / optimistic-mutation logic. This store centralizes it:
 *
 *  - The gallery is fetched once and cached; reopening the picker is instant.
 *  - Mutations (adopt / enable / remove) patch local state and only re-pull the
 *    cheap, local `companion.info` — never the network manifest again.
 *  - Thumbnails are deduped in a process-global cache (the backend disk-caches
 *    too, so a slug is fetched at most once per session).
 *
 * Consumers just `useStore($companionGallery)` and call the actions; no component
 * owns gallery state anymore.
 */

export interface GalleryCompanion {
  slug: string
  displayName: string
  installed: boolean
  spritesheetUrl?: string
  /** companiondex's hand-picked set — used only to rank "popular" companions first. */
  curated?: boolean
  /** Hatched locally by the user (createdBy=generator) — badged + ranked first. */
  generated?: boolean
}

export interface CompanionGallery {
  enabled: boolean
  active: string
  companions: GalleryCompanion[]
}

export type CompanionGalleryStatus = 'idle' | 'loading' | 'ready' | 'stale' | 'error'

/** The recovering `requestGateway` from `useGatewayRequest` — passed in so the
 *  store reuses the hook's reconnect/reauth handling instead of duplicating it. */
export type GatewayRequest = <T>(
  method: string,
  params?: Record<string, unknown>,
  timeoutMs?: number,
  signal?: AbortSignal
) => Promise<T>

/** Profile-scoped companion RPC. Companions are per-profile, so every call carries the active
 *  profile (the gateway no-ops it for the launch profile). One chokepoint so no
 *  call site can forget it. */
const companionRpc = <T>(request: GatewayRequest, method: string, params: Record<string, unknown> = {}): Promise<T> =>
  request<T>(method, { ...params, profile: companionProfile() })

/** A JSON-RPC "method not found" — the backend predates the companion RPCs. */
export const $companionGallery = atom<CompanionGallery | null>(null)
export const $companionGalleryStatus = atom<CompanionGalleryStatus>('idle')
export const $companionGalleryError = atom<string | null>(null)

// Which action is in flight, so rows/buttons can show a spinner. A slug for a
// per-companion mutation; the `TOGGLE_*` sentinels for the on/off switch.
export const TOGGLE_ON = '\u0000on'
export const TOGGLE_OFF = '\u0000off'
export const $companionBusy = atom<string | null>(null)

// Process-global caches (survive component unmount → instant reopen).
const thumbCache = new Map<string, Promise<string | null>>()
let galleryLoad: Promise<void> | null = null

/**
 * Drop the cached gallery, thumbnails, and in-flight load so the next open
 * refetches against the now-active profile's backend. Called on a profile switch
 * (companions are per-profile) — the floating companion's own `companion.info` poll repaints the
 * new profile's mascot, and the picker reloads its gallery on next mount.
 */
export function resetCompanionGallery(): void {
  galleryLoad = null
  thumbCache.clear()
  $companionGallery.set(null)
  $companionGalleryStatus.set('idle')
  $companionGalleryError.set(null)
  $companionBusy.set(null)
}

export function loadCompanionThumb(request: GatewayRequest, slug: string, url?: string): Promise<string | null> {
  let pending = thumbCache.get(slug)

  if (!pending) {
    pending = companionRpc<{ ok: boolean; dataUri?: string }>(request, 'companion.thumb', { slug, url: url ?? '' })
      .then(result => (result?.ok && result.dataUri ? result.dataUri : null))
      .catch(() => null)
    thumbCache.set(slug, pending)
  }

  return pending
}

/**
 * Fetch the gallery once and cache it. Subsequent calls are no-ops while a
 * ready snapshot is held; pass `{ force: true }` to bypass the cache (e.g. a
 * manual refresh). Concurrent callers share a single in-flight request.
 */
export function loadCompanionGallery(request: GatewayRequest, options: { force?: boolean } = {}): Promise<void> {
  if (!options.force && $companionGallery.get() && $companionGalleryStatus.get() === 'ready') {
    return Promise.resolve()
  }

  if (galleryLoad) {
    return galleryLoad
  }

  galleryLoad = (async () => {
    if (!$companionGallery.get()) {
      $companionGalleryStatus.set('loading')
    }

    let localOk = false

    try {
      // Phase 1: local companions only — instant, never blocks on the remote companiondex
      // manifest. The user's own/generated companions render right away.
      const [local] = await Promise.all([
        companionRpc<CompanionGallery>(request, 'companion.gallery', { localOnly: true }),
        syncInfo(request)
      ])

      if (local) {
        $companionGallery.set(local)
        $companionGalleryStatus.set('ready')
        $companionGalleryError.set(null)
        localOk = true
      }
    } catch (e) {
      if (isMissingRpcMethod(e)) {
        $companionGalleryStatus.set('stale')
      } else if (!$companionGallery.get()) {
        // Only surface a hard error when we have nothing to show; a transient
        // hiccup mid-session leaves the cached gallery intact.
        $companionGalleryStatus.set('error')
        $companionGalleryError.set(e instanceof Error ? e.message : 'Could not reach the companiondex gallery.')
      }
    } finally {
      galleryLoad = null
    }

    // Phase 2: merge in the full companiondex catalog in the background. A slow/failed
    // manifest fetch never hides the local companions shown in phase 1.
    if (localOk) {
      try {
        const full = await companionRpc<CompanionGallery>(request, 'companion.gallery')

        if (full) {
          $companionGallery.set(full)
          $companionGalleryStatus.set('ready')
        }
      } catch {
        // Keep the local-only gallery; the companiondex catalog just stays unmerged.
      }
    }
  })()

  return galleryLoad
}

// Push the live mascot state (cheap, local config read) without re-pulling the
// network gallery — the floating companion repaints, the picker keeps its cache.
async function syncInfo(request: GatewayRequest): Promise<void> {
  try {
    let meta: CompanionInfoMeta | null = null

    try {
      meta = await companionRpc<CompanionInfoMeta>(request, 'companion.info.meta')
    } catch (e) {
      if (!isMissingRpcMethod(e)) {
        throw e
      }

      const info = await companionRpc<CompanionInfo>(request, 'companion.info')

      if (info) {
        setCompanionInfo(info)
      }

      return
    }

    if (!meta) {
      return
    }

    if (!meta.enabled) {
      setCompanionInfo({ enabled: false })

      return
    }

    const current = $companionInfo.get()

    if (hasCompanionSpriteForMeta(current, meta)) {
      const merged = mergeCompanionInfoMeta(current, meta)

      if (merged !== current) {
        setCompanionInfo(merged)
      }

      return
    }

    const info = await companionRpc<CompanionInfo>(request, 'companion.info')

    if (info) {
      setCompanionInfo(info)
    }
  } catch {
    // The mutation already succeeded; a stale mascot self-heals on its poll.
  }
}

/**
 * Reflect a just-adopted *local* companion without any network: optimistically mark it
 * active/installed in the cached gallery and repaint the live mascot via the
 * local `companion.info`. Adopting a generated companion is a disk+config op — it must never
 * wait on `companion.gallery`'s remote companiondex manifest fetch.
 */
export async function applyAdoptedCompanion(request: GatewayRequest, slug: string, displayName: string): Promise<void> {
  patchGallery(gallery => ({
    ...gallery,
    enabled: true,
    active: slug,
    companions: gallery.companions.some(p => p.slug === slug)
      ? gallery.companions.map(p => (p.slug === slug ? { ...p, installed: true, displayName } : p))
      : [...gallery.companions, { slug, displayName, installed: true, spritesheetUrl: '' }]
  }))
  await syncInfo(request)
}

/**
 * Filter (drop the internal `clawd*` companions + apply a search query) and rank the
 * gallery for a picker. Ranking has no popularity data, so it leans on the
 * signals we do have: active companion first, then installed, then curated. Shared by
 * the Cmd-K palette and the Settings grid so the two can't drift — each caller
 * applies its own cap and reads `.length` for the total.
 */
export function rankedGalleryCompanions(gallery: CompanionGallery | null, query = ''): GalleryCompanion[] {
  if (!gallery) {
    return []
  }

  const needle = normalize(query)

  // User-generated companions first, then the active companion, then installed, then curated.
  // Guard every term with a boolean — local-only companions omit curated/generated, and
  // `Number(undefined)` is NaN, which poisons the sort (it would sink those companions
  // below the render cap and hide them entirely).
  const rank = (p: GalleryCompanion) =>
    (p.generated ? 8 : 0) +
    (gallery.enabled && p.slug === gallery.active ? 4 : 0) +
    (p.installed ? 2 : 0) +
    (p.curated ? 1 : 0)

  return gallery.companions
    .filter(
      p =>
        !/^clawd(-|$)/i.test(p.slug) &&
        (!needle || p.slug.toLowerCase().includes(needle) || p.displayName.toLowerCase().includes(needle))
    )
    .sort((a, b) => rank(b) - rank(a))
}

function patchGallery(fn: (gallery: CompanionGallery) => CompanionGallery): void {
  const current = $companionGallery.get()

  if (current) {
    $companionGallery.set(fn(current))
  }
}

/** Shared mutation wrapper: spin, fire, patch on success, surface failures. */
async function mutate(
  busyKey: string,
  fallback: string,
  request: GatewayRequest,
  run: () => Promise<void>
): Promise<boolean> {
  $companionBusy.set(busyKey)
  $companionGalleryError.set(null)

  try {
    await run()
    await syncInfo(request)

    return true
  } catch (e) {
    if (isMissingRpcMethod(e)) {
      $companionGalleryStatus.set('stale')
    } else {
      $companionGalleryError.set(e instanceof Error ? e.message : fallback)
    }

    return false
  } finally {
    $companionBusy.set(null)
  }
}

/** Install (if needed) + activate a companion. Optimistically marks it active. */
export function adoptCompanion(request: GatewayRequest, slug: string, fallback: string): Promise<boolean> {
  return mutate(slug, fallback, request, async () => {
    await companionRpc(request, 'companion.select', { slug })
    patchGallery(g => ({
      ...g,
      enabled: true,
      active: slug,
      companions: g.companions.map(p => (p.slug === slug ? { ...p, installed: true } : p))
    }))
  })
}

/**
 * Turn the floating mascot on/off. On enable, activates the current companion (or the
 * first installed one). Returns false without firing if there's nothing to show.
 */
export function setCompanionEnabled(
  request: GatewayRequest,
  on: boolean,
  copy: { noneAvailable: string; fallback: string }
): Promise<boolean> {
  const gallery = $companionGallery.get()

  if (!on && !(gallery?.enabled ?? false)) {
    return Promise.resolve(true)
  }

  let slug = gallery?.active || ''

  if (on) {
    slug = slug || gallery?.companions.find(p => p.installed)?.slug || ''

    if (!slug) {
      $companionGalleryError.set(copy.noneAvailable)

      return Promise.resolve(false)
    }
  }

  return mutate(on ? TOGGLE_ON : TOGGLE_OFF, copy.fallback, request, async () => {
    if (on) {
      await companionRpc(request, 'companion.select', { slug })
    } else {
      await companionRpc(request, 'companion.disable')
    }

    patchGallery(g => ({ ...g, enabled: on, active: on ? slug : g.active }))
  })
}

// Companion scale bounds — mirror `agent/companion/constants.py` (MIN_SCALE / MAX_SCALE) so
// the slider and the server clamp to the same range.
export const COMPANION_SCALE_MIN = 0.1
export const COMPANION_SCALE_MAX = 3.0
export const COMPANION_SCALE_DEFAULT = 0.33
export const clampCompanionScale = (n: number) => Math.max(COMPANION_SCALE_MIN, Math.min(COMPANION_SCALE_MAX, n))

// Wheel → scale. Multiplicative so one notch feels the same at any size. Tuned
// for a discrete mouse-wheel notch (deltaY ≈ ±100); trackpad two-finger scroll
// (smaller deltas) just resizes more gently, which is fine.
const WHEEL_SCALE_K = 0.0015

/**
 * Next companion scale for one mouse-wheel step over the companion. Scrolling up (deltaY < 0)
 * grows it, scrolling down shrinks it; the result is clamped to the slider's range.
 */
export function nextScaleFromWheel(current: number | undefined, deltaY: number): number {
  const base = current ?? COMPANION_SCALE_DEFAULT

  return clampCompanionScale(base * Math.exp(-deltaY * WHEEL_SCALE_K))
}

let scalePersist: ReturnType<typeof setTimeout> | undefined

/**
 * Resize the floating companion. Updates `$companionInfo` synchronously so the on-screen companion
 * (and the slider) react on the same frame, then debounce-persists to
 * `display.companion.scale` so a slider drag fires one RPC, not one per pixel. No poll
 * or event needed — the companion already renders from `$companionInfo.scale`.
 */
export function setCompanionScale(request: GatewayRequest, scale: number): void {
  const next = clampCompanionScale(scale)

  setCompanionInfo({ ...$companionInfo.get(), scale: next })

  clearTimeout(scalePersist)
  scalePersist = setTimeout(() => {
    companionRpc<{ ok: boolean; scale?: number }>(request, 'companion.scale', { scale: next })
      .then(result => {
        // Reconcile with the server's clamp (cheap; only matters at the bounds).
        if (typeof result?.scale === 'number' && result.scale !== $companionInfo.get().scale) {
          setCompanionInfo({ ...$companionInfo.get(), scale: result.scale })
        }
      })
      .catch(() => {
        // Cosmetic — the companion already resized; persistence self-heals next write.
      })
  }, 200)
}

/** Export a companion as a `.zip` (companion.json + spritesheet) and save it via the browser. */
export async function exportCompanion(request: GatewayRequest, slug: string, fallback: string): Promise<boolean> {
  $companionBusy.set(slug)
  $companionGalleryError.set(null)

  try {
    const res = await companionRpc<{ ok: boolean; filename: string; zipBase64: string }>(request, 'companion.export', { slug })

    if (!res?.ok || !res.zipBase64) {
      throw new Error(fallback)
    }

    const bytes = Uint8Array.from(atob(res.zipBase64), c => c.charCodeAt(0))
    const url = URL.createObjectURL(new Blob([bytes], { type: 'application/zip' }))
    const anchor = document.createElement('a')
    anchor.href = url
    anchor.download = res.filename || `${slug}.zip`
    anchor.click()
    URL.revokeObjectURL(url)

    return true
  } catch (e) {
    $companionGalleryError.set(e instanceof Error ? e.message : fallback)

    return false
  } finally {
    $companionBusy.set(null)
  }
}

/**
 * Rename a companion — optimistic. The new name shows instantly (so the dialog can
 * close immediately); the RPC runs in the background and the backend also
 * realigns the slug/dir, so we reconcile the slug + thumb cache when it returns,
 * and roll the name back if it fails.
 */
export function renameCompanion(request: GatewayRequest, slug: string, name: string, fallback: string): Promise<boolean> {
  const trimmed = name.trim()

  if (!trimmed) {
    return Promise.resolve(false)
  }

  const prev = $companionGallery.get()?.companions.find(p => p.slug === slug)?.displayName ?? ''

  // Optimistic: paint the new name now (slug reconciles when the RPC returns).
  patchGallery(g => ({
    ...g,
    companions: g.companions.map(p => (p.slug === slug ? { ...p, displayName: trimmed } : p))
  }))
  $companionGalleryError.set(null)

  return (async () => {
    try {
      const res = await companionRpc<{ ok: boolean; slug: string; displayName: string }>(request, 'companion.rename', {
        slug,
        name: trimmed
      })

      if (!res?.ok) {
        throw new Error(fallback)
      }

      const newSlug = res.slug || slug

      if (newSlug !== slug) {
        thumbCache.delete(slug)
        patchGallery(g => ({
          ...g,
          active: g.active === slug ? newSlug : g.active,
          companions: g.companions
            .filter(p => p.slug !== newSlug || p.slug === slug)
            .map(p => (p.slug === slug ? { ...p, slug: newSlug, displayName: res.displayName || trimmed } : p))
        }))
      }

      return true
    } catch (e) {
      // Roll the optimistic name back so the list reflects on-disk truth.
      patchGallery(g => ({
        ...g,
        companions: g.companions.map(p => (p.slug === slug ? { ...p, displayName: prev } : p))
      }))
      $companionGalleryError.set(e instanceof Error ? e.message : fallback)

      return false
    }
  })()
}

/** Uninstall a companion; turns the mascot off if it was the active one. */
export function removeCompanion(request: GatewayRequest, slug: string, fallback: string): Promise<boolean> {
  return mutate(slug, fallback, request, async () => {
    await companionRpc(request, 'companion.remove', { slug })
    // Evict the by-slug thumb cache so a reused slug doesn't render this companion's
    // stale thumbnail (the backend drops its disk thumb in parallel).
    thumbCache.delete(slug)
    patchGallery(g => ({
      ...g,
      enabled: g.active === slug ? false : g.enabled,
      active: g.active === slug ? '' : g.active,
      // Companiondex companions can be reinstalled from the manifest, so we just mark them
      // uninstalled. Generated / local-only companions have no remote source — once
      // deleted they're gone, so drop them from the list entirely.
      companions: g.companions.flatMap(p => {
        if (p.slug !== slug) {
          return [p]
        }

        return p.generated || !p.spritesheetUrl ? [] : [{ ...p, installed: false }]
      })
    }))
  })
}
