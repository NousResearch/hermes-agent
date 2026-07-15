import { atom } from 'nanostores'

import { normalize } from '@/lib/text'
import { $petInfo, type PetInfo, petProfile, setPetInfo } from '@/store/pet'

/**
 * Feature store for the petdex gallery picker (Cmd+K "Pets…" + Settings).
 *
 * Why this exists: `pet.gallery` does a *network* manifest fetch on the gateway,
 * so re-pulling it after every adopt/toggle made the picker feel laggy and made
 * two components (palette + settings) each carry their own copy of the same
 * fetch / thumb-cache / optimistic-mutation logic. This store centralizes it:
 *
 *  - The gallery is fetched once and cached; reopening the picker is instant.
 *  - Mutations (adopt / enable / remove) patch local state and only re-pull the
 *    cheap, local `pet.info` — never the network manifest again.
 *  - Thumbnails are deduped in a process-global cache (the backend disk-caches
 *    too, so a slug is fetched at most once per session).
 *
 * Consumers just `useStore($petGallery)` and call the actions; no component
 * owns gallery state anymore.
 */

export interface GalleryPet {
  slug: string
  displayName: string
  installed: boolean
  spritesheetUrl?: string
  /** petdex's hand-picked set — used only to rank "popular" pets first. */
  curated?: boolean
  /** Hatched locally by the user (createdBy=generator) — badged + ranked first. */
  generated?: boolean
}

export interface PetGallery {
  enabled: boolean
  active: string
  pets: GalleryPet[]
}

export type PetGalleryStatus = 'idle' | 'loading' | 'ready' | 'stale' | 'error'

/** The recovering `requestGateway` from `useGatewayRequest` — passed in so the
 *  store reuses the hook's reconnect/reauth handling instead of duplicating it. */
export type GatewayRequest = <T>(
  method: string,
  params?: Record<string, unknown>,
  timeoutMs?: number,
  signal?: AbortSignal
) => Promise<T>

/** Profile-scoped pet RPC. Pets are per-profile, so every call carries the active
 *  profile (the gateway no-ops it for the launch profile). One chokepoint so no
 *  call site can forget it. */
const petRpc = <T>(request: GatewayRequest, method: string, params: Record<string, unknown> = {}): Promise<T> =>
  request<T>(method, { ...params, profile: petProfile() })

/** A JSON-RPC "method not found" — the backend predates the pet RPCs. */
function isMissingMethod(error: unknown): boolean {
  const message = error instanceof Error ? error.message : String(error)

  return /method not found|-32601|unknown method|no such method/i.test(message)
}

export const $petGallery = atom<PetGallery | null>(null)
export const $petGalleryStatus = atom<PetGalleryStatus>('idle')
export const $petGalleryError = atom<string | null>(null)

// Which action is in flight, so rows/buttons can show a spinner. A slug for a
// per-pet mutation; the `TOGGLE_*` sentinels for the on/off switch.
export const TOGGLE_ON = '\u0000on'
export const TOGGLE_OFF = '\u0000off'
export const $petBusy = atom<string | null>(null)

// Process-global caches (survive component unmount → instant reopen).
const thumbCache = new Map<string, Promise<string | null>>()
let galleryLoad: Promise<void> | null = null

/**
 * Drop the cached gallery, thumbnails, and in-flight load so the next open
 * refetches against the now-active profile's backend. Called on a profile switch
 * (pets are per-profile) — the floating pet's own `pet.info` poll repaints the
 * new profile's mascot, and the picker reloads its gallery on next mount.
 */
export function resetPetGallery(): void {
  galleryLoad = null
  thumbCache.clear()
  $petGallery.set(null)
  $petGalleryStatus.set('idle')
  $petGalleryError.set(null)
  $petBusy.set(null)
}

export function loadPetThumb(request: GatewayRequest, slug: string, url?: string): Promise<string | null> {
  let pending = thumbCache.get(slug)

  if (!pending) {
    pending = petRpc<{ ok: boolean; dataUri?: string }>(request, 'pet.thumb', { slug, url: url ?? '' })
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
export function loadPetGallery(request: GatewayRequest, options: { force?: boolean } = {}): Promise<void> {
  if (!options.force && $petGallery.get() && $petGalleryStatus.get() === 'ready') {
    return Promise.resolve()
  }

  if (galleryLoad) {
    return galleryLoad
  }

  galleryLoad = (async () => {
    if (!$petGallery.get()) {
      $petGalleryStatus.set('loading')
    }

    let localOk = false

    try {
      // Phase 1: local pets only — instant, never blocks on the remote petdex
      // manifest. The user's own/generated pets render right away.
      const [local, info] = await Promise.all([
        petRpc<PetGallery>(request, 'pet.gallery', { localOnly: true }),
        petRpc<PetInfo>(request, 'pet.info')
      ])

      if (local) {
        $petGallery.set(local)
        $petGalleryStatus.set('ready')
        $petGalleryError.set(null)
        localOk = true
      }

      if (info) {
        setPetInfo(info)
      }
    } catch (e) {
      if (isMissingMethod(e)) {
        $petGalleryStatus.set('stale')
      } else if (!$petGallery.get()) {
        // Only surface a hard error when we have nothing to show; a transient
        // hiccup mid-session leaves the cached gallery intact.
        $petGalleryStatus.set('error')
        $petGalleryError.set(e instanceof Error ? e.message : 'Could not reach the petdex gallery.')
      }
    } finally {
      galleryLoad = null
    }

    // Phase 2: merge in the full petdex catalog in the background. A slow/failed
    // manifest fetch never hides the local pets shown in phase 1.
    if (localOk) {
      try {
        const full = await petRpc<PetGallery>(request, 'pet.gallery')

        if (full) {
          $petGallery.set(full)
          $petGalleryStatus.set('ready')
        }
      } catch {
        // Keep the local-only gallery; the petdex catalog just stays unmerged.
      }
    }
  })()

  return galleryLoad
}

// Push the live mascot state (cheap, local config read) without re-pulling the
// network gallery — the floating pet repaints, the picker keeps its cache.
async function syncInfo(request: GatewayRequest): Promise<void> {
  try {
    const info = await petRpc<PetInfo>(request, 'pet.info')

    if (info) {
      setPetInfo(info)
    }
  } catch {
    // The mutation already succeeded; a stale mascot self-heals on its poll.
  }
}

/**
 * Reflect a just-adopted *local* pet without any network: optimistically mark it
 * active/installed in the cached gallery and repaint the live mascot via the
 * local `pet.info`. Adopting a generated pet is a disk+config op — it must never
 * wait on `pet.gallery`'s remote petdex manifest fetch.
 */
export async function applyAdoptedPet(request: GatewayRequest, slug: string, displayName: string): Promise<void> {
  patchGallery(gallery => ({
    ...gallery,
    enabled: true,
    active: slug,
    pets: gallery.pets.some(p => p.slug === slug)
      ? gallery.pets.map(p => (p.slug === slug ? { ...p, installed: true, displayName } : p))
      : [...gallery.pets, { slug, displayName, installed: true, spritesheetUrl: '' }]
  }))
  await syncInfo(request)
}

/**
 * Filter (drop the internal `clawd*` pets + apply a search query) and rank the
 * gallery for a picker. Ranking has no popularity data, so it leans on the
 * signals we do have: active pet first, then installed, then curated. Shared by
 * the Cmd-K palette and the Settings grid so the two can't drift — each caller
 * applies its own cap and reads `.length` for the total.
 */
export function rankedGalleryPets(gallery: PetGallery | null, query = ''): GalleryPet[] {
  if (!gallery) {
    return []
  }

  const needle = normalize(query)

  // User-generated pets first, then the active pet, then installed, then curated.
  // Guard every term with a boolean — local-only pets omit curated/generated, and
  // `Number(undefined)` is NaN, which poisons the sort (it would sink those pets
  // below the render cap and hide them entirely).
  const rank = (p: GalleryPet) =>
    (p.generated ? 8 : 0) +
    (gallery.enabled && p.slug === gallery.active ? 4 : 0) +
    (p.installed ? 2 : 0) +
    (p.curated ? 1 : 0)

  return gallery.pets
    .filter(
      p =>
        !/^clawd(-|$)/i.test(p.slug) &&
        (!needle || p.slug.toLowerCase().includes(needle) || p.displayName.toLowerCase().includes(needle))
    )
    .sort((a, b) => rank(b) - rank(a))
}

function patchGallery(fn: (gallery: PetGallery) => PetGallery): void {
  const current = $petGallery.get()

  if (current) {
    $petGallery.set(fn(current))
  }
}

/** Shared mutation wrapper: spin, fire, patch on success, surface failures. */
async function mutate(
  busyKey: string,
  fallback: string,
  request: GatewayRequest,
  run: () => Promise<void>
): Promise<boolean> {
  $petBusy.set(busyKey)
  $petGalleryError.set(null)

  try {
    await run()
    await syncInfo(request)

    return true
  } catch (e) {
    if (isMissingMethod(e)) {
      $petGalleryStatus.set('stale')
    } else {
      $petGalleryError.set(e instanceof Error ? e.message : fallback)
    }

    return false
  } finally {
    $petBusy.set(null)
  }
}

/** Install (if needed) + activate a pet. Optimistically marks it active. */
export function adoptPet(request: GatewayRequest, slug: string, fallback: string): Promise<boolean> {
  return mutate(slug, fallback, request, async () => {
    await petRpc(request, 'pet.select', { slug })
    patchGallery(g => ({
      ...g,
      enabled: true,
      active: slug,
      pets: g.pets.map(p => (p.slug === slug ? { ...p, installed: true } : p))
    }))
  })
}

/**
 * Turn the floating mascot on/off. On enable, activates the current pet (or the
 * first installed one). Returns false without firing if there's nothing to show.
 */
export function setPetEnabled(
  request: GatewayRequest,
  on: boolean,
  copy: { noneAvailable: string; fallback: string }
): Promise<boolean> {
  const gallery = $petGallery.get()

  if (!on && !(gallery?.enabled ?? false)) {
    return Promise.resolve(true)
  }

  let slug = gallery?.active || ''

  if (on) {
    slug = slug || gallery?.pets.find(p => p.installed)?.slug || ''

    if (!slug) {
      $petGalleryError.set(copy.noneAvailable)

      return Promise.resolve(false)
    }
  }

  return mutate(on ? TOGGLE_ON : TOGGLE_OFF, copy.fallback, request, async () => {
    if (on) {
      await petRpc(request, 'pet.select', { slug })
    } else {
      await petRpc(request, 'pet.disable')
    }

    patchGallery(g => ({ ...g, enabled: on, active: on ? slug : g.active }))
  })
}

// Pet scale bounds — mirror `agent/pet/constants.py` (MIN_SCALE / MAX_SCALE) so
// the slider and the server clamp to the same range.
export const PET_SCALE_MIN = 0.1
export const PET_SCALE_MAX = 3.0
export const PET_SCALE_DEFAULT = 0.33
export const clampPetScale = (n: number) => Math.max(PET_SCALE_MIN, Math.min(PET_SCALE_MAX, n))

// Wheel → scale. Multiplicative so one notch feels the same at any size. Tuned
// for a discrete mouse-wheel notch (deltaY ≈ ±100); trackpad two-finger scroll
// (smaller deltas) just resizes more gently, which is fine.
const WHEEL_SCALE_K = 0.0015

/**
 * Next pet scale for one mouse-wheel step over the pet. Scrolling up (deltaY < 0)
 * grows it, scrolling down shrinks it; the result is clamped to the slider's range.
 */
export function nextScaleFromWheel(current: number | undefined, deltaY: number): number {
  const base = current ?? PET_SCALE_DEFAULT

  return clampPetScale(base * Math.exp(-deltaY * WHEEL_SCALE_K))
}

let scalePersist: ReturnType<typeof setTimeout> | undefined

/**
 * Resize the floating pet. Updates `$petInfo` synchronously so the on-screen pet
 * (and the slider) react on the same frame, then debounce-persists to
 * `display.pet.scale` so a slider drag fires one RPC, not one per pixel. No poll
 * or event needed — the pet already renders from `$petInfo.scale`.
 */
export function setPetScale(request: GatewayRequest, scale: number): void {
  const next = clampPetScale(scale)

  setPetInfo({ ...$petInfo.get(), scale: next })

  clearTimeout(scalePersist)
  scalePersist = setTimeout(() => {
    petRpc<{ ok: boolean; scale?: number }>(request, 'pet.scale', { scale: next })
      .then(result => {
        // Reconcile with the server's clamp (cheap; only matters at the bounds).
        if (typeof result?.scale === 'number' && result.scale !== $petInfo.get().scale) {
          setPetInfo({ ...$petInfo.get(), scale: result.scale })
        }
      })
      .catch(() => {
        // Cosmetic — the pet already resized; persistence self-heals next write.
      })
  }, 200)
}

/** Export a pet as a `.zip` (pet.json + spritesheet) and save it via the browser. */
export async function exportPet(request: GatewayRequest, slug: string, fallback: string): Promise<boolean> {
  $petBusy.set(slug)
  $petGalleryError.set(null)

  try {
    const res = await petRpc<{ ok: boolean; filename: string; zipBase64: string }>(request, 'pet.export', { slug })

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
    $petGalleryError.set(e instanceof Error ? e.message : fallback)

    return false
  } finally {
    $petBusy.set(null)
  }
}

/**
 * Rename a pet — optimistic. The new name shows instantly (so the dialog can
 * close immediately); the RPC runs in the background and the backend also
 * realigns the slug/dir, so we reconcile the slug + thumb cache when it returns,
 * and roll the name back if it fails.
 */
export function renamePet(request: GatewayRequest, slug: string, name: string, fallback: string): Promise<boolean> {
  const trimmed = name.trim()

  if (!trimmed) {
    return Promise.resolve(false)
  }

  const prev = $petGallery.get()?.pets.find(p => p.slug === slug)?.displayName ?? ''

  // Optimistic: paint the new name now (slug reconciles when the RPC returns).
  patchGallery(g => ({
    ...g,
    pets: g.pets.map(p => (p.slug === slug ? { ...p, displayName: trimmed } : p))
  }))
  $petGalleryError.set(null)

  return (async () => {
    try {
      const res = await petRpc<{ ok: boolean; slug: string; displayName: string }>(request, 'pet.rename', {
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
          pets: g.pets
            .filter(p => p.slug !== newSlug || p.slug === slug)
            .map(p => (p.slug === slug ? { ...p, slug: newSlug, displayName: res.displayName || trimmed } : p))
        }))
      }

      return true
    } catch (e) {
      // Roll the optimistic name back so the list reflects on-disk truth.
      patchGallery(g => ({
        ...g,
        pets: g.pets.map(p => (p.slug === slug ? { ...p, displayName: prev } : p))
      }))
      $petGalleryError.set(e instanceof Error ? e.message : fallback)

      return false
    }
  })()
}

/** Uninstall a pet; turns the mascot off if it was the active one. */
export function removePet(request: GatewayRequest, slug: string, fallback: string): Promise<boolean> {
  return mutate(slug, fallback, request, async () => {
    await petRpc(request, 'pet.remove', { slug })
    // Evict the by-slug thumb cache so a reused slug doesn't render this pet's
    // stale thumbnail (the backend drops its disk thumb in parallel).
    thumbCache.delete(slug)
    patchGallery(g => ({
      ...g,
      enabled: g.active === slug ? false : g.enabled,
      active: g.active === slug ? '' : g.active,
      // Petdex pets can be reinstalled from the manifest, so we just mark them
      // uninstalled. Generated / local-only pets have no remote source — once
      // deleted they're gone, so drop them from the list entirely.
      pets: g.pets.flatMap(p => {
        if (p.slug !== slug) {
          return [p]
        }

        return p.generated || !p.spritesheetUrl ? [] : [{ ...p, installed: false }]
      })
    }))
  })
}
