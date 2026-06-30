import { atom } from 'nanostores'

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
}

export interface PetGallery {
  enabled: boolean
  active: string
  pets: GalleryPet[]
}

export type PetGalleryStatus = 'idle' | 'loading' | 'ready' | 'stale' | 'error'

/** The recovering `requestGateway` from `useGatewayRequest` — passed in so the
 *  store reuses the hook's reconnect/reauth handling instead of duplicating it. */
export type GatewayRequest = <T>(method: string, params?: Record<string, unknown>) => Promise<T>

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

    try {
      const [next, info] = await Promise.all([
        petRpc<PetGallery>(request, 'pet.gallery'),
        petRpc<PetInfo>(request, 'pet.info')
      ])

      if (next) {
        $petGallery.set(next)
        $petGalleryStatus.set('ready')
        $petGalleryError.set(null)
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

  const needle = query.trim().toLowerCase()

  const rank = (p: GalleryPet) =>
    Number(gallery.enabled && p.slug === gallery.active) * 4 + Number(p.installed) * 2 + Number(p.curated)

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

/** Uninstall a pet; turns the mascot off if it was the active one. */
export function removePet(request: GatewayRequest, slug: string, fallback: string): Promise<boolean> {
  return mutate(slug, fallback, request, async () => {
    await petRpc(request, 'pet.remove', { slug })
    patchGallery(g => ({
      ...g,
      enabled: g.active === slug ? false : g.enabled,
      pets: g.pets.map(p => (p.slug === slug ? { ...p, installed: false } : p))
    }))
  })
}
