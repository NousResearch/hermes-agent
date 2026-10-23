/**
 * The Pet tab: browse the petdex gallery and use a companion's first sprite
 * frame as the bot's profile picture.
 */

import { Button, cn, GlyphSpinner, host, Input, LruCache, RowButton, useQuery } from '@hermes/plugin-sdk'
import { useEffect, useState } from 'react'

import { useBots } from './i18n'
import { ID } from './shared'

// ── pet tab: attach a petdex companion that lives beside the avatar ─────────

// A petdex "spritesheet" is the FULL animation sheet (1536×1872 webp, ~2MB;
// 8×9 grid of 192×208 frames). Cropping frame 0 client-side meant downloading
// megabytes per tile, a UA-less CDN fetch the petdex CDN rejects with 403
// (#90465), and it could never serve pets hatched locally — those are absent
// from the petdex manifest, so `pet.gallery` reports an EMPTY spritesheetUrl.
// The gateway's `pet.thumb` crops + downsamples frame 0 server-side (installed
// sheet off disk, else the host-validated CDN URL) and returns a small PNG data
// URI: one path for every pet, same-origin through the authenticated gateway.
//
// The gallery is 4500+ pets browsed 24 at a time, and each entry is a decoded
// PNG data URL — an unbounded cache holds every pet the user ever scrolled
// past for the life of the window. Five pages' worth keeps scrolling back up
// instant; past that a revisit pays the RPC again.
const PET_THUMB_CACHE_MAX = 120
// A hung gateway call must not park a pending promise in the cache forever.
const PET_THUMB_TIMEOUT_MS = 15000
const petThumbCache = new LruCache<string, Promise<null | string>>(PET_THUMB_CACHE_MAX)

interface PetThumbResult {
  dataUri?: string
  ok: boolean
}

function petThumbIcon(slug: string, spriteUrl: null | string | undefined): Promise<null | string> {
  if (!slug) {
    return Promise.resolve(null)
  }

  if (!petThumbCache.has(slug)) {
    const deadline = new Promise<null>(resolve => setTimeout(() => resolve(null), PET_THUMB_TIMEOUT_MS))

    const pending = Promise.race([
      host
        .request<PetThumbResult>('pet.thumb', { slug, url: spriteUrl || '' })
        .then(result => (result?.ok && result.dataUri ? result.dataUri : null))
        .catch(() => null),
      deadline
    ]).then(icon => {
      // Never cache a failure: a transient backend error must not poison
      // the tile for the rest of the session.
      if (!icon) {
        petThumbCache.delete(slug)
      }

      return icon
    })

    petThumbCache.set(slug, pending)
  }

  return petThumbCache.get(slug)!
}

interface PetThumbProps {
  size?: number
  slug: string
  spriteUrl?: null | string
}

/** One pet tile image: server-cropped frame 0, resolved lazily through the cache. */
function PetThumb({ slug, spriteUrl, size = 40 }: PetThumbProps) {
  const [icon, setIcon] = useState<null | string>(null)
  useEffect(() => {
    let alive = true
    petThumbIcon(slug, spriteUrl).then(url => {
      if (alive) {
        setIcon(url)
      }
    })

    return () => {
      alive = false
    }
  }, [slug, spriteUrl])

  if (!icon) {
    return (
      <div
        style={{
          width: size,
          height: size,
          borderRadius: 6,
          background: 'var(--chrome-action-hover, rgba(255,255,255,0.06))'
        }}
      />
    )
  }

  return (
    <img
      alt=""
      src={icon}
      style={{
        width: size,
        height: size,
        objectFit: 'contain',
        imageRendering: 'pixelated',
        borderRadius: 6
      }}
    />
  )
}

/** One petdex companion as `pet.gallery` reports it. */
interface PetGalleryEntry {
  curated?: boolean
  displayName?: string
  installed?: boolean
  slug: string
  /** Full animation sheet (1536×1872 webp); empty for locally hatched pets. */
  spritesheetUrl?: null | string
}

interface PetTabProps {
  image: null | string
  onImage: (image: null | string) => void
}

export function PetTab({ image, onImage }: PetTabProps) {
  const b = useBots()
  // Selection is dialog-local: committed by the dialog's Save like any
  // uploaded/generated image (a direct meta write here gets clobbered by
  // Save's own image state).
  const [selectedSlug, setSelectedSlug] = useState<null | string>(null)

  const { data, isLoading } = useQuery({
    queryKey: [ID, 'pet-gallery'],
    queryFn: () => host.request<{ pets?: PetGalleryEntry[] }>('pet.gallery', {}),
    staleTime: 300000
  })

  const [query, setQuery] = useState('')
  // Windowed rendering: the gallery is 4500+ pets — mounting an <img> per pet
  // froze the dialog. Render `limit` at a time and grow on scroll-to-bottom.
  const [limit, setLimit] = useState(24)
  const pets = data?.pets ?? []

  if (isLoading) {
    return (
      <div className="flex justify-center py-4">
        <GlyphSpinner className="text-(--ui-text-tertiary)" spinner="breathe" />
      </div>
    )
  }

  if (!pets.length) {
    return (
      <div className="px-2 py-3 text-center text-xs text-(--ui-text-tertiary)">
        No pets in the petdex gallery. Run `hermes pets` to explore.
      </div>
    )
  }

  const q = query.trim().toLowerCase()

  const filtered = q
    ? pets.filter(pet => (pet.displayName || '').toLowerCase().includes(q) || (pet.slug || '').includes(q))
    : pets

  // Installed and curated pets surface first — they're the likeliest picks.
  const ranked = filtered.slice().sort((a, b) => {
    const rank = (pet: PetGalleryEntry) => (pet.installed ? 0 : pet.curated ? 1 : 2)

    return rank(a) - rank(b)
  })

  const visible = ranked.slice(0, limit)

  const onScroll = (event: { currentTarget: HTMLDivElement }) => {
    const el = event.currentTarget

    if (el.scrollTop + el.clientHeight >= el.scrollHeight - 120 && limit < ranked.length) {
      setLimit(prev => Math.min(prev + 24, ranked.length))
    }
  }

  return (
    <div className="grid w-full gap-2">
      <div className="text-center text-[0.65rem] text-(--ui-text-quaternary)">{b.avatar.pickPet}</div>
      <Input
        className="h-7 text-xs"
        onChange={event => {
          setQuery(event.target.value)
          setLimit(24)
        }}
        placeholder={`Search ${pets.length} pets…`}
        value={query}
      />
      {image && selectedSlug ? (
        <Button
          className="justify-center"
          onClick={() => {
            setSelectedSlug(null)
            onImage(null)
          }}
          size="sm"
          type="button"
          variant="ghost"
        >
          {b.avatar.removeBackToShape}
        </Button>
      ) : null}
      {filtered.length === 0 ? (
        <div className="py-3 text-center text-xs text-(--ui-text-quaternary)">No pets match.</div>
      ) : (
        <div
          className="overflow-y-auto"
          onScroll={onScroll}
          style={{
            // Leave room for the selection ring outside boundary tiles.
            maxHeight: 220,
            padding: 2
          }}
        >
          <div className="grid grid-cols-3 gap-1.5">
            {visible.map(pet => (
              <RowButton
                className={cn(
                  'grid justify-items-center gap-1 rounded-md p-1.5 transition-colors hover:bg-(--chrome-action-hover)',
                  selectedSlug === pet.slug && 'ring-1 ring-(--ui-accent)'
                )}
                key={pet.slug}
                onClick={() => {
                  // The pet IS the profile picture: the server-cropped frame 0
                  // becomes the dialog's avatar image (works for locally
                  // hatched pets too — they have no spritesheet URL at all).
                  // Persisted when the user hits Save.
                  setSelectedSlug(pet.slug)
                  void petThumbIcon(pet.slug, pet.spritesheetUrl).then(icon => {
                    if (icon) {
                      onImage(icon)
                    } else {
                      setSelectedSlug(null)
                      host.notify({
                        kind: 'error',
                        message: b.avatar.petLoadFailed
                      })
                    }
                  })
                }}
              >
                <PetThumb size={40} slug={pet.slug} spriteUrl={pet.spritesheetUrl} />
                <span className="w-full truncate text-center text-[0.6rem] text-(--ui-text-tertiary)">
                  {pet.displayName}
                </span>
              </RowButton>
            ))}
          </div>
          {limit < ranked.length ? (
            <div className="py-2 text-center text-[0.65rem] text-(--ui-text-quaternary)">{`Scroll for more (${limit} of ${ranked.length})`}</div>
          ) : null}
        </div>
      )}
    </div>
  )
}
