/**
 * The Pet tab: browse the petdex gallery and use a companion's first sprite
 * frame as the bot's profile picture.
 */

import { Button, cn, GlyphSpinner, host, Input, RowButton, useQuery } from '@hermes/plugin-sdk'
import { useEffect, useState } from 'react'

import { useBots } from './i18n'
import { ID } from './shared'

// ── pet tab: attach a petdex companion that lives beside the avatar ─────────

// A petdex "spritesheet" is the FULL animation sheet (1536×1872 webp, ~2MB;
// 8×9 grid of 192×208 frames). Using it as an <img> both downloads megabytes
// per tile and shows the whole sheet squashed. Extract frame 0 once per slug
// via canvas, downscale to 96px, and cache the data URL. Concurrency-capped
// so opening the tab doesn't fire dozens of 2MB fetches at once.
const PET_FRAME_W = 192
const PET_FRAME_H = 208
const petFrameCache = new Map<string, Promise<null | string>>()
let petFetchActive = 0
const petFetchQueue: Array<() => Promise<void>> = []

function pumpPetQueue() {
  while (petFetchActive < 4 && petFetchQueue.length) {
    const job = petFetchQueue.shift()!
    petFetchActive++
    job().finally(() => {
      petFetchActive--
      pumpPetQueue()
    })
  }
}

function petFrameIcon(spriteUrl: null | string | undefined): Promise<null | string> {
  if (!spriteUrl) {
    return Promise.resolve(null)
  }

  if (!petFrameCache.has(spriteUrl)) {
    petFrameCache.set(
      spriteUrl,
      new Promise(resolve => {
        petFetchQueue.push(async () => {
          try {
            const resp = await fetch(spriteUrl, {
              signal: AbortSignal.timeout(15000)
            })

            const blob = await resp.blob()
            // Crop frame 0 during decode — never materialize the full sheet.
            const bitmap = await createImageBitmap(blob, 0, 0, PET_FRAME_W, PET_FRAME_H)
            const canvas = document.createElement('canvas')
            canvas.width = 96
            canvas.height = 104
            canvas.getContext('2d')!.drawImage(bitmap, 0, 0, 96, 104)
            bitmap.close()
            resolve(canvas.toDataURL('image/png'))
          } catch {
            petFrameCache.delete(spriteUrl)
            resolve(null)
          }
        })
        pumpPetQueue()
      })
    )
  }

  return petFrameCache.get(spriteUrl)!
}

interface PetThumbProps {
  size?: number
  spriteUrl?: null | string
}

/** One pet tile image: frame 0 only, resolved lazily through the cache. */
function PetThumb({ spriteUrl, size = 40 }: PetThumbProps) {
  const [icon, setIcon] = useState<null | string>(null)
  useEffect(() => {
    let alive = true
    petFrameIcon(spriteUrl).then(url => {
      if (alive) {
        setIcon(url)
      }
    })

    return () => {
      alive = false
    }
  }, [spriteUrl])

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
  /** Full animation sheet (1536×1872 webp); frame 0 is cropped out of it. */
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
      <div className="text-center text-[0.65rem] text-(--ui-text-quaternary)">
        {b.avatar.pickPet}
      </div>
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
          onScroll={onScroll}
          style={{
            maxHeight: 220,
            overflowY: 'auto'
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
                  // The pet IS the profile picture: extract frame 0
                  // and hand it to the dialog as the avatar image.
                  // Persisted when the user hits Save.
                  setSelectedSlug(pet.slug)
                  void petFrameIcon(pet.spritesheetUrl).then(icon => {
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
                <PetThumb size={40} spriteUrl={pet.spritesheetUrl} />
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
