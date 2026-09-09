'use client'

import { useEffect, useState } from 'react'

import { WIDGET_SHELL_CLASS } from '@/components/chat/widget-shell'
import { ImageLightbox } from '@/components/chat/zoomable-image'
import { Codicon } from '@/components/ui/codicon'
import { Tip } from '@/components/ui/tooltip'
import { useImageDownload } from '@/hooks/use-image-download'
import { useI18n } from '@/i18n'
import { ExternalLink } from '@/lib/external-link'
import { type Listing, parseListings, specLine } from '@/lib/listing-embed'
import { cn } from '@/lib/utils'

import type { RichFenceProps } from './types'

/** The conversation's own type scale, so a card reads at the prose's size. */
const TEXT_CLASS = 'text-[length:var(--conversation-text-font-size)]'
const META_CLASS = 'text-[length:var(--conversation-tool-font-size)]'

/**
 * ```listing — a property listing as a native transcript card.
 *
 * The portals can't be framed (no oEmbed, deprecated APIs, X-Frame-Options),
 * so the agent authors the card from what it already gathered and this
 * renderer makes it browsable.
 *
 * It sits on the same shell as every other inline widget
 * (`WIDGET_SHELL_CLASS`: one radius above the composer, mode-derived fill, no
 * border) and uses the conversation's own type scale, so a listing reads as
 * app surface rather than a pasted webpage.
 *
 * Facts ride a quiet meta line; catches get ONE muted line, not a wall of
 * amber badges — a card that shouts every caveat has no hierarchy left for
 * the one that matters.
 */
export default function ListingEmbed({ code, streaming }: RichFenceProps) {
  // Mid-stream the JSON is a prefix and JSON.parse fails, which would flash
  // the raw fence; hold the fallback until the turn settles.
  const listings = streaming ? [] : parseListings(code)

  if (listings.length === 0) {
    // Nothing valid parsed: the boundary's fallback (the plain code block) is
    // the honest render. Throwing lets RichBoundary do it.
    throw new Error('listing: no renderable listings')
  }

  return (
    // Outer breathing room only: `my-*` stands the SET off the prose around
    // it, while `gap-2` owns the space between stacked cards — so two
    // listings stay tight to each other but neither butts into a message.
    // The transcript's own paragraph rhythm, so a card is spaced like any
    // other block; the prose container's first/last-child rules collapse it
    // at a message's edges.
    <span className="my-(--paragraph-gap) flex w-full max-w-128 flex-col gap-2" data-slot="aui_listing-set">
      {listings.map(listing => (
        <ListingCard key={listing.id} listing={listing} />
      ))}
    </span>
  )
}

function ListingCard({ listing }: { listing: Listing }) {
  const specs = specLine(listing)
  // Facts are supporting detail, not headline — one dot-joined meta line
  // beside the specs reads as one thought instead of a badge grid.
  const meta = [specs, ...listing.facts].filter(Boolean).join(' · ')

  return (
    // The shell's fill and radius, but NOT its uniform padding: the photos
    // bleed to the card's rounded edge (clipped by `overflow-hidden`) the way
    // every portal card is built, and only the text carries the shell's inset.
    //
    // `not-prose` is load-bearing. The transcript ancestor is `.aui-md.prose`,
    // and the Typography plugin's `:where(img)` rule adds `2em` of margin to
    // every image inside it — a band of dead card fill above the hero. That
    // selector outranks a `my-0` utility, so the fix is the plugin's own
    // opt-out, which excludes this whole subtree.
    <span className={cn(WIDGET_SHELL_CLASS, 'not-prose block overflow-hidden p-0')} data-slot="aui_listing-card">
      {listing.images.length > 0 && <ListingGallery address={listing.address} images={listing.images} />}

      <span className="block px-3.5 py-3">
        <span className="flex flex-wrap items-baseline justify-between gap-x-3">
          <span className={cn(TEXT_CLASS, 'font-medium text-foreground')}>{listing.address}</span>
          {listing.price && (
            <span className={cn(TEXT_CLASS, 'font-medium tabular-nums text-foreground')}>{listing.price}</span>
          )}
        </span>

        {meta && <span className={cn(META_CLASS, 'mt-0.5 block tabular-nums text-muted-foreground')}>{meta}</span>}

        {listing.note && (
          <span className={cn(TEXT_CLASS, 'mt-1.5 block leading-relaxed text-foreground/80')}>{listing.note}</span>
        )}

        {listing.catches.length > 0 && (
          <span className={cn(META_CLASS, 'mt-2 flex gap-1.5 text-muted-foreground')}>
            <Codicon className="mt-px shrink-0 text-amber-500" name="warning" size="0.75rem" />
            <span className="min-w-0">{listing.catches.join(' · ')}</span>
          </span>
        )}

        {listing.links.length > 0 && (
          <span className={cn(META_CLASS, 'mt-2 flex flex-wrap gap-x-3 gap-y-1')}>
            {listing.links.map(link => (
              <ExternalLink href={link.url} key={link.url}>
                {link.label}
              </ExternalLink>
            ))}
          </span>
        )}
      </span>
    </span>
  )
}

/**
 * The browse surface: a photo MOSAIC, the way the portals themselves show a
 * property — one hero carrying the house plus supporting frames, all visible
 * at once. A single paged hero made you click to learn anything; a grid lets
 * you take in the place in one look and dive only where you want.
 *
 * The tiling adapts to how many photos exist rather than forcing every
 * listing into one shape:
 *   1 photo  → a lone 4:3 hero
 *   2 photos → a 3:2 pair, side by side
 *   3+       → hero on the left (2 cols × 2 rows) + two stacked frames right
 *
 * Overflow past the visible tiles folds into a `+N` on the last frame, so the
 * gallery's depth is legible without paying rows for it. Clicking any frame
 * opens the shared lightbox at that photo; ← / → page the whole set there,
 * which is where a big image belongs anyway.
 */
function ListingGallery({ address, images }: { address: string; images: string[] }) {
  const { t } = useI18n()
  const [broken, setBroken] = useState<string[]>([])
  const [openAt, setOpenAt] = useState<number | null>(null)

  // Portal CDNs expire URLs; a dead photo drops out of the mosaic rather than
  // leaving a broken frame in the grid.
  const live = images.filter(src => !broken.includes(src))
  const position = openAt === null ? 0 : Math.min(openAt, live.length - 1)
  const { download, saving } = useImageDownload(live[position])

  // Page the set inside the lightbox — a full-size photo is the right place
  // to go through 40 frames, and the arrow keys are what hands reach for.
  useEffect(() => {
    if (openAt === null || live.length < 2) {
      return
    }

    const onKey = (event: KeyboardEvent) => {
      const delta = event.key === 'ArrowRight' ? 1 : event.key === 'ArrowLeft' ? -1 : 0

      if (delta !== 0) {
        event.preventDefault()
        setOpenAt(prev => ((prev ?? 0) + delta + live.length) % live.length)
      }
    }

    window.addEventListener('keydown', onKey)

    return () => window.removeEventListener('keydown', onKey)
  }, [live.length, openAt])

  if (live.length === 0) {
    return null
  }

  // Tiles the layout shows; anything beyond folds into the `+N` badge.
  const mosaic = live.length >= 3
  const visible = live.slice(0, mosaic ? 3 : live.length)

  return (
    <>
      <span
        className={cn(
          // No radius of its own: the photos bleed to the card's edge and the
          // card's `overflow-hidden` clips the top corners for them.
          'grid w-full gap-1',
          live.length === 1 && 'aspect-[4/3]',
          live.length === 2 && 'aspect-[3/2] grid-cols-2',
          mosaic && 'aspect-[3/2] grid-cols-3 grid-rows-2'
        )}
        data-slot="aui_listing-gallery"
      >
        {visible.map((src, tile) => (
          <GalleryTile
            address={address}
            // The hero earns its scale only in the 3+ mosaic; with one or two
            // photos every frame is equal.
            className={cn(mosaic && tile === 0 && 'col-span-2 row-span-2')}
            key={src}
            label={t.desktop.openImage}
            more={tile === visible.length - 1 ? live.length - visible.length : 0}
            onError={() => setBroken(prev => (prev.includes(src) ? prev : [...prev, src]))}
            onOpen={() => setOpenAt(tile)}
            src={src}
          />
        ))}
      </span>

      <ImageLightbox
        alt={address}
        copy={t.desktop}
        onClick={download}
        onOpenChange={open => setOpenAt(open ? position : null)}
        open={openAt !== null}
        saving={saving}
        src={live[position]}
      />
    </>
  )
}

function GalleryTile({
  address,
  className,
  more,
  onError,
  onOpen,
  src,
  label
}: {
  address: string
  className?: string
  /** Photos past the visible tiles; rendered as `+N` over this frame. */
  more: number
  onError: () => void
  onOpen: () => void
  src: string
  label: string
}) {
  return (
    <Tip label={label}>
      <button
        className={cn('relative block size-full cursor-zoom-in overflow-hidden bg-muted/55', className)}
        onClick={onOpen}
        type="button"
      >
        <img
          alt={address}
          className="size-full object-cover transition-opacity hover:opacity-90"
          loading="lazy"
          onError={onError}
          referrerPolicy="no-referrer"
          src={src}
        />
        {more > 0 && (
          <span
            className={cn(
              TEXT_CLASS,
              'absolute inset-0 grid place-items-center bg-background/55 font-medium tabular-nums text-foreground backdrop-blur-[2px]'
            )}
          >
            +{more}
          </span>
        )}
      </button>
    </Tip>
  )
}
