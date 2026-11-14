// The redesign's rotating artwork (Figma "Prod - Desktop" › Brand, frame 16292:10764): every app
// launch paints the next artwork of the theme's set behind the shell. The art itself is CSS
// (portal-artwork.css); this module only decides which one and mirrors it onto <html>.
import { randomUuid } from '@hermes/shared'

import { readJson, readKey, writeJson } from '@/lib/storage'
import { $portalShell } from '@/store/shell-variant'

/** The cycle every theme walks: Figma's rows, with Vanilla (a faint watermark, close to no artwork)
 *  moved to the end so a first launch opens on a picture. */
export const ARTWORKS = ['wing', 'angel', 'ascend', 'karborn', 'vanilla'] as const

export type Artwork = (typeof ARTWORKS)[number]
export type ArtworkMode = 'dark' | 'light'

/** Light draws two of them (rows "Ascension / Light" and "Vanilla / Light"). */
const MODE_ARTWORKS: Record<ArtworkMode, ReadonlySet<Artwork>> = {
  dark: new Set(ARTWORKS),
  light: new Set(['vanilla', 'ascend'])
}

/** The first artwork after `previous` in the cycle that `mode` draws; the cycle's first without one. */
export function nextArtwork(previous: Artwork | null, mode: ArtworkMode): Artwork {
  const after = previous === null ? 0 : ARTWORKS.indexOf(previous) + 1
  const cycle = [...ARTWORKS.slice(after), ...ARTWORKS.slice(0, after)]

  return cycle.find(candidate => MODE_ARTWORKS[mode].has(candidate)) ?? ARTWORKS[0]
}

/** What `mode` paints for the launch's pick: the same artwork when it draws it, else the next. */
export const artworkForMode = (pick: Artwork, mode: ArtworkMode): Artwork =>
  MODE_ARTWORKS[mode].has(pick) ? pick : nextArtwork(pick, mode)

export interface ArtworkLaunch {
  launch: string
  artwork: Artwork
}

const isArtwork = (value: unknown): value is Artwork => ARTWORKS.includes(value as Artwork)

/** The launch's artwork. A launch that already picked keeps its pick (a second window, a reload);
 *  a new one takes the next after the previous launch's, in the theme it opens in, so two
 *  launches in a row never paint the same art. */
export function artworkForLaunch(stored: unknown, launch: string, mode: ArtworkMode): ArtworkLaunch {
  const previous = stored as Partial<ArtworkLaunch> | null
  const artwork = isArtwork(previous?.artwork) ? previous.artwork : null

  if (artwork && previous?.launch === launch) {
    return { launch, artwork }
  }

  return { launch, artwork: nextArtwork(artwork, mode) }
}

/** Per device, not a shared preference: each machine and browser keeps its own rotation. */
export const ARTWORK_KEY = 'hermes.desktop.artwork.v1'

// The theme the window opens in, before the theme provider runs: the scheme it last painted
// (themes/context.tsx, the same key index.html pre-paints from), else the OS's.
const bootMode = (): ArtworkMode =>
  (readKey('hermes-boot-color-scheme') ?? (matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light')) ===
  'light'
    ? 'light'
    : 'dark'

/** Native windows share their main process's launch id; in a browser every page load is a launch. */
const LAUNCH_ID = window.hermesDesktop?.launchId ?? randomUuid()

/** Picks the launch's artwork the first time this window renders the redesign, and mirrors it onto
 *  `<html data-artwork-dark/-light>` before the first paint, so a theme switch is a class flip with
 *  nothing to decide. Classic windows (and the HUD and Browser, which render classic) get none. */
export function installArtworkAttributes(root: HTMLElement = document.documentElement): () => void {
  let pick: Artwork | null = null

  return $portalShell.subscribe(portal => {
    if (!portal) {
      delete root.dataset.artworkDark
      delete root.dataset.artworkLight

      return
    }

    if (pick === null) {
      const launch = artworkForLaunch(readJson(ARTWORK_KEY), LAUNCH_ID, bootMode())

      writeJson(ARTWORK_KEY, launch)
      pick = launch.artwork
    }

    root.dataset.artworkDark = artworkForMode(pick, 'dark')
    root.dataset.artworkLight = artworkForMode(pick, 'light')
  })
}
