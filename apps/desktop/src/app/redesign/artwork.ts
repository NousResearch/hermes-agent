// The redesign's artwork (Figma "Prod - Desktop" › Brand, frame 16292:10764): every app launch
// paints the next artwork of the theme's set behind the shell, until the person chooses one in "Make
// it yours" (Onboarding 18695:18059), which it then paints for good. The art itself is CSS
// (portal-artwork.css); this module only decides which one and mirrors it onto <html>.
import { randomUuid } from '@hermes/shared'
import { atom, computed } from 'nanostores'

import { persistentAtom } from '@/lib/persisted'
import { readJson, readKey, writeJson } from '@/lib/storage'
import { $portalShell } from '@/store/shell-variant'

/** The cycle every theme walks, and the order "Make it yours" offers them: the picker's own order
 *  (Onboarding 18695:18059, 18695:18219: Ascend, Cathedral, Forest, Angel, Sword, Throne, Lattice),
 *  then the Brand rows it does not show, with Vanilla (a faint watermark, close to no artwork) last so
 *  a first launch opens on a picture. */
export const ARTWORKS = [
  'ascend',
  'cathedral',
  'forest',
  'angel',
  'sword',
  'throne',
  'lattice',
  'wing',
  'karborn',
  'vanilla'
] as const

export type Artwork = (typeof ARTWORKS)[number]
export type ArtworkMode = 'dark' | 'light'

/** Light draws two of them (Brand rows "Light/Ascend" and "Light/Wing", the latter Vanilla's hatch);
 *  the Brand page draws no light version of the others. */
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

/** The artworks `mode` draws, in the cycle's order: what "Make it yours" offers in that mode. */
export const artworksFor = (mode: ArtworkMode): readonly Artwork[] =>
  ARTWORKS.filter(artwork => MODE_ARTWORKS[mode].has(artwork))

/** What `mode` paints for the launch's pick: the same artwork when it draws it, else the next. */
export const artworkForMode = (pick: Artwork, mode: ArtworkMode): Artwork =>
  MODE_ARTWORKS[mode].has(pick) ? pick : nextArtwork(pick, mode)

export interface ArtworkLaunch {
  launch: string
  artwork: Artwork
}

const isArtwork = (value: unknown): value is Artwork => ARTWORKS.includes(value as Artwork)

/** No artwork: the ground colour and the panels' grain alone. A choice, never part of the cycle. */
export const NO_ARTWORK = 'none'

export type ArtworkChoice = Artwork | typeof NO_ARTWORK

/** The artwork the person chose (or none at all), or null: never chosen, so the launches rotate. An
 *  appearance preference like the accent (a machine-level UI preference, lib/ui-preferences/keys.ts),
 *  where the rotation itself stays per device. A stored value this build does not know reads as no
 *  choice, which is also what an older build makes of "none". */
export const ARTWORK_CHOICE_KEY = 'hermes.desktop.portalArtwork.v1'

export const $artworkChoice = persistentAtom<ArtworkChoice | null>(ARTWORK_CHOICE_KEY, null, {
  decode: raw => (raw === NO_ARTWORK || isArtwork(raw) ? raw : null),
  encode: value => value
})

export function chooseArtwork(artwork: Artwork): void {
  $artworkChoice.set(artwork)
}

/** Settings' artwork switch. Off paints none; on again goes back to the rotating launches (the
 *  welcome's "Make it yours" is where one artwork is picked for good). */
export function setArtworkShown(shown: boolean): void {
  $artworkChoice.set(shown ? null : NO_ARTWORK)
}

/** What each theme paints right now, as mirrored onto <html>; null where the redesign is not shown. */
export const $paintedArtwork = atom<null | Record<ArtworkMode, ArtworkChoice>>(null)

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

/** Mirrors the artwork onto `<html data-artwork-dark/-light>` before the first paint, so a theme
 *  switch is a class flip with nothing to decide: the person's choice, else the launch's pick, made
 *  the first time this window renders the redesign without a choice. Classic windows (and the Browser,
 *  which renders classic) get none; the transparent HUD has no canvas to show its pick on. */
export function installArtworkAttributes(root: HTMLElement = document.documentElement): () => void {
  let launchPick: Artwork | null = null

  const launchArtwork = (): Artwork => {
    if (launchPick === null) {
      const launch = artworkForLaunch(readJson(ARTWORK_KEY), LAUNCH_ID, bootMode())

      writeJson(ARTWORK_KEY, launch)
      launchPick = launch.artwork
    }

    return launchPick
  }

  return computed([$portalShell, $artworkChoice], (portal, choice) => ({ portal, choice })).subscribe(
    ({ choice, portal }) => {
      if (!portal) {
        delete root.dataset.artworkDark
        delete root.dataset.artworkLight
        $paintedArtwork.set(null)

        return
      }

      const pick = choice ?? launchArtwork()

      const painted =
        pick === NO_ARTWORK
          ? { dark: pick, light: pick }
          : { dark: artworkForMode(pick, 'dark'), light: artworkForMode(pick, 'light') }

      root.dataset.artworkDark = painted.dark
      root.dataset.artworkLight = painted.light
      $paintedArtwork.set(painted)
    }
  )
}
