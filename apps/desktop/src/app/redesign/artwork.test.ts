import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { $shellChoice } from '@/store/shell-variant'

import {
  $artworkChoice,
  $paintedArtwork,
  type Artwork,
  ARTWORK_CHOICE_KEY,
  ARTWORKS,
  ARTWORK_KEY,
  artworkForLaunch,
  artworkForMode,
  type ArtworkMode,
  artworksFor,
  chooseArtwork,
  installArtworkAttributes,
  nextArtwork
} from './artwork'

function launches(mode: ArtworkMode, count: number): Artwork[] {
  let stored: unknown = null
  const seen: Artwork[] = []

  for (let index = 0; index < count; index += 1) {
    const launch = artworkForLaunch(stored, `launch-${index}`, mode)

    seen.push(launch.artwork)
    stored = JSON.parse(JSON.stringify(launch))
  }

  return seen
}

describe('the launch rotation', () => {
  it('walks every dark artwork in Figma order, then starts over', () => {
    expect(launches('dark', 11)).toEqual([...ARTWORKS, 'ascend'])
  })

  it('alternates the two light artworks', () => {
    expect(launches('light', 4)).toEqual(['ascend', 'vanilla', 'ascend', 'vanilla'])
  })

  it('never paints the same artwork on two launches in a row, even across a theme change', () => {
    let stored: unknown = null
    let previous: Artwork | null = null

    for (let index = 0; index < 12; index += 1) {
      const launch = artworkForLaunch(stored, `launch-${index}`, index % 3 === 0 ? 'light' : 'dark')

      expect(launch.artwork).not.toBe(previous)
      previous = launch.artwork
      stored = launch
    }
  })

  it('keeps the pick for every window of one launch', () => {
    const first = artworkForLaunch({ launch: 'old', artwork: 'angel' }, 'now', 'dark')

    expect(first.artwork).toBe('sword')
    expect(artworkForLaunch(first, 'now', 'dark')).toEqual(first)
  })

  it('starts the cycle over from an unreadable record', () => {
    expect(artworkForLaunch({ launch: 'old', artwork: 'retired' }, 'now', 'dark').artwork).toBe('ascend')
    expect(artworkForLaunch('garbage', 'now', 'light').artwork).toBe('ascend')
  })

  it('maps a pick onto the other theme by name, else to the next artwork that theme draws', () => {
    expect(artworkForMode('vanilla', 'light')).toBe('vanilla')
    expect(artworkForMode('ascend', 'light')).toBe('ascend')
    expect(artworkForMode('wing', 'light')).toBe('vanilla')
    expect(artworkForMode('angel', 'light')).toBe('vanilla')
    expect(artworkForMode('throne', 'light')).toBe('vanilla')
    expect(artworkForMode('karborn', 'light')).toBe('vanilla')
    expect(artworkForMode('karborn', 'dark')).toBe('karborn')
    expect(nextArtwork('karborn', 'dark')).toBe('vanilla')
  })
})

describe('installArtworkAttributes', () => {
  let root: HTMLElement

  let dispose = () => {}

  beforeEach(() => {
    root = document.createElement('html')
    localStorage.clear()
    localStorage.setItem('hermes-boot-color-scheme', 'dark')
  })

  afterEach(() => {
    dispose()
    $shellChoice.set(null)
    $artworkChoice.set(null)
    localStorage.clear()
  })

  it('names the launch artwork for both themes before the first paint, and records the launch', () => {
    localStorage.setItem(ARTWORK_KEY, JSON.stringify({ launch: 'previous-launch', artwork: 'wing' }))
    dispose = installArtworkAttributes(root)

    expect(root.dataset.artworkDark).toBe('karborn')
    expect(root.dataset.artworkLight).toBe('vanilla')
    expect(JSON.parse(localStorage.getItem(ARTWORK_KEY) ?? 'null')).toMatchObject({ artwork: 'karborn' })
  })

  it('gives a second window of the same launch the same artwork', () => {
    dispose = installArtworkAttributes(root)
    const first = root.dataset.artworkDark
    const second = document.createElement('html')
    const disposeSecond = installArtworkAttributes(second)

    expect(second.dataset.artworkDark).toBe(first)
    disposeSecond()
  })

  it('advances in the theme the window opens in', () => {
    localStorage.setItem('hermes-boot-color-scheme', 'light')
    localStorage.setItem(ARTWORK_KEY, JSON.stringify({ launch: 'previous-launch', artwork: 'vanilla' }))
    dispose = installArtworkAttributes(root)

    expect(root.dataset.artworkLight).toBe('ascend')
  })

  it('leaves classic windows without artwork and without touching the rotation', () => {
    $shellChoice.set(false)
    dispose = installArtworkAttributes(root)

    expect(root.dataset.artworkDark).toBeUndefined()
    expect(localStorage.getItem(ARTWORK_KEY)).toBeNull()

    $shellChoice.set(true)
    expect(root.dataset.artworkDark).toBe('ascend')
    $shellChoice.set(false)
    expect(root.dataset.artworkDark).toBeUndefined()
  })

  it('paints a chosen artwork at once, in both themes, and keeps the choice', () => {
    dispose = installArtworkAttributes(root)
    chooseArtwork('karborn')

    // Light does not draw Karborn: it paints the next artwork it does, as a theme switch would.
    expect(root.dataset.artworkDark).toBe('karborn')
    expect(root.dataset.artworkLight).toBe('vanilla')
    expect($paintedArtwork.get()).toEqual({ dark: 'karborn', light: 'vanilla' })
    expect(localStorage.getItem(ARTWORK_CHOICE_KEY)).toBe('karborn')
  })

  it('beats the rotation on every later launch, which it leaves where it was', async () => {
    localStorage.setItem(ARTWORK_KEY, JSON.stringify({ launch: 'previous-launch', artwork: 'wing' }))
    localStorage.setItem(ARTWORK_CHOICE_KEY, 'ascend')
    vi.resetModules()
    const fresh = await import('./artwork')
    const { $shellChoice: freshShell } = await import('@/store/shell-variant')
    freshShell.set(true)
    const disposeFresh = fresh.installArtworkAttributes(root)

    expect(root.dataset.artworkDark).toBe('ascend')
    expect(root.dataset.artworkLight).toBe('ascend')
    expect(JSON.parse(localStorage.getItem(ARTWORK_KEY) ?? 'null')).toEqual({
      launch: 'previous-launch',
      artwork: 'wing'
    })
    disposeFresh()
    freshShell.set(null)
  })
})

describe('artworksFor', () => {
  it('offers only what the mode draws, in the cycle order', () => {
    expect(artworksFor('dark')).toEqual(ARTWORKS)
    expect(artworksFor('light')).toEqual(['ascend', 'vanilla'])
  })
})
