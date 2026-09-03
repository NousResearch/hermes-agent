/**
 * The avatar face: blobatar shape strings, the markup they render to, and the
 * catchlight polarity fix.
 *
 * Shape strings are stored per bot and must round-trip forever —
 * `blobatar[:seed[:kind]]`, where an unlocked seed follows the bot's name and
 * an unknown silhouette is ignored rather than trusted. The silhouette is
 * pinned by handing the library a TRAIT position inside its frozen band, so
 * the band a kind lands in is a stored-appearance contract, not an
 * implementation detail.
 *
 * Catchlight (image14 report, Aug 2026): the sparkle's contrast follows the
 * PUPIL, not the body. Dark bodies flip the pupils to light cream, and a white
 * catchlight on a cream pupil is invisible — maroon/ink/oxblood avatars looked
 * like they had "no dots in their eyes".
 */

import { render } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'

const { blobatarSvgMock } = vi.hoisted(() => ({ blobatarSvgMock: vi.fn() }))

vi.mock('@hermes/plugin-sdk', async () => {
  const { atom } = await import('nanostores')

  return {
    atom,
    blobatarSvg: (seed: string, opts: unknown) => blobatarSvgMock(seed, opts) as string,
    createBudgetedLoop: undefined,
    host: { state: { connectionId: { get: () => 'local' } } },
    profileColor: (name: string) => (name === 'inbox-triage' ? '#38bdf8' : '#8b5cf6'),
    PROFILE_SWATCHES: ['#38bdf8', '#8b5cf6'],
    queryClient: undefined,
    useQuery: vi.fn(),
    useValue: vi.fn()
  }
})

vi.mock('./shared', () => ({ getPluginCtx: () => null, ID: 'hermes-bots' }))

/** The library's frozen band per silhouette (gen2 thresholds). */
const BANDS: Record<string, [number, number]> = {
  boxy: [0.48, 0.6],
  capsule: [0.6, 0.7],
  cloud: [0.79, 0.86],
  droplet: [0.86, 0.915],
  hexagon: [0.915, 0.95],
  nub: [0.7, 0.79],
  organic: [0.22, 0.48],
  round: [0, 0.22],
  sun: [0.95, 0.98],
  triangle: [0.98, 1]
}

interface BlobOptions {
  size: number
  traits?: { shape: number }
}

const lastBlobCall = () => blobatarSvgMock.mock.calls.at(-1) as [string, BlobOptions]

beforeEach(() => {
  vi.clearAllMocks()
  blobatarSvgMock.mockReturnValue('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"></svg>')
})

describe('blob shape strings round-trip through parse/build', () => {
  it('recognizes only the blobatar family', async () => {
    const { isBlobShape } = await import('./avatar')

    expect(isBlobShape('blobatar')).toBe(true)
    expect(isBlobShape('blobatar:seed123')).toBe(true)
    expect(isBlobShape('blobatar::sun')).toBe(true)
    expect(isBlobShape('circle')).toBe(false)
    expect(isBlobShape(undefined)).toBe(false)
  })

  it('parses the seed/silhouette segments, ignoring an unknown silhouette', async () => {
    const { parseBlobShape } = await import('./avatar')

    // Unlocked: the seed follows the name.
    expect(parseBlobShape('blobatar', 'inbox-triage')).toEqual({ kind: '', seed: 'inbox-triage', seedPart: '' })
    expect(parseBlobShape('blobatar:abc123', 'inbox-triage')).toEqual({
      kind: '',
      seed: 'abc123',
      seedPart: 'abc123'
    })
    expect(parseBlobShape('blobatar::cloud', 'inbox-triage')).toEqual({
      kind: 'cloud',
      seed: 'inbox-triage',
      seedPart: ''
    })
    expect(parseBlobShape('blobatar:abc:mystery', 'x').kind).toBe('')
  })

  it('rebuilds every segment combination', async () => {
    const { blobShapeString } = await import('./avatar')

    expect(blobShapeString('', '')).toBe('blobatar')
    expect(blobShapeString('abc', '')).toBe('blobatar:abc')
    expect(blobShapeString('abc', 'sun')).toBe('blobatar:abc:sun')
    expect(blobShapeString('', 'sun')).toBe('blobatar::sun')
  })
})

describe('rendering a blob face', () => {
  it('tags the markup data-bot-face so the roster PNG backfill can find it', async () => {
    // pushLocalAvatars → rasterizeSvgToPng queries `svg[data-bot-face=…]`;
    // without the tag a vector face never reaches the inter-agent notices.
    const { BotFace } = await import('./avatar')
    const { container } = render(<BotFace color="#38bdf8" name="inbox-triage" shape="blobatar" size={56} />)

    expect(container.querySelector('svg[data-bot-face="inbox-triage"]')).toBeTruthy()

    const [seed, opts] = lastBlobCall()

    expect(seed).toBe('inbox-triage')
    expect(opts.size).toBe(56)
    // No pinned silhouette means no traits at all — the library picks.
    expect('traits' in opts).toBe(false)
  })

  it('passes the locked seed and the pinned silhouette’s trait', async () => {
    const { BotFace } = await import('./avatar')

    render(<BotFace color="#38bdf8" name="inbox-triage" shape="blobatar:abc:sun" size={32} />)

    const [seed, opts] = lastBlobCall()

    expect(seed).toBe('abc')
    expect(opts.traits?.shape).toBeGreaterThanOrEqual(BANDS.sun[0])
    expect(opts.traits?.shape).toBeLessThan(BANDS.sun[1])
  })

  it('puts every silhouette inside its own frozen band', async () => {
    const { BLOB_KINDS, BotFace } = await import('./avatar')

    for (const kind of BLOB_KINDS) {
      render(<BotFace color="#38bdf8" name="agent" shape={`blobatar::${kind}`} size={32} />)

      const trait = lastBlobCall()[1].traits?.shape

      expect(typeof trait, kind).toBe('number')
      expect(trait, kind).toBeGreaterThanOrEqual(BANDS[kind][0])
      expect(trait, kind).toBeLessThan(BANDS[kind][1])
    }
  })

  it('falls back to the legacy math face when the renderer throws', async () => {
    blobatarSvgMock.mockImplementation(() => {
      throw new Error('boom')
    })

    const { BotFace } = await import('./avatar')
    const { container } = render(<BotFace color="#38bdf8" name="agent" shape="blobatar" size={32} />)

    expect(container.querySelector('svg[data-hb-math]')).toBeTruthy()
  })
})

describe('catchlight contrast follows the pupil, not the body', () => {
  const catchlights = (container: HTMLElement) =>
    ['l', 'r'].map(side => container.querySelector(`[data-hb-hl-${side}]`)?.getAttribute('fill'))

  it('puts a DARK sparkle on the cream pupils of a dark body', async () => {
    const { BotFace } = await import('./avatar')
    const { container } = render(<BotFace color="#3b0910" name="oxblood" shape="circle" />)

    expect(catchlights(container)).toEqual(['rgba(0,0,0,0.6)', 'rgba(0,0,0,0.6)'])
  })

  it('keeps the white sparkle on the dark pupils of a light body', async () => {
    const { BotFace } = await import('./avatar')
    const { container } = render(<BotFace color="#f5e6c8" name="cream" shape="circle" />)

    expect(catchlights(container)).toEqual(['rgba(255,255,255,0.85)', 'rgba(255,255,255,0.85)'])
  })
})

// Last: re-mocking the SDK re-links the whole avatar graph, so anything after
// this would be running against the swapped module.
describe('an SDK that predates blobatarSvg', () => {
  it('renders the legacy deterministic shape instead of nothing', async () => {
    vi.resetModules()
    vi.doMock('@hermes/plugin-sdk', async () => {
      const { atom } = await import('nanostores')

      return {
        atom,
        blobatarSvg: undefined,
        createBudgetedLoop: undefined,
        host: { state: { connectionId: { get: () => 'local' } } },
        profileColor: () => '#8b5cf6',
        PROFILE_SWATCHES: [],
        queryClient: undefined,
        useQuery: vi.fn(),
        useValue: vi.fn()
      }
    })

    const { BotFace, defaultShapeFor } = await import('./avatar')
    const { container } = render(<BotFace color="#38bdf8" name="agent" shape="blobatar" size={32} />)

    expect(container.querySelector('svg[data-hb-math]')?.getAttribute('data-hb-shape')).toBe(defaultShapeFor('agent'))
  })
})
