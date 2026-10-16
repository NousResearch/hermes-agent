import { describe, expect, it } from 'vitest'

import {
  blobatarExpressionFor,
  blobatarMotionAvailable,
  mapBotMoodToExpression
} from './avatar-motion'
import { blobMarkup } from './avatar'

describe('bot mood -> blobatar expression mapping', () => {
  it('maps the three bot moods onto the expression roster', () => {
    expect(mapBotMoodToExpression('idle')).toBeUndefined()
    expect(mapBotMoodToExpression('think')).toBe('thinking')
    expect(mapBotMoodToExpression('work')).toBe('happy')
  })
})

describe('blobatar motion feature detection', () => {
  it('resolves the animated component and expression roster in this build', async () => {
    // blobatar 2.0.0 publishes both subpaths; when a future SDK drops one,
    // these trip and every caller falls back to the static path.
    expect(await blobatarMotionAvailable()).toBe(true)
  })

  it('resolves the mood expressions by name from the roster', async () => {
    const thinking = await blobatarExpressionFor('think')

    expect(thinking).toBeDefined()
    expect(await blobatarExpressionFor('work')).toBeDefined()
    // idle needs no pose: the ambient loop is the idle personality.
    expect(await blobatarExpressionFor('idle')).toBeUndefined()
  })
})

describe('static fallback path unchanged', () => {
  it('still renders the static svg with the roster backfill hook', () => {
    const markup = blobMarkup('blobatar', 'atlas', 34)

    expect(markup).toContain('data-bot-face="atlas"')
    expect(markup).not.toContain('mo-root')
  })
})

describe('animated path (real blobatar/react component)', () => {
  it('renders the animated svg with the backfill hook and the mood expression', async () => {
    // With motion available, BotFace renders the library's component: the
    // data-bot-face hook must land on the svg (roster PNG backfill) and the
    // mood's expression rides along. The dynamic import resolves in an
    // effect, so the first paint is the static path — assert on the
    // post-resolution render.
    const { BotFace } = await import('./avatar')
    const { cleanup, render, waitFor } = await import('@testing-library/react')

    const { container } = render(
      <BotFace color="#38bdf8" mood="think" name="inbox-triage" shape="blobatar" size={56} />
    )

    try {
      await waitFor(
        () => {
          const svg = container.querySelector('svg[data-bot-face="inbox-triage"]')

          expect(svg).toBeTruthy()
          // The motion root class hangs on the inner <g> (the library renders
          // the svg itself): blink/saccade/breathe/bob all run from it.
          expect(svg!.querySelector('.mo-root')).toBeTruthy()
          // The animated render wraps eyes in .mo-eyes with seeded phases.
          expect(svg!.querySelector('.mo-eyes')).toBeTruthy()
        },
        { timeout: 2_000 }
      )
    } finally {
      cleanup()
    }
  })
})
