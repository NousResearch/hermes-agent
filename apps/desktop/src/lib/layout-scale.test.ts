import { expect, it } from 'vitest'

import { scaleMediaQuery } from './layout-scale'

it('leaves a query alone at 100%', () => {
  expect(scaleMediaQuery('(max-width: 1023.98px)', 1)).toBe('(max-width: 1023.98px)')
})

it('asks size features against the room left at the scale', () => {
  // A 1440px window at 200% has 720px of room: it is below the 1024px breakpoint.
  expect(scaleMediaQuery('(max-width: 1023.98px)', 2)).toBe('(max-width: 2047.96px)')
  expect(scaleMediaQuery('(min-width: 640px) and (min-height: 30.01rem)', 1.5)).toBe(
    '(min-width: 960px) and (min-height: 45.015rem)'
  )
  expect(scaleMediaQuery('(max-width: 639.98px), (max-height: 30rem)', 0.9)).toBe(
    '(max-width: 575.982px), (max-height: 27rem)'
  )
})

it('scales the range syntax a stylesheet compiler writes', () => {
  expect(scaleMediaQuery('(width >= 48rem)', 2)).toBe('(width >= 96rem)')
  expect(scaleMediaQuery('(40rem <= width < 64rem)', 1.25)).toBe('(50rem <= width < 80rem)')
})

it('leaves features that are not a size as written', () => {
  const query = '(max-width: 639.98px) and (hover: none), (prefers-reduced-motion: reduce), (resolution: 2dppx)'

  expect(scaleMediaQuery(query, 2)).toBe(
    '(max-width: 1279.96px) and (hover: none), (prefers-reduced-motion: reduce), (resolution: 2dppx)'
  )
})
