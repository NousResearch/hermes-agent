import { expect, it } from 'vitest'

import manifestData from '../../../../public/lunar-city/v2-review/world-manifest.v2.json'
import { parseWorldManifest } from '../manifest'

import { isLocalInteriorReview } from './create-world'
import { eligibleInteriorPlans } from './interior-plan-data'

it('binds plans to actual exterior bytes and metre placement, rejecting a changed source or placement independently', () => {
  const manifest = parseWorldManifest(manifestData)
  const original = eligibleInteriorPlans(manifest)
  expect(original.length).toBeGreaterThan(0)
  const id = original[0].id
  const model = manifest.models.find(item => item.id === id)!

  for (const changed of [
    { ...model, statistics: { ...model.statistics, sha256: '0'.repeat(64) } },
    { ...model, uri: 'models/different.glb' },
    { ...model, transform: { ...model.transform, position: { ...model.transform.position, x: model.transform.position.x + 1 } } },
    { ...model, transform: { ...model.transform, scale: { ...model.transform.scale, y: model.transform.scale.y * 2 } } }
  ]) {
    const eligible = eligibleInteriorPlans({ ...manifest, models: manifest.models.map(item => item.id === id ? changed : item) })
    expect(eligible.some(plan => plan.id === id)).toBe(false)
    expect(eligible.map(plan => plan.id)).toEqual(original.filter(plan => plan.id !== id).map(plan => plan.id))
  }
})

it('only opts in for the local review manifest, never the standard pack or a foreign origin', () => {
  const page = new URL('http://127.0.0.1:5178/lunar-city-review.html')
  expect(isLocalInteriorReview(new URL('/lunar-city/v2-review/world-manifest.v2.json', page), page)).toBe(true)
  expect(isLocalInteriorReview(new URL('/lunar-city/v2/world-manifest.v2.json', page), page)).toBe(false)
  expect(isLocalInteriorReview(new URL('https://elsewhere.test/lunar-city/v2-review/world-manifest.v2.json'), page)).toBe(false)
})
