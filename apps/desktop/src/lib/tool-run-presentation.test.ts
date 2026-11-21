import { describe, expect, it } from 'vitest'

import type { Contribution } from '@/contrib/types'

import { resolveToolRunPresentation } from './tool-run-presentation'

const pref = (id: string, mode: string, order?: number, apiVersion: unknown = 1): Contribution => ({
  id,
  area: 'chat.toolRun.presentation',
  order,
  data: { apiVersion, mode }
})

describe('tool run presentation preference resolution', () => {
  it('defaults to native grouped and accepts explicit modes', () => {
    expect(resolveToolRunPresentation([])).toBe('grouped')
    expect(resolveToolRunPresentation([pref('p', 'individual')])).toBe('individual')
    expect(resolveToolRunPresentation([pref('p', 'grouped')])).toBe('grouped')
  })
  it('uses finite ascending order, then ordinal namespaced id regardless of registration order', () => {
    const a = pref('alpha:p', 'individual'),
      z = pref('zeta:p', 'grouped')

    expect(resolveToolRunPresentation([z, a])).toBe('individual')
    expect(resolveToolRunPresentation([a, z])).toBe('individual')
    expect(resolveToolRunPresentation([a, pref('z', 'grouped', -1)])).toBe('grouped')
  })
  it.each([
    undefined,
    null,
    {},
    { apiVersion: 0, mode: 'individual' },
    { apiVersion: 2, mode: 'individual' },
    { apiVersion: '1', mode: 'individual' },
    { apiVersion: 1, mode: 'flat' },
    {
      get mode() {
        throw new Error('bad payload')
      }
    }
  ])('ignores malformed/incompatible payload %#', data => {
    expect(resolveToolRunPresentation([{ id: 'bad', area: 'chat.toolRun.presentation', data }])).toBe('grouped')
  })
  it('ignores nonfinite order', () => {
    expect(resolveToolRunPresentation([pref('p', 'individual', NaN), pref('q', 'individual', Infinity)])).toBe(
      'grouped'
    )
  })
})
