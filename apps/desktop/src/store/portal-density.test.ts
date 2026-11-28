import { afterEach, expect, it, vi } from 'vitest'

const KEY = 'hermes.desktop.portalDensity.v1'

/** One page load: the store reads what this browser kept. */
async function load(stored?: string) {
  window.localStorage.clear()

  if (stored !== undefined) {
    window.localStorage.setItem(KEY, stored)
  }

  vi.resetModules()

  return import('./portal-density')
}

const root = () => window.document.createElement('html')

afterEach(() => window.localStorage.clear())

it('marks nothing at the default, so the redesign is spaced as it always was', async () => {
  const { $portalDensity, installPortalDensityAttribute } = await load()
  const html = root()
  installPortalDensityAttribute(html)

  expect($portalDensity.get()).toBe('comfortable')
  expect(html.attributes).toHaveLength(0)
})

it('marks the document compact when compact is chosen, and unmarks it on the way back', async () => {
  const { installPortalDensityAttribute, setPortalDensity } = await load()
  const html = root()
  installPortalDensityAttribute(html)

  setPortalDensity('compact')
  expect(html.dataset.portalDensity).toBe('compact')
  expect(window.localStorage.getItem(KEY)).toBe('compact')

  setPortalDensity('comfortable')
  expect(html.attributes).toHaveLength(0)
  expect(window.localStorage.getItem(KEY)).toBeNull()
})

it('reads a stored value it does not know as comfortable', async () => {
  const { $portalDensity } = await load('cosy')

  expect($portalDensity.get()).toBe('comfortable')
})
