import { readFileSync } from 'node:fs'

import { expect, it } from 'vitest'

import registry from '../locale-registry.generated.js'

it('ships the canonical locale identities and aliases inside the shared package', () => {
  const canonical = JSON.parse(readFileSync(new URL('../../../../locales/registry.json', import.meta.url), 'utf8'))

  expect(registry).toEqual(canonical)
})
