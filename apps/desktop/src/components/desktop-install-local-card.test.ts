import assert from 'node:assert/strict'

import { test } from 'vitest'

import { localCardPresentation } from './desktop-install-local-card'

test('none is the installer offer with the install-to footer', () => {
  assert.deepEqual(localCardPresentation('none'), {
    title: 'installLocalTitle',
    desc: 'installLocalDesc',
    showInstallTo: true
  })
})

test('installed uses the existing-runtime copy and hides the install-to footer', () => {
  assert.deepEqual(localCardPresentation('installed'), {
    title: 'useLocalTitle',
    desc: 'useLocalDesc',
    showInstallTo: false
  })
})

test('bundled uses the bundled flavor of the existing-runtime copy', () => {
  assert.deepEqual(localCardPresentation('bundled'), {
    title: 'useLocalTitle',
    desc: 'bundledLocalDesc',
    showInstallTo: false
  })
})

test('an absent local field falls back to the installer offer (old backends)', () => {
  assert.equal(localCardPresentation(undefined).title, 'installLocalTitle')
})
