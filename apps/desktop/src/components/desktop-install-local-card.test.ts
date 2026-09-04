import assert from 'node:assert/strict'

import { test } from 'vitest'

import { localCardPresentation } from './desktop-install-local-card'

test('none is the installer offer with the install-to footer', () => {
  assert.deepEqual(localCardPresentation('none'), {
    title: 'installLocalTitle',
    desc: 'installLocalDesc',
    disabled: false,
    showInstallTo: true
  })
})

test('installed uses the existing-runtime copy and hides the install-to footer', () => {
  assert.deepEqual(localCardPresentation('installed'), {
    title: 'useLocalTitle',
    desc: 'useLocalDesc',
    disabled: false,
    showInstallTo: false
  })
})

test('bundled uses the bundled flavor of the existing-runtime copy', () => {
  assert.deepEqual(localCardPresentation('bundled'), {
    title: 'useLocalTitle',
    desc: 'bundledLocalDesc',
    disabled: false,
    showInstallTo: false
  })
})

test('bundled-damaged is disabled and never shows the install-to footer', () => {
  assert.deepEqual(localCardPresentation('bundled-damaged'), {
    title: 'bundledDamagedTitle',
    desc: 'bundledDamagedDesc',
    disabled: true,
    showInstallTo: false
  })
})

test('an absent local field falls back to the installer offer (old backends)', () => {
  assert.equal(localCardPresentation(undefined).title, 'installLocalTitle')
  assert.equal(localCardPresentation(undefined).disabled, false)
})
