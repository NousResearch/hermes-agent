import assert from 'node:assert/strict'

import { test } from 'vitest'

import { webImportIntent } from './pen-web-import-intent'

test('a URL beside a design word names the page to import', () => {
  assert.deepEqual(webImportIntent('copy https://www.stripe.com/pricing into a mockup'), {
    host: 'stripe.com',
    url: 'https://www.stripe.com/pricing'
  })
})

test('a bare URL without design intent is not an import', () => {
  assert.equal(webImportIntent('check https://example.com for me', 'https://example.com/'), null)
})

test('a clone verb about "this site" imports the page on screen, and only when there is one', () => {
  const text = 'i want to copy this site to a design file'

  assert.deepEqual(webImportIntent(text, 'https://example.com/'), { host: 'example.com' })
  assert.equal(webImportIntent(text), null)
  assert.equal(webImportIntent(text, 'about:blank'), null)
})

test('design talk with no page and no URL stays with the plain canvas pill', () => {
  assert.equal(webImportIntent('make me a wireframe for a dashboard', 'https://example.com/'), null)
})
