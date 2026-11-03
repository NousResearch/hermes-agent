import assert from 'node:assert/strict'

import { test } from 'vitest'

import { isAuthWall, resolveLinkTitle } from './link-title-wall'

// A published Doc answers the cookieless curl tier with its own <title>, so these
// links must stay fetchable — the wall is proven at fetch time, not from the host.
function tier1(payload: { authWall?: boolean; title?: string } = {}) {
  return async () => ({ authWall: payload.authWall ?? false, title: payload.title ?? '' })
}

test('a proven sign-in wall never escalates to the hidden renderer', async () => {
  let rendererCalls = 0

  const title = await resolveLinkTitle({
    curl: tier1({ authWall: true, title: '' }),
    renderer: async () => {
      rendererCalls += 1

      return 'Google Drive: Sign-in'
    },
    url: 'https://docs.google.com/document/d/1ExAmPlEdOcId0000000000000000000000/edit'
  })

  assert.equal(title, '')
  assert.equal(rendererCalls, 0)

  // The three measured shapes of the wall are all proven from the curl tier:
  // arrival URL (Apps Script → www.google.com/a/<domain>/ServiceLogin), sign-in
  // title (Drive → accounts.google.com), and markup (a Doc that stays on its host).
  assert.equal(
    isAuthWall({ body: '', effectiveUrl: 'https://www.google.com/a/x.org/ServiceLogin?c=1', title: '' }),
    true
  )
  assert.equal(
    isAuthWall({
      body: '',
      effectiveUrl: 'https://accounts.google.com/v3/signin/identifier',
      title: 'Google Drive: Sign-in'
    }),
    true
  )
  assert.equal(
    isAuthWall({
      body: '<a href="https://accounts.google.com/ServiceLogin">',
      effectiveUrl: 'https://docs.google.com/document/d/1/edit',
      title: ''
    }),
    true
  )
  assert.equal(
    isAuthWall({
      body: '<title>Q3 plan</title>',
      effectiveUrl: 'https://docs.google.com/document/d/1/pub',
      title: 'Q3 plan'
    }),
    false
  )
})

test('an ordinary title-less page still escalates to the hidden renderer', async () => {
  // The tier-2 behaviour that must survive: a JS-rendered page curl can't read
  // gets its title from the renderer.
  let rendererCalls = 0

  const title = await resolveLinkTitle({
    curl: tier1(),
    renderer: async () => {
      rendererCalls += 1

      return 'Lab AI service — guides'
    },
    url: 'https://example.com/guides'
  })

  assert.equal(title, 'Lab AI service — guides')
  assert.equal(rendererCalls, 1)
})
