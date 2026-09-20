import assert from 'node:assert/strict'

import { test } from 'vitest'

import { isAuthWallBody, needsRendererFallback, resolveLinkTitle } from './link-title-wall'

const SIGNIN_IDENTIFIER_BODY =
  '<html><head><title>Google Drive: Sign-in</title></head><body><input id="identifierId"></body></html>'

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
})

test('a title-less page on a host that can only answer with a wall skips the renderer', async () => {
  // The measured domain-restricted Apps Script shape: a 2 KB body carrying none
  // of the markers the identifier page does, so the host is what holds.
  let rendererCalls = 0

  const title = await resolveLinkTitle({
    curl: tier1({ authWall: false, title: '' }),
    renderer: async () => {
      rendererCalls += 1

      return ''
    },
    url: 'https://script.google.com/a/example.edu/macros/s/abc/exec'
  })

  assert.equal(title, '')
  assert.equal(rendererCalls, 0)
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

test('a usable tier-1 title never escalates', async () => {
  let rendererCalls = 0

  const title = await resolveLinkTitle({
    curl: tier1({ title: 'Which model to use' }),
    renderer: async () => {
      rendererCalls += 1

      return 'Something else'
    },
    url: 'https://docs.google.com/document/d/abc123/pub'
  })

  assert.equal(title, 'Which model to use')
  assert.equal(rendererCalls, 0)
})

test('an error/captcha title is not usable and escalates on an ordinary host', async () => {
  // usableTitle drops the captcha title, so it reads as "nothing found" — the
  // renderer tier is still allowed to try (that is its documented purpose).
  let rendererCalls = 0

  const title = await resolveLinkTitle({
    curl: tier1({ title: 'Just a moment...' }),
    renderer: async () => {
      rendererCalls += 1

      return 'Real page title'
    },
    url: 'https://example.com/cloudflare-protected'
  })

  assert.equal(title, 'Real page title')
  assert.equal(rendererCalls, 1)

  // On a wall host the same empty answer must not escalate.
  assert.equal(needsRendererFallback({ authWall: false, title: '', url: 'https://drive.google.com/x' }), false)
  assert.equal(needsRendererFallback({ authWall: false, title: '', url: 'https://example.com/x' }), true)
})

test('the sign-in wall is recognised in the body curl already returned', () => {
  assert.equal(isAuthWallBody(SIGNIN_IDENTIFIER_BODY), true)
  assert.equal(isAuthWallBody('<a href="https://accounts.google.com/ServiceLogin?continue=x">Sign in</a>'), true)
  assert.equal(isAuthWallBody('<html><head><title>Which model to use</title></head></html>'), false)
  assert.equal(isAuthWallBody(''), false)
})

test('needsRendererFallback only refuses the hosts that can only answer with a wall', () => {
  for (const url of [
    'https://accounts.google.com/signin',
    'https://docs.google.com/document/d/abc/edit',
    'https://drive.google.com/drive/folders/abc',
    'https://sheets.google.com/spreadsheets/d/abc',
    'https://slides.google.com/presentation/d/abc',
    'https://sites.google.com/example.edu/x/',
    'https://script.google.com/a/example.edu/macros/s/abc/exec',
    'https://console.cloud.google.com/apis/library/docs.googleapis.com'
  ]) {
    assert.equal(needsRendererFallback({ authWall: false, title: '', url }), false, url)
  }

  // A lookalike host is not a Google host, and a URL that won't parse is not a refusal.
  for (const url of ['https://example.com/docs', 'https://github.com/NousResearch/hermes-agent', 'not a url']) {
    assert.equal(needsRendererFallback({ authWall: false, title: '', url }), true, url)
  }
})
