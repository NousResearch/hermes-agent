import assert from 'node:assert/strict'

import { test } from 'vitest'

import { CLERK_DEV_RENDERER_SCHEME, CLERK_RENDERER_HOST, CLERK_RENDERER_SCHEME, clerkOAuthRenderer } from './clerk-config'

test('packaged builds use the unitalk-app renderer origin', () => {
  assert.deepEqual(clerkOAuthRenderer({ devServer: false, platform: 'win32', registerProtocol: true }), {
    host: CLERK_RENDERER_HOST,
    scheme: CLERK_RENDERER_SCHEME
  })
  assert.deepEqual(clerkOAuthRenderer({ devServer: false, platform: 'darwin', registerProtocol: true }), {
    host: CLERK_RENDERER_HOST,
    scheme: CLERK_RENDERER_SCHEME
  })
})

test('Windows development uses a separate scheme so it never takes over the installed app', () => {
  assert.deepEqual(clerkOAuthRenderer({ devServer: true, platform: 'win32', registerProtocol: true }), {
    host: CLERK_RENDERER_HOST,
    scheme: CLERK_DEV_RENDERER_SCHEME
  })
  assert.notEqual(CLERK_DEV_RENDERER_SCHEME, CLERK_RENDERER_SCHEME)
})

test('macOS and Linux development stay disabled until the app is packaged', () => {
  assert.equal(clerkOAuthRenderer({ devServer: true, platform: 'darwin', registerProtocol: true }), undefined)
  assert.equal(clerkOAuthRenderer({ devServer: true, platform: 'linux', registerProtocol: true }), undefined)
})

test('skipping OS protocol registration disables every OAuth redirect', () => {
  assert.equal(clerkOAuthRenderer({ devServer: false, platform: 'win32', registerProtocol: false }), undefined)
  assert.equal(clerkOAuthRenderer({ devServer: true, platform: 'win32', registerProtocol: false }), undefined)
})
