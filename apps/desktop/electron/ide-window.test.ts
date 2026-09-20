import assert from 'node:assert/strict'

import { test } from 'vitest'

import { buildIdeWindowUrl } from './ide-window'

test('buildIdeWindowUrl puts win=ide before the hash route (dev server)', () => {
  const url = buildIdeWindowUrl({ devServer: 'http://localhost:5174' })

  assert.equal(url, 'http://localhost:5174/?win=ide#/')
  assert.ok(url.indexOf('?win=ide') < url.indexOf('#'))
})

test('buildIdeWindowUrl carries the opener profile and connection before the hash', () => {
  const url = buildIdeWindowUrl({ connectionId: '', devServer: 'http://localhost:5174', profile: 'work' })

  assert.equal(url, 'http://localhost:5174/?win=ide&profile=work&connectionId=#/')
  assert.ok(url.indexOf('profile=work') < url.indexOf('#'))
})

test('buildIdeWindowUrl carries a remote connection id and encodes it', () => {
  const url = buildIdeWindowUrl({
    connectionId: 'remote one',
    devServer: 'http://localhost:5174',
    profile: 'work'
  })

  assert.equal(url, 'http://localhost:5174/?win=ide&profile=work&connectionId=remote%20one#/')
})

test('buildIdeWindowUrl seeds the workspace cwd and encodes it', () => {
  const url = buildIdeWindowUrl({
    cwd: 'D:\\My apps\\COAI',
    devServer: 'http://localhost:5174',
    profile: 'work'
  })

  assert.ok(url.indexOf('cwd=') < url.indexOf('#'))
  assert.equal(
    url,
    `http://localhost:5174/?win=ide&profile=work&connectionId=&cwd=${encodeURIComponent('D:\\My apps\\COAI')}#/`
  )
})

test('buildIdeWindowUrl omits an empty profile — no override, plain boot', () => {
  assert.equal(
    buildIdeWindowUrl({ devServer: 'http://localhost:5174', profile: '  ' }),
    'http://localhost:5174/?win=ide#/'
  )
})

test('buildIdeWindowUrl keeps cwd even without a profile carry', () => {
  assert.equal(
    buildIdeWindowUrl({ cwd: '/home/me/code', devServer: 'http://localhost:5174' }),
    'http://localhost:5174/?win=ide&cwd=%2Fhome%2Fme%2Fcode#/'
  )
})

test('buildIdeWindowUrl avoids a double slash when the dev server has a trailing slash', () => {
  const url = buildIdeWindowUrl({ devServer: 'http://localhost:5174/', profile: 'work' })

  assert.equal(url, 'http://localhost:5174/?win=ide&profile=work&connectionId=#/')
})

test('buildIdeWindowUrl builds a packaged file URL with the flags before the hash', () => {
  const url = buildIdeWindowUrl({ profile: 'coder', rendererIndexPath: '/opt/app/index.html' })

  assert.match(url, /^file:\/\/.*index\.html\?win=ide&profile=coder&connectionId=#\/$/)
})
