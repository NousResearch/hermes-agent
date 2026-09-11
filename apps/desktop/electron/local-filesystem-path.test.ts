import assert from 'node:assert/strict'

import { test } from 'vitest'

import { absolutizeProtocolRelativeUrl, looksLikeLocalFilesystemPath } from './local-filesystem-path'

test('looksLikeLocalFilesystemPath accepts POSIX, home, drive, and UNC paths', () => {
  assert.equal(looksLikeLocalFilesystemPath('/tmp/demo.mp4'), true)
  assert.equal(looksLikeLocalFilesystemPath('~/Movies/clip.mp4'), true)
  assert.equal(looksLikeLocalFilesystemPath('D:/payload.json'), true)
  assert.equal(looksLikeLocalFilesystemPath('C:\\Users\\a\\file.png'), true)
  assert.equal(looksLikeLocalFilesystemPath('\\\\fileserver\\share\\a.png'), true)
  assert.equal(looksLikeLocalFilesystemPath('///tmp/triple-slash.png'), true)
})

test('looksLikeLocalFilesystemPath does not treat protocol-relative URLs as paths', () => {
  assert.equal(looksLikeLocalFilesystemPath('//cdn.example.com/img.png'), false)
  assert.equal(looksLikeLocalFilesystemPath('//cdn.example.com/img.png?x=1'), false)
})

test('absolutizeProtocolRelativeUrl prefixes https on //host/…', () => {
  assert.equal(
    absolutizeProtocolRelativeUrl('//cdn.example.com/img.png'),
    'https://cdn.example.com/img.png'
  )
  assert.equal(
    absolutizeProtocolRelativeUrl('//cdn.example.com/img.png?x=1#h'),
    'https://cdn.example.com/img.png?x=1#h'
  )
})

test('absolutizeProtocolRelativeUrl leaves real paths and absolute URLs alone', () => {
  assert.equal(absolutizeProtocolRelativeUrl('/tmp/demo.mp4'), '/tmp/demo.mp4')
  assert.equal(absolutizeProtocolRelativeUrl('///tmp/foo.png'), '///tmp/foo.png')
  assert.equal(absolutizeProtocolRelativeUrl('https://example.com/a'), 'https://example.com/a')
  assert.equal(absolutizeProtocolRelativeUrl('~/clip.mp4'), '~/clip.mp4')
})
