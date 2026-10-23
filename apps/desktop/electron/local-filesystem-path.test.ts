import assert from 'node:assert/strict'

import { test } from 'vitest'

import { absolutizeProtocolRelativeUrl, looksLikeLocalFilesystemPath } from './local-filesystem-path'

test('recognizes every bare local filesystem path shape', () => {
  for (const value of [
    '/tmp/report.pdf',
    '///tmp/foo',
    '~/logs/desktop.log',
    'C:\\Users\\x\\a.md',
    'C:/Work/report.html',
    '\\\\server\\share\\a.md'
  ]) {
    assert.equal(looksLikeLocalFilesystemPath(value), true, `expected ${value} to be a local path`)
  }
})

test('does not claim URLs, relative names, or scheme-ful input', () => {
  for (const value of [
    '',
    'hermes-support-slack.png',
    'https://example.com',
    'file:///tmp/report.pdf',
    'mailto:a@b.c',
    '//cdn.example.com/img.png'
  ]) {
    assert.equal(looksLikeLocalFilesystemPath(value), false, `expected ${value} NOT to be a local path`)
  }
})

test('absolutizes protocol-relative URLs as https and leaves everything else alone', () => {
  assert.equal(absolutizeProtocolRelativeUrl('//cdn.example.com/img.png'), 'https://cdn.example.com/img.png')
  assert.equal(absolutizeProtocolRelativeUrl('///tmp/foo'), '///tmp/foo')
  assert.equal(absolutizeProtocolRelativeUrl('https://example.com'), 'https://example.com')
})
