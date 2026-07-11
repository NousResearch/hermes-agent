/**
 * Tests for electron/render-cache-config.ts.
 *
 * Run with: node --test electron/render-cache-config.test.ts
 */

import assert from 'node:assert/strict'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import test from 'node:test'

import { parseRenderCacheEnabled, readRenderCacheEnabled } from './render-cache-config.ts'

test('default is ON: empty/absent/malformed configs enable the cache', () => {
  assert.equal(parseRenderCacheEnabled(''), true)
  assert.equal(parseRenderCacheEnabled('completely: unrelated\nkeys: here\n'), true)
  assert.equal(parseRenderCacheEnabled(':::: not yaml at all ::::'), true)
})

test('explicit false disables (the rollback lever)', () => {
  const yaml = ['desktop:', '  render_cache:', '    enabled: false', ''].join('\n')
  assert.equal(parseRenderCacheEnabled(yaml), false)
})

test('explicit true / other truthy values stay enabled', () => {
  for (const v of ['true', 'yes', 'on', '1']) {
    const yaml = ['desktop:', '  render_cache:', `    enabled: ${v}`, ''].join('\n')
    assert.equal(parseRenderCacheEnabled(yaml), true, v)
  }
})

test('falsy variants disable', () => {
  for (const v of ['false', 'no', 'off', '0', '"false"']) {
    const yaml = ['desktop:', '  render_cache:', `    enabled: ${v}`, ''].join('\n')
    assert.equal(parseRenderCacheEnabled(yaml), false, v)
  }
})

test('enabled under a DIFFERENT block does not count', () => {
  const yaml = [
    'other:',
    '  render_cache:',
    '    enabled: false',
    'desktop:',
    '  something_else: 1',
    ''
  ].join('\n')
  assert.equal(parseRenderCacheEnabled(yaml), true)
})

test('desktop block with other keys before render_cache still resolves', () => {
  const yaml = [
    'desktop:',
    '  theme: dark',
    '  render_cache:',
    '    max_files: 100',
    '    enabled: false',
    '  after: 1',
    ''
  ].join('\n')
  assert.equal(parseRenderCacheEnabled(yaml), false)
})

test('comments and blank lines are ignored', () => {
  const yaml = [
    'desktop:',
    '  # the rollback flag',
    '',
    '  render_cache:',
    '    # flip to false to disable cache paint',
    '    enabled: false  # off for debugging',
    ''
  ].join('\n')
  assert.equal(parseRenderCacheEnabled(yaml), false)
})

test('readRenderCacheEnabled reads HERMES_HOME/config.yaml and fails open', () => {
  const home = fs.mkdtempSync(path.join(os.tmpdir(), 'rc-config-test-'))
  // no config.yaml at all → ON
  assert.equal(readRenderCacheEnabled(home), true)
  fs.writeFileSync(
    path.join(home, 'config.yaml'),
    'desktop:\n  render_cache:\n    enabled: false\n'
  )
  assert.equal(readRenderCacheEnabled(home), false)
})
