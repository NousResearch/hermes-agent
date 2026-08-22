import assert from 'node:assert/strict'

import { test } from 'vitest'

import { runningAppBundle } from './running-app-bundle'

const missing = () => false
const darwin = 'darwin' as const

test('runningAppBundle: canonical install is used as-is', () => {
  assert.equal(
    runningAppBundle('/Applications/Hermes.app/Contents/MacOS/Hermes', () => true, darwin),
    '/Applications/Hermes.app'
  )
})

test('runningAppBundle: migration backup relaunches into an existing canonical install', () => {
  const exists = (p: string) => p === '/Applications/Hermes.app'
  assert.equal(
    runningAppBundle(
      '/Users/pool/.hermes.preclone-20260715-132112/hermes-agent/apps/desktop/release/mac-arm64/Hermes.app/Contents/MacOS/Hermes',
      exists,
      darwin
    ),
    '/Applications/Hermes.app'
  )
})

test('runningAppBundle: mounted image relaunches into an existing canonical install', () => {
  const exists = (p: string) => p === '/Applications/Hermes.app'
  assert.equal(
    runningAppBundle('/Volumes/Hermes/Hermes.app/Contents/MacOS/Hermes', exists, darwin),
    '/Applications/Hermes.app'
  )
})

test('runningAppBundle: arbitrary alternate install is not silently retargeted', () => {
  const exists = (p: string) => p === '/Applications/Hermes.app'
  assert.equal(
    runningAppBundle('/Users/pool/Applications/Hermes.app/Contents/MacOS/Hermes', exists, darwin),
    '/Users/pool/Applications/Hermes.app'
  )
})

test('runningAppBundle: canonical name follows the running bundle name', () => {
  const exists = (p: string) => p === '/Applications/Hermes Beta.app'
  assert.equal(
    runningAppBundle('/Volumes/Hermes Beta/Hermes Beta.app/Contents/MacOS/Hermes Beta', exists, darwin),
    '/Applications/Hermes Beta.app'
  )
})

test('runningAppBundle: transient location falls back when no canonical install exists', () => {
  assert.equal(
    runningAppBundle('/Volumes/Hermes/Hermes.app/Contents/MacOS/Hermes', missing, darwin),
    '/Volumes/Hermes/Hermes.app'
  )
})

test('runningAppBundle: non-mac returns null', () => {
  assert.equal(runningAppBundle('/usr/bin/electron', missing, 'linux'), null)
})
