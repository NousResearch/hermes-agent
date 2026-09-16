import assert from 'node:assert/strict'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { test } from 'vitest'

import {
  DESKTOP_UPDATE_CONFIG_SCHEMA_VERSION,
  establishDesktopUpdateTrack,
  loadDesktopUpdateConfig,
  parseDesktopUpdateConfig
} from './desktop-update-config'

function withTempDir(run: (directory: string) => void) {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-update-config-'))

  try {
    run(directory)
  } finally {
    fs.rmSync(directory, { recursive: true, force: true })
  }
}

function readJson(filePath: string) {
  return JSON.parse(fs.readFileSync(filePath, 'utf8'))
}

test('fresh installations default to the release track and persist the inference', () =>
  withTempDir(directory => {
    const configPath = path.join(directory, 'updates.json')
    const config = loadDesktopUpdateConfig({ configPath, installationAlreadyExisted: false })

    assert.deepEqual(config, {
      schemaVersion: DESKTOP_UPDATE_CONFIG_SCHEMA_VERSION,
      track: 'release',
      trackSource: 'default',
      branch: 'main'
    })
    assert.deepEqual(readJson(configPath), config)
  }))

test('completed default release tracks become established without changing explicit tracks', () => {
  const inferred = {
    schemaVersion: DESKTOP_UPDATE_CONFIG_SCHEMA_VERSION,
    track: 'release',
    trackSource: 'default',
    branch: 'main'
  } as const

  const explicit = { ...inferred, trackSource: 'explicit' as const }

  assert.deepEqual(establishDesktopUpdateTrack(inferred), { ...inferred, trackSource: 'established' })
  assert.equal(establishDesktopUpdateTrack(explicit), explicit)
})

test('pre-existing installations remain on the main track', () =>
  withTempDir(directory => {
    const configPath = path.join(directory, 'updates.json')
    const config = loadDesktopUpdateConfig({ configPath, installationAlreadyExisted: true })

    assert.equal(config.track, 'main')
    assert.equal(config.trackSource, 'migration')
    assert.deepEqual(readJson(configPath), config)
  }))

test('persisted track wins over installation age', () =>
  withTempDir(directory => {
    const configPath = path.join(directory, 'updates.json')
    fs.writeFileSync(configPath, JSON.stringify({ schemaVersion: 2, track: 'release', branch: 'beta' }))

    const config = loadDesktopUpdateConfig({ configPath, installationAlreadyExisted: true })

    assert.deepEqual(config, { schemaVersion: 2, track: 'release', trackSource: 'explicit', branch: 'beta' })
  }))

test('legacy branch-only configuration migrates to main without losing the branch', () =>
  withTempDir(directory => {
    const configPath = path.join(directory, 'updates.json')
    fs.writeFileSync(configPath, JSON.stringify({ branch: 'bb/gui' }))

    const config = loadDesktopUpdateConfig({ configPath, installationAlreadyExisted: false })

    assert.deepEqual(config, { schemaVersion: 2, track: 'main', trackSource: 'migration', branch: 'bb/gui' })
    assert.deepEqual(readJson(configPath), config)
  }))

test('malformed configuration falls back according to installation age', () =>
  withTempDir(directory => {
    const oldPath = path.join(directory, 'old.json')
    const freshPath = path.join(directory, 'fresh.json')
    fs.writeFileSync(oldPath, '{')
    fs.writeFileSync(freshPath, JSON.stringify({ track: 'nightly' }))

    const oldConfig = loadDesktopUpdateConfig({ configPath: oldPath, installationAlreadyExisted: true })
    const freshConfig = loadDesktopUpdateConfig({ configPath: freshPath, installationAlreadyExisted: false })

    assert.equal(oldConfig.track, 'main')
    assert.equal(oldConfig.trackSource, 'migration')
    assert.equal(freshConfig.track, 'release')
    assert.equal(freshConfig.trackSource, 'default')
  }))

test('first and second launch keep the same inferred release track', () =>
  withTempDir(directory => {
    const configPath = path.join(directory, 'updates.json')
    const first = loadDesktopUpdateConfig({ configPath, installationAlreadyExisted: false })
    const second = loadDesktopUpdateConfig({ configPath, installationAlreadyExisted: true })

    assert.equal(first.track, 'release')
    assert.equal(first.trackSource, 'default')
    assert.equal(second.track, 'release')
    assert.equal(second.trackSource, 'default')
  }))

test('parseDesktopUpdateConfig normalises supported records only', () => {
  assert.equal(parseDesktopUpdateConfig('{'), null)
  assert.equal(parseDesktopUpdateConfig(JSON.stringify({ track: 'fast' })), null)
  assert.deepEqual(parseDesktopUpdateConfig(JSON.stringify({ track: 'main' })), {
    schemaVersion: 2,
    track: 'main',
    trackSource: 'explicit',
    branch: 'main'
  })
  assert.equal(
    parseDesktopUpdateConfig(JSON.stringify({ track: 'release', trackSource: 'unknown' }))?.trackSource,
    'explicit'
  )
  assert.equal(
    parseDesktopUpdateConfig(JSON.stringify({ track: 'release', trackSource: 'established' }))?.trackSource,
    'established'
  )
})
