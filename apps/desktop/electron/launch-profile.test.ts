import assert from 'node:assert/strict'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { test } from 'vitest'

import { serveBackendArgs } from './backend-command'
import { createDesktopProfilePreferences } from './desktop-profile'
import { applyLaunchProfileOverride, parseLaunchProfile } from './launch-profile'

function withStoredProfile(profile: string, run: (target: string) => void) {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'launch-profile-'))
  const target = path.join(root, 'active-profile.json')
  const preferences = createDesktopProfilePreferences(target)

  try {
    preferences.remember(profile)
    preferences.setDefault({ connectionId: null, profile })
    run(target)
  } finally {
    fs.rmSync(root, { recursive: true, force: true })
  }
}

test('parses both --profile spellings and ignores a missing flag', () => {
  assert.equal(parseLaunchProfile(['Hermes.exe', '--profile', 'desktop']), 'desktop')
  assert.equal(parseLaunchProfile(['Hermes.exe', '--profile=desktop']), 'desktop')
  assert.equal(parseLaunchProfile(['open', '-a', 'Hermes', '--args', '--profile', 'work']), 'work')
  assert.equal(parseLaunchProfile(['Hermes.exe', '--local']), null)
  assert.equal(parseLaunchProfile(['Hermes.exe']), null)
})

test('normalizes a launch profile the same way the CLI does', () => {
  assert.equal(parseLaunchProfile(['Hermes.exe', '--profile', '  Desktop  ']), 'desktop')
  assert.equal(parseLaunchProfile(['Hermes.exe', '--profile=Work']), 'work')
  assert.equal(parseLaunchProfile(['Hermes.exe', '--profile', 'default']), 'default')
  assert.equal(parseLaunchProfile(['Hermes.exe', '--profile']), null)
  assert.equal(parseLaunchProfile(['Hermes.exe', '--profile=']), null)
  assert.equal(parseLaunchProfile(['Hermes.exe', '--profile', 'my profile']), null)
  assert.equal(parseLaunchProfile(['Hermes.exe', '--profile', '-desktop']), null)
})

test('a launch --profile is persisted before the backend reads active-profile.json', () => {
  withStoredProfile('stored', target => {
    const preferences = createDesktopProfilePreferences(target)

    const launched = applyLaunchProfileOverride(['Hermes.exe', '--profile', 'desktop'], name => {
      preferences.remember(name)
    })

    const restarted = createDesktopProfilePreferences(target)

    assert.equal(launched, 'desktop')
    assert.equal(restarted.readActive(), 'desktop')
    assert.deepEqual(restarted.getDefault(), { connectionId: null, profile: 'stored' })
    assert.deepEqual(serveBackendArgs(restarted.readActive() ?? undefined).slice(0, 2), ['--profile', 'desktop'])
  })
})

test('--profile=<name> persists the same profile the space spelling does', () => {
  withStoredProfile('stored', target => {
    const preferences = createDesktopProfilePreferences(target)

    applyLaunchProfileOverride(['Hermes.exe', '--ozone-platform=wayland', '--profile=desktop'], name => {
      preferences.remember(name)
    })

    assert.equal(createDesktopProfilePreferences(target).readActive(), 'desktop')
  })
})

test('a missing or invalid flag does not change the stored profile', () => {
  for (const argv of [
    ['Hermes.exe'],
    ['Hermes.exe', '--local'],
    ['Hermes.exe', '--profile'],
    ['Hermes.exe', '--profile', 'Not a name'],
    ['Hermes.exe', '--profile=../desktop']
  ]) {
    withStoredProfile('stored', target => {
      const before = fs.readFileSync(target, 'utf8')
      let persisted = false

      const launched = applyLaunchProfileOverride(argv, () => {
        persisted = true
      })

      assert.equal(launched, null)
      assert.equal(persisted, false)
      assert.equal(fs.readFileSync(target, 'utf8'), before)
    })
  }
})
