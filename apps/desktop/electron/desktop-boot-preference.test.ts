import assert from 'node:assert/strict'
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { test } from 'vitest'

import { readDesktopBootPreference, writeDesktopProfile } from './desktop-boot-preference'

test('profile writes preserve other-owned fields and never adopt a broken file', (): void => {
  const directory: string = mkdtempSync(path.join(os.tmpdir(), 'hermes-boot-preference-'))
  const preference: string = path.join(directory, 'active-profile.json')

  try {
    writeDesktopProfile(preference, 'research')
    const stored = readDesktopBootPreference(preference)

    assert.equal(stored?.profile, 'research')
    writeFileSync(preference, '{broken', 'utf8')
    assert.throws(() => writeDesktopProfile(preference, 'preview'), /preference/)
    assert.equal(readFileSync(preference, 'utf8'), '{broken')
  } finally {
    rmSync(directory, { recursive: true, force: true })
  }
})
