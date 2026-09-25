import assert from 'node:assert/strict'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { test } from 'vitest'

import { localSkinHome, readLocalDisplaySkin, readLocalSkinPayload } from './local-skin'

function makeHome() {
  return fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-local-skin-'))
}

test('reads a configured custom skin without a live gateway', () => {
  const home = makeHome()

  try {
    fs.mkdirSync(path.join(home, 'skins'))
    fs.writeFileSync(path.join(home, 'config.yaml'), 'display:\n  skin: neon\n')
    fs.writeFileSync(
      path.join(home, 'skins', 'neon.yaml'),
      'name: neon\ndescription: neon after dark\ncolors:\n  background: "#101020"\n  ui_text: "#eeeeee"\n  ui_accent: "#ff33aa"\n'
    )

    const skin = readLocalDisplaySkin(home, null)

    assert.equal(skin?.name, 'neon')
    assert.equal(skin?.description, 'neon after dark')
    assert.equal(skin?.colors?.background, '#101020')
    assert.equal(skin?.colors?.ui_text, '#eeeeee')
    assert.equal(skin?.colors?.ui_accent, '#ff33aa')
    assert.equal(skin?.colors?.status_bar_bg, '#1a1a2e')
  } finally {
    fs.rmSync(home, { recursive: true, force: true })
  }
})

test('reads the active named profile instead of the root home', () => {
  const home = makeHome()
  const profileHome = localSkinHome(home, 'research')

  try {
    fs.mkdirSync(path.join(home, 'skins'))
    fs.mkdirSync(path.join(profileHome, 'skins'), { recursive: true })
    fs.writeFileSync(path.join(home, 'config.yaml'), 'display:\n  skin: root\n')
    fs.writeFileSync(path.join(profileHome, 'config.yaml'), 'display:\n  skin: profile\n')
    fs.writeFileSync(path.join(profileHome, 'skins', 'profile.yaml'), 'name: profile\ncolors:\n  ui_accent: "#00ffff"\n')

    const skin = readLocalDisplaySkin(home, 'research')

    assert.equal(skin?.name, 'profile')
    assert.equal(skin?.colors?.ui_accent, '#00ffff')
    assert.equal(skin?.colors?.status_bar_bg, '#1a1a2e')
  } finally {
    fs.rmSync(home, { recursive: true, force: true })
  }
})

test('uses a routed session or hud profile before any gateway can connect', () => {
  const home = makeHome()
  const profileHome = localSkinHome(home, 'research')

  try {
    fs.mkdirSync(path.join(profileHome, 'skins'), { recursive: true })
    fs.writeFileSync(path.join(profileHome, 'config.yaml'), 'display:\n  skin: research\n')
    fs.writeFileSync(path.join(profileHome, 'skins', 'research.yaml'), 'name: research\ncolors:\n  ui_accent: "#00ffff"\n')

    const payload = readLocalSkinPayload(home, 'research', 'default')

    assert.equal(payload?.profile, 'research')
    assert.equal(payload?.skin.name, 'research')
    assert.equal(payload?.skin.colors?.ui_accent, '#00ffff')
  } finally {
    fs.rmSync(home, { recursive: true, force: true })
  }
})

test('returns built-in names even though they have no local skin file', () => {
  const home = makeHome()

  try {
    fs.writeFileSync(path.join(home, 'config.yaml'), 'display: { skin: mono }\n')
    assert.deepEqual(readLocalDisplaySkin(home, null), { name: 'mono' })
  } finally {
    fs.rmSync(home, { recursive: true, force: true })
  }
})

test('fails closed for malformed config and traversal-like skin names', () => {
  const home = makeHome()

  try {
    fs.writeFileSync(path.join(home, 'config.yaml'), 'display: [not, a, mapping]\n')
    assert.equal(readLocalDisplaySkin(home, null), null)

    fs.writeFileSync(path.join(home, 'config.yaml'), 'display:\n  skin: ../../config\n')
    assert.equal(readLocalDisplaySkin(home, null), null)
  } finally {
    fs.rmSync(home, { recursive: true, force: true })
  }
})
