import assert from 'node:assert/strict'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { test } from 'vitest'

import {
  COMPOSER_SEND_DEFAULT_MODE,
  DOUBLE_ENTER_DEFAULT_MS,
  DOUBLE_ENTER_MAX_MS,
  DOUBLE_ENTER_MIN_MS,
  SEND_GRACE_DEFAULT_MS,
  SEND_GRACE_DEFAULT_SCOPE,
  SEND_GRACE_MIN_MS,
  TYPING_IDLE_DEFAULT_MS,
  TYPING_IDLE_MAX_MS
} from '../../shared/src/composer-send'

import { readComposerSendPrefs, writeComposerSendPrefs } from './composer-send-store'

function withTempDir(run: (directory: string) => void) {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-composer-send-'))

  try {
    run(directory)
  } finally {
    fs.rmSync(directory, { recursive: true, force: true })
  }
}

const configFile = (directory: string) => path.join(directory, 'composer-send.json')

/** The whole shipped default set: a field added to the prefs without a clamp
 *  and a default shows up here as a shape change. */
const DEFAULTS = {
  mode: COMPOSER_SEND_DEFAULT_MODE,
  doubleEnterMs: DOUBLE_ENTER_DEFAULT_MS,
  typingIdleMs: TYPING_IDLE_DEFAULT_MS,
  sendGrace: SEND_GRACE_DEFAULT_SCOPE,
  sendGraceMs: SEND_GRACE_DEFAULT_MS
}

test('readComposerSendPrefs falls back to the defaults when the file is missing', () => {
  withTempDir(directory => {
    assert.deepEqual(readComposerSendPrefs(configFile(directory)), DEFAULTS)
  })
})

test('readComposerSendPrefs accepts a hand-edited file', () => {
  withTempDir(directory => {
    fs.writeFileSync(configFile(directory), JSON.stringify({ mode: 'double-enter', doubleEnterMs: 640 }), 'utf8')

    assert.deepEqual(readComposerSendPrefs(configFile(directory)), {
      ...DEFAULTS,
      mode: 'double-enter',
      doubleEnterMs: 640
    })
  })
})

test('readComposerSendPrefs clamps an out-of-range window and rejects an unknown mode', () => {
  withTempDir(directory => {
    fs.writeFileSync(configFile(directory), JSON.stringify({ mode: 'triple-enter', doubleEnterMs: 99_999 }), 'utf8')

    assert.deepEqual(readComposerSendPrefs(configFile(directory)), {
      ...DEFAULTS,
      doubleEnterMs: DOUBLE_ENTER_MAX_MS
    })

    fs.writeFileSync(configFile(directory), JSON.stringify({ mode: 'mod-enter', doubleEnterMs: 1 }), 'utf8')

    assert.deepEqual(readComposerSendPrefs(configFile(directory)), {
      ...DEFAULTS,
      mode: 'mod-enter',
      doubleEnterMs: DOUBLE_ENTER_MIN_MS
    })
  })
})

test('readComposerSendPrefs survives a truncated file', () => {
  withTempDir(directory => {
    fs.writeFileSync(configFile(directory), '{ "mode": "double-', 'utf8')

    assert.deepEqual(readComposerSendPrefs(configFile(directory)), DEFAULTS)
  })
})

test('writeComposerSendPrefs creates the parent directory and round-trips', () => {
  withTempDir(directory => {
    const nested = path.join(directory, 'does', 'not', 'exist', 'composer-send.json')
    const written = writeComposerSendPrefs({ mode: 'double-enter', doubleEnterMs: 250 }, nested)

    assert.deepEqual(written, { ...DEFAULTS, mode: 'double-enter', doubleEnterMs: 250 })
    assert.deepEqual(readComposerSendPrefs(nested), written)
    assert.deepEqual(JSON.parse(fs.readFileSync(nested, 'utf8')), written)
  })
})

test('writeComposerSendPrefs clamps what it is handed', () => {
  withTempDir(directory => {
    const written = writeComposerSendPrefs({ mode: 'nope', doubleEnterMs: 'wat' }, configFile(directory))

    assert.deepEqual(written, DEFAULTS)
  })
})

test('readComposerSendPrefs clamps the pause knobs and rejects an unknown grace scope', () => {
  withTempDir(directory => {
    fs.writeFileSync(
      configFile(directory),
      JSON.stringify({ mode: 'pause', sendGrace: 'sometimes', sendGraceMs: -5, typingIdleMs: 99_999 }),
      'utf8'
    )

    assert.deepEqual(readComposerSendPrefs(configFile(directory)), {
      ...DEFAULTS,
      mode: 'pause',
      sendGrace: SEND_GRACE_DEFAULT_SCOPE,
      // -5 clamps UP to the minimum (0 = no hold), not down to the maximum.
      sendGraceMs: SEND_GRACE_MIN_MS,
      typingIdleMs: TYPING_IDLE_MAX_MS
    })
  })
})

test('writeComposerSendPrefs keeps a chosen grace scope', () => {
  withTempDir(directory => {
    const written = writeComposerSendPrefs({ ...DEFAULTS, sendGrace: 'all', sendGraceMs: 0 }, configFile(directory))

    assert.deepEqual(written, { ...DEFAULTS, sendGrace: 'all', sendGraceMs: 0 })
    assert.deepEqual(readComposerSendPrefs(configFile(directory)), written)
  })
})
