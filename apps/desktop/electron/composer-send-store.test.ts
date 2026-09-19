import assert from 'node:assert/strict'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { test } from 'vitest'

import {
  DOUBLE_ENTER_DEFAULT_MS,
  DOUBLE_ENTER_MAX_MS,
  DOUBLE_ENTER_MIN_MS,
  HOLD_DEFAULT_MS,
  IDLE_SEND_DEFAULT_MS,
  SEND_GRACE_DEFAULT_MS,
  SEND_GRACE_DEFAULT_REASONS,
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
  doubleEnterMs: DOUBLE_ENTER_DEFAULT_MS,
  enterNewline: true,
  enterSends: true,
  holdMs: HOLD_DEFAULT_MS,
  idleSendMs: IDLE_SEND_DEFAULT_MS,
  sendOnDoubleTap: false,
  sendOnHold: false,
  sendOnIdle: false,
  sendOnPause: false,
  sendGraceFor: SEND_GRACE_DEFAULT_REASONS,
  sendGraceMs: SEND_GRACE_DEFAULT_MS,
  typingIdleMs: TYPING_IDLE_DEFAULT_MS
}

test('readComposerSendPrefs falls back to the defaults when the file is missing', () => {
  withTempDir(directory => {
    assert.deepEqual(readComposerSendPrefs(configFile(directory)), DEFAULTS)
  })
})

test('readComposerSendPrefs accepts a hand-edited file', () => {
  withTempDir(directory => {
    fs.writeFileSync(
      configFile(directory),
      JSON.stringify({ doubleEnterMs: 640, enterSends: false, sendOnDoubleTap: true }),
      'utf8'
    )

    assert.deepEqual(readComposerSendPrefs(configFile(directory)), {
      ...DEFAULTS,
      doubleEnterMs: 640,
      enterSends: false,
      sendOnDoubleTap: true
    })
  })
})

test('readComposerSendPrefs clamps an out-of-range window', () => {
  withTempDir(directory => {
    fs.writeFileSync(configFile(directory), JSON.stringify({ doubleEnterMs: 99_999 }), 'utf8')

    assert.deepEqual(readComposerSendPrefs(configFile(directory)), {
      ...DEFAULTS,
      doubleEnterMs: DOUBLE_ENTER_MAX_MS
    })

    fs.writeFileSync(configFile(directory), JSON.stringify({ enterSends: false, doubleEnterMs: 1 }), 'utf8')

    assert.deepEqual(readComposerSendPrefs(configFile(directory)), {
      ...DEFAULTS,
      doubleEnterMs: DOUBLE_ENTER_MIN_MS,
      enterSends: false
    })
  })
})

test('readComposerSendPrefs survives a truncated file', () => {
  withTempDir(directory => {
    fs.writeFileSync(configFile(directory), '{ "enterSends": fal', 'utf8')

    assert.deepEqual(readComposerSendPrefs(configFile(directory)), DEFAULTS)
  })
})

test('writeComposerSendPrefs creates the parent directory and round-trips', () => {
  withTempDir(directory => {
    const nested = path.join(directory, 'does', 'not', 'exist', 'composer-send.json')
    const written = writeComposerSendPrefs({ ...DEFAULTS, doubleEnterMs: 250, enterSends: false, sendOnHold: true }, nested)

    assert.deepEqual(written, { ...DEFAULTS, doubleEnterMs: 250, enterSends: false, sendOnHold: true })
    assert.deepEqual(readComposerSendPrefs(nested), written)
    assert.deepEqual(JSON.parse(fs.readFileSync(nested, 'utf8')), written)
  })
})

test('writeComposerSendPrefs clamps what it is handed', () => {
  withTempDir(directory => {
    const written = writeComposerSendPrefs({ doubleEnterMs: 'wat' } as never, configFile(directory))

    assert.deepEqual(written, DEFAULTS)
  })
})

test('readComposerSendPrefs clamps the pause knobs and ignores an unknown grace entry', () => {
  withTempDir(directory => {
    fs.writeFileSync(
      configFile(directory),
      JSON.stringify({
        enterSends: false,
        sendGraceFor: ['sometimes'],
        sendGraceMs: -5,
        sendOnPause: true,
        typingIdleMs: 99_999
      }),
      'utf8'
    )

    assert.deepEqual(readComposerSendPrefs(configFile(directory)), {
      ...DEFAULTS,
      // An unknown situation is dropped rather than defaulted: the user asked
      // for something, and silently delaying a send they did not ask for is the
      // worse failure. The empty set is a real answer.
      sendGraceFor: [],
      // -5 clamps UP to the minimum (0 = no hold), not down to the maximum.
      sendGraceMs: SEND_GRACE_MIN_MS,
      enterSends: false,
      sendOnPause: true,
      typingIdleMs: TYPING_IDLE_MAX_MS
    })
  })
})

// ---------------------------------------------------------------------------
// Migration. Every one of these values is on somebody's disk right now, and the
// failure mode is the same in each case: landing an old setting on the DEFAULT
// turns Enter back into a send, which is what the user opened Settings to stop.
// ---------------------------------------------------------------------------

test('readComposerSendPrefs migrates a retired `mode` into behaviour plus gestures', () => {
  withTempDir(directory => {
    const migrate = (mode: string) => {
      fs.writeFileSync(configFile(directory), JSON.stringify({ mode }), 'utf8')

      const prefs = readComposerSendPrefs(configFile(directory))

      return { enterSends: prefs.enterSends, gestures: [prefs.sendOnDoubleTap, prefs.sendOnPause, prefs.sendOnHold] }
    }

    assert.deepEqual(migrate('enter'), { enterSends: true, gestures: [false, false, false] })
    assert.deepEqual(migrate('double-enter'), { enterSends: false, gestures: [true, false, false] })
    assert.deepEqual(migrate('pause'), { enterSends: false, gestures: [false, true, false] })
    // `mod-enter` was "Enter only breaks the line, the chord sends" — which is
    // this model with no gestures armed.
    assert.deepEqual(migrate('mod-enter'), { enterSends: false, gestures: [false, false, false] })
    // `hold` was briefly a mode of its own before becoming a gesture.
    assert.deepEqual(migrate('hold'), { enterSends: false, gestures: [false, false, true] })
    // An unknown value is the one case where the default is the right answer.
    assert.deepEqual(migrate('triple-enter'), { enterSends: true, gestures: [false, false, false] })
  })
})

test('readComposerSendPrefs migrates the retired three-way grace scope', () => {
  withTempDir(directory => {
    const legacy = (scope: string) => {
      fs.writeFileSync(configFile(directory), JSON.stringify({ sendGrace: scope }), 'utf8')

      return readComposerSendPrefs(configFile(directory)).sendGraceFor
    }

    // `inferred` was the guessed send only — a held key was never inferred.
    assert.deepEqual(legacy('inferred'), ['pause'])
    assert.deepEqual(legacy('off'), [])
    assert.deepEqual(legacy('all'), ['enter', 'doubleTap', 'pause', 'hold'])
  })
})

test('readComposerSendPrefs never turns the auto-send on for an existing file', () => {
  withTempDir(directory => {
    // The auto-send did not exist when these were written. Defaulting it on
    // would start sending messages nobody asked to send.
    for (const stored of [{ mode: 'pause' }, { mode: 'hold' }, { enterSends: false }]) {
      fs.writeFileSync(configFile(directory), JSON.stringify(stored), 'utf8')

      assert.equal(readComposerSendPrefs(configFile(directory)).sendOnIdle, false)
    }
  })
})

test('writeComposerSendPrefs keeps a chosen set of situations', () => {
  withTempDir(directory => {
    const sendGraceFor = ['doubleTap', 'hold'] as const
    const written = writeComposerSendPrefs({ ...DEFAULTS, sendGraceFor, sendGraceMs: 0 }, configFile(directory))

    assert.deepEqual(written, { ...DEFAULTS, sendGraceFor, sendGraceMs: 0 })
    assert.deepEqual(readComposerSendPrefs(configFile(directory)), written)
  })
})
