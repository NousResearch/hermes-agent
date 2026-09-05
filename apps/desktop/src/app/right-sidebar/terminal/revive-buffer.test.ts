import { describe, expect, it } from 'vitest'

import {
  cleanReviveSnapshot,
  mergeReviveSnapshot,
  parseOscCwd,
  resolveLiveSnapshotWindow
} from './use-terminal-session'

// A default-PowerShell idle prompt: no blank-line separator before it.
const PS_PROMPT = 'PS C:\\Users\\Aleksandr>'

const FISH_GREETING = [
  'Welcome to fish, the friendly interactive shell',
  'Type \u001b[32mhelp\u001b[0m for instructions on how to use fish'
]

const FISH_PROMPT = '/repo \u001b[90mfeat/desktop-web*'
const FISH_MARKER = '\u001b[35m❯\u001b[0m'

describe('cleanReviveSnapshot', () => {
  it('drops a spaced trailing prompt block after a blank separator (starship)', () => {
    const snapshot = ['echo hi', 'hi', '', PS_PROMPT].join('\r\n')

    expect(cleanReviveSnapshot(snapshot)).toBe('echo hi\r\nhi')
  })

  it('drops a multi-line prompt block after a blank separator (powerline)', () => {
    const snapshot = ['work', '', '┌─ user@host ~/project', '└─$'].join('\r\n')

    expect(cleanReviveSnapshot(snapshot)).toBe('work')
  })

  it('drops a single-line trailing prompt with no preceding blank line (PowerShell)', () => {
    // Default PowerShell prints no blank line before its prompt; the fresh shell
    // reprints it on boot, so the redundant idle prompt must be trimmed here.
    const snapshot = ['echo hi', 'hi', PS_PROMPT].join('\r\n')

    expect(cleanReviveSnapshot(snapshot)).toBe('echo hi\r\nhi')
  })

  it('keeps command output and drops only the trailing prompt on a long history', () => {
    const history = ['cmd1', 'out1', 'cmd2', 'out2']
    const snapshot = [...history, PS_PROMPT].join('\r\n')

    expect(cleanReviveSnapshot(snapshot)).toBe(history.join('\r\n'))
  })

  it('reduces a lone prompt to an empty buffer', () => {
    expect(cleanReviveSnapshot(PS_PROMPT)).toBe('')
    expect(cleanReviveSnapshot([PS_PROMPT, '', ''].join('\r\n'))).toBe('')
  })

  it('returns empty for a blank-only buffer without throwing', () => {
    expect(cleanReviveSnapshot('')).toBe('')
    expect(cleanReviveSnapshot('\r\n  \r\n')).toBe('')
  })

  it('drops only the live Fish greeting and preserves the current prompt tail', () => {
    const snapshot = [
      ...FISH_GREETING,
      FISH_PROMPT,
      `${FISH_MARKER} echo hi`,
      'hi',
      FISH_PROMPT,
      FISH_MARKER
    ].join('\r\n')

    expect(cleanReviveSnapshot(snapshot, 'fish')).toBe(
      [FISH_PROMPT, `${FISH_MARKER} echo hi`, 'hi', FISH_PROMPT, FISH_MARKER].join('\r\n')
    )
  })

  it('preserves incomplete Fish output instead of guessing that its tail is a prompt', () => {
    const snapshot = [...FISH_GREETING, FISH_PROMPT, `${FISH_MARKER} printf repeated`, 'same', 'same'].join(
      '\r\n'
    )

    expect(cleanReviveSnapshot(snapshot, 'fish')).toBe(
      [FISH_PROMPT, `${FISH_MARKER} printf repeated`, 'same', 'same'].join('\r\n')
    )
  })

  it('preserves greeting-shaped Fish command output after the live boot greeting', () => {
    const snapshot = [
      ...FISH_GREETING,
      FISH_PROMPT,
      `${FISH_MARKER} printf greeting`,
      ...FISH_GREETING,
      'done',
      FISH_PROMPT,
      FISH_MARKER
    ].join('\r\n')

    expect(cleanReviveSnapshot(snapshot, 'fish')).toBe(
      [FISH_PROMPT, `${FISH_MARKER} printf greeting`, ...FISH_GREETING, 'done', FISH_PROMPT, FISH_MARKER].join(
        '\r\n'
      )
    )
  })
})

describe('mergeReviveSnapshot', () => {
  it('preserves legacy duplicate rows while removing only the new live greeting', () => {
    const restored = [
      FISH_PROMPT,
      `${FISH_MARKER} whoami`,
      'arkouda',
      ...Array.from({ length: 8 }, () => FISH_PROMPT)
    ].join('\r\n')

    const live = [
      ...FISH_GREETING,
      FISH_PROMPT,
      `${FISH_MARKER} echo next`,
      'next',
      FISH_PROMPT,
      FISH_MARKER
    ].join('\r\n')

    expect(mergeReviveSnapshot(restored, live, 'fish')).toBe(
      [restored, FISH_PROMPT, `${FISH_MARKER} echo next`, 'next', FISH_PROMPT, FISH_MARKER].join('\r\n')
    )
  })

  it('preserves arbitrary and repeated restored output byte-for-byte', () => {
    const restored = ['same', '❯ decorative output', ...FISH_GREETING, ...Array.from({ length: 240 }, () => 'same')].join(
      '\r\n'
    )

    const live = [...FISH_GREETING, FISH_PROMPT, FISH_MARKER].join('\r\n')
    const merged = mergeReviveSnapshot(restored, live, 'fish')

    expect(merged.startsWith(`${restored}\r\n`)).toBe(true)
    expect(merged.slice(restored.length + 2)).toBe([FISH_PROMPT, FISH_MARKER].join('\r\n'))
  })

  it('does not strip greeting-shaped output when a row budget starts inside live history', () => {
    const truncatedLive = [...FISH_GREETING, 'real output', FISH_PROMPT, FISH_MARKER].join('\r\n')

    expect(mergeReviveSnapshot('', truncatedLive, 'fish', false)).toBe(truncatedLive)
  })
})

describe('resolveLiveSnapshotWindow', () => {
  it('uses a live marker at or before the current cursor', () => {
    expect(resolveLiveSnapshotWindow(2, 3, 3, 200)).toEqual({ keepRestored: true, start: 2 })
  })

  it('rejects a numerically valid marker left below a reset cursor', () => {
    expect(resolveLiveSnapshotWindow(2, 4, 0, 200)).toBeNull()
  })

  it('rejects disposed or unregistered markers and drops restored history once live output exceeds the row budget', () => {
    expect(resolveLiveSnapshotWindow(-1, 4, 3, 200)).toBeNull()
    expect(resolveLiveSnapshotWindow(2, 4, 3, 200, false)).toBeNull()
    expect(resolveLiveSnapshotWindow(2, 12, 12, 5)).toEqual({ keepRestored: false, start: 8 })
  })
})

describe('parseOscCwd', () => {
  it('parses an OSC 7 file URI and percent-decodes it', () => {
    expect(parseOscCwd(7, 'file://host/Users/al/my%20project')).toBe('/Users/al/my project')
  })

  it('strips the leading slash from a Windows OSC 7 file URI', () => {
    expect(parseOscCwd(7, 'file:///C:/Users/Aleksandr/project')).toBe('C:/Users/Aleksandr/project')
  })

  it('ignores non-file OSC 7 payloads', () => {
    expect(parseOscCwd(7, 'https://example.com')).toBeNull()
    expect(parseOscCwd(7, '')).toBeNull()
  })

  it('parses an OSC 9;9 cwd payload and unquotes it', () => {
    expect(parseOscCwd(9, '9;"C:\\Users\\Aleksandr"')).toBe('C:\\Users\\Aleksandr')
    expect(parseOscCwd(9, '9;/home/al/src')).toBe('/home/al/src')
  })

  it('ignores OSC 9 sub-commands other than 9;<path> (e.g. progress)', () => {
    expect(parseOscCwd(9, '4;3')).toBeNull()
    expect(parseOscCwd(9, 'some notification')).toBeNull()
  })
})
