import { describe, expect, it } from 'vitest'

import { ansiColorClass, hasAnsiCodes, parseAnsi } from './ansi'

const ESC = '\x1b'

describe('parseAnsi', () => {
  it('returns a single default segment for plain text', () => {
    expect(parseAnsi('hello world')).toEqual([{ bold: false, fg: null, text: 'hello world' }])
  })

  it('returns nothing for an empty string', () => {
    expect(parseAnsi('')).toEqual([])
  })

  it('parses a basic foreground color sequence and resets', () => {
    const input = `${ESC}[31merror${ESC}[0m ok`

    expect(parseAnsi(input)).toEqual([
      { bold: false, fg: 'red', text: 'error' },
      { bold: false, fg: null, text: ' ok' }
    ])
  })

  it('treats bold (1) and bold-off (22) as toggles without affecting fg', () => {
    const input = `${ESC}[1mloud${ESC}[22m quiet`

    expect(parseAnsi(input)).toEqual([
      { bold: true, fg: null, text: 'loud' },
      { bold: false, fg: null, text: ' quiet' }
    ])
  })

  it('treats default-fg (39) as a foreground-only reset (keeps bold)', () => {
    const input = `${ESC}[1;31mboth${ESC}[39mbold-only`

    expect(parseAnsi(input)).toEqual([
      { bold: true, fg: 'red', text: 'both' },
      { bold: true, fg: null, text: 'bold-only' }
    ])
  })

  it('handles bright colors via the 90-97 range', () => {
    expect(parseAnsi(`${ESC}[92mgreen`)).toEqual([{ bold: false, fg: 'bright-green', text: 'green' }])
  })

  it('coalesces adjacent runs with the same style', () => {
    const input = `${ESC}[31ma${ESC}[31mb${ESC}[31mc`

    expect(parseAnsi(input)).toEqual([{ bold: false, fg: 'red', text: 'abc' }])
  })

  it('skips 256-color (38;5) trailing args without painting fg or leaking the params as text', () => {
    // 256-color and truecolor aren't rendered (FG_BY_CODE doesn't cover them),
    // but the parser must consume the trailing `;5;<n>` / `;2;r;g;b` args so
    // they never bleed into the visible segment text.
    const segments = parseAnsi(`${ESC}[38;5;208morange${ESC}[0m`)

    expect(segments).toHaveLength(1)
    expect(segments[0].fg).toBe(null)
    expect(segments[0].text).toBe('orange')
  })

  it('skips truecolor (38;2;r;g;b) trailing args', () => {
    const segments = parseAnsi(`${ESC}[38;2;10;20;30mrgb${ESC}[0m`)

    expect(segments).toHaveLength(1)
    expect(segments[0].fg).toBe(null)
    expect(segments[0].text).toBe('rgb')
  })

  it('skips extended-background (48;5;n) trailing args instead of leaking them as SGR codes', () => {
    // We don't paint backgrounds, but the parser must still consume the
    // `;5;<n>` payload after 48 so its index value isn't re-read as a
    // standalone SGR code. Here `1` would otherwise turn bold on.
    const segments = parseAnsi(`${ESC}[48;5;1mtext`)

    expect(segments).toEqual([{ bold: false, fg: null, text: 'text' }])
  })

  it('does not let a 48;5 index value leak in and set a foreground color', () => {
    // `31` is the leaked index here; before the fix it was parsed as the
    // standalone red-foreground code.
    const segments = parseAnsi(`${ESC}[48;5;31mtext`)

    expect(segments).toEqual([{ bold: false, fg: null, text: 'text' }])
  })

  it('skips truecolor extended-background (48;2;r;g;b) trailing args', () => {
    // The trailing `0` would otherwise be read as a full reset (SGR 0).
    const segments = parseAnsi(`${ESC}[1;31m${ESC}[48;2;255;0;0mtext`)

    expect(segments).toEqual([{ bold: true, fg: 'red', text: 'text' }])
  })

  it('keeps the foreground when a 48 background is set in the same SGR run', () => {
    // 31 (red fg) then 48;5;2 (bg) — the bg index must not clobber the fg.
    const segments = parseAnsi(`${ESC}[31;48;5;2mtext`)

    expect(segments).toEqual([{ bold: false, fg: 'red', text: 'text' }])
  })

  it('drops non-SGR CSI sequences (cursor motion, erase) without consuming surrounding text', () => {
    const input = `before${ESC}[2Jmiddle${ESC}[10;5Hafter`

    expect(parseAnsi(input)).toEqual([{ bold: false, fg: null, text: 'beforemiddleafter' }])
  })

  it('treats an empty SGR parameter (ESC[m) as a full reset', () => {
    const input = `${ESC}[1;31mfoo${ESC}[mbar`

    expect(parseAnsi(input)).toEqual([
      { bold: true, fg: 'red', text: 'foo' },
      { bold: false, fg: null, text: 'bar' }
    ])
  })
})

describe('hasAnsiCodes', () => {
  it('returns false for plain text', () => {
    expect(hasAnsiCodes('hello world')).toBe(false)
  })

  it('returns true when any CSI introducer is present', () => {
    expect(hasAnsiCodes(`${ESC}[31mred`)).toBe(true)
  })
})

describe('ansiColorClass', () => {
  it('returns a non-empty Tailwind class string for every supported color', () => {
    const colors = [
      'black',
      'red',
      'green',
      'yellow',
      'blue',
      'magenta',
      'cyan',
      'white',
      'bright-black',
      'bright-red',
      'bright-green',
      'bright-yellow',
      'bright-blue',
      'bright-magenta',
      'bright-cyan',
      'bright-white'
    ] as const

    for (const color of colors) {
      expect(ansiColorClass(color)).toMatch(/\S/)
    }
  })
})
