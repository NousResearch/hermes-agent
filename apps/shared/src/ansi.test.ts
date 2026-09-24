import { describe, expect, it } from 'vitest'

import { hasAnsi, sanitizeAnsiForRender, stripAnsi } from './ansi'

const ESC = String.fromCharCode(27)
const BEL = String.fromCharCode(7)

describe('stripAnsi', () => {
  it('strips CSI/OSC/control bytes from plain previews', () => {
    const sample = `A${ESC}[31mB${ESC}[39m${ESC}[2J${ESC}]0;title${BEL}C${ESC}[?25lD`

    expect(stripAnsi(sample)).toBe('ABCD')
  })

  it('strips incomplete CSI prefixes and carriage returns', () => {
    const sample = `A${ESC}[31mB${ESC}[12;${ESC}[CD\rE`

    expect(stripAnsi(sample)).toBe('ABDE')
  })

  it('strips multi-byte non-CSI ESC sequences without leaving trailing bytes', () => {
    const sample = `A${ESC}(0B${ESC}%GC${ESC})0D`

    expect(stripAnsi(sample)).toBe('ABCD')
    expect(sanitizeAnsiForRender(sample)).toBe('ABCD')
  })

  // Desktop's former SGR-only stripper left OSC-8 hyperlink payloads, DCS strings and
  // truncated CSI tails visible in chat system messages; every surface now gets the
  // full coverage.
  it('leaves only visible text when OSC-8 hyperlinks, DCS strings, partial CSI and SGR mix', () => {
    const sample = `${ESC}]8;;https://example.com${BEL}link${ESC}]8;;${BEL} ${ESC}Pq#0${ESC}\\${ESC}[1;32mok${ESC}[0m${ESC}[12;`

    const stripped = stripAnsi(sample)

    expect(stripped).toBe('link ok')
    expect(stripped).not.toContain(ESC)
    expect(stripped).not.toContain(BEL)
  })
})

describe('sanitizeAnsiForRender', () => {
  it('keeps SGR color spans but removes cursor controls for Ansi rendering', () => {
    const sample = `A${ESC}[31mB${ESC}[39m${ESC}[2J${ESC}]0;title${BEL}${ESC}[?25lC`

    expect(sanitizeAnsiForRender(sample)).toBe(`A${ESC}[31mB${ESC}[39mC`)
  })

  it('keeps valid SGR while removing dangling CSI and carriage returns', () => {
    const sample = `A${ESC}[31mB${ESC}[12;${ESC}[39mC\rD`

    expect(sanitizeAnsiForRender(sample)).toBe(`A${ESC}[31mB${ESC}[39mCD`)
  })
})

describe('hasAnsi', () => {
  it('detects non-CSI escape prefixes too', () => {
    expect(hasAnsi(`ok${ESC}Ppayload${ESC}\\`)).toBe(true)
    expect(hasAnsi('plain')).toBe(false)
  })
})
