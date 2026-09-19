import { describe, expect, it } from 'vitest'

import {
  directiveFrameHeight,
  frameSizeFromMessage,
  intentFromMessage,
  themePrelude,
  withInlineChrome
} from './inline-preview-directive'

describe('directiveFrameHeight', () => {
  it('returns null (auto-size) when absent or garbage', () => {
    expect(directiveFrameHeight(undefined)).toBeNull()
    expect(directiveFrameHeight('')).toBeNull()
    expect(directiveFrameHeight('tall')).toBeNull()
    expect(directiveFrameHeight('12.5')).toBeNull()
  })

  it('clamps an explicit height to the sane band', () => {
    expect(directiveFrameHeight('50')).toBe(120)
    expect(directiveFrameHeight('480')).toBe(480)
    expect(directiveFrameHeight('99999')).toBe(1200)
  })
})

describe('withInlineChrome', () => {
  const prelude = themePrelude({ '--foreground': '#eee' }, 'Inter', 'dark')

  it('puts the theme prelude FIRST so page styles override it', () => {
    const doc = '<html><head><style>body{color:red}</style></head><body><h1>hi</h1></body></html>'
    const framed = withInlineChrome(doc, 'tok', prelude)

    expect(framed.startsWith(prelude)).toBe(true)
    expect(framed.indexOf(prelude)).toBeLessThan(framed.indexOf('color:red'))
  })

  it('injects the measuring script before </body>', () => {
    const doc = '<html><body><h1>hi</h1></body></html>'
    const framed = withInlineChrome(doc, 'tok', prelude)

    expect(framed.indexOf('postMessage')).toBeGreaterThan(framed.indexOf('<h1>'))
    expect(framed.indexOf('postMessage')).toBeLessThan(framed.indexOf('</body>'))
    expect(framed).toContain('"tok"')
  })

  it('appends the script when there is no body close tag', () => {
    const framed = withInlineChrome('<h1>fragment</h1>', 'tok', prelude)

    expect(framed).toContain('<h1>fragment</h1>')
    expect(framed).toContain('postMessage')
  })
})

describe('themePrelude', () => {
  it.each(['light', 'dark'] as const)('sets the iframe document color scheme to %s', colorScheme => {
    expect(themePrelude({}, '', colorScheme)).toContain(`color-scheme:${colorScheme}`)
  })

  it('carries resolved tokens, transparent background, and the app font', () => {
    const prelude = themePrelude(
      { '--foreground': 'oklch(0.9 0 0)', '--accent': '#7aa2f7' },
      'Inter, sans-serif',
      'dark'
    )

    expect(prelude).toContain('--foreground:oklch(0.9 0 0)')
    expect(prelude).toContain('--accent:#7aa2f7')
    expect(prelude).toContain('background:transparent')
    expect(prelude).toContain('font-family:Inter, sans-serif')
  })

  it('omits the font rule when no font resolved', () => {
    expect(themePrelude({}, '', 'light')).not.toContain('font-family')
  })
})

describe('frameSizeFromMessage', () => {
  const msg = (over: Record<string, unknown> = {}) => ({
    type: 'hermes-inline-preview-size',
    token: 'tok',
    height: 500,
    width: 300,
    ...over
  })

  it('accepts our message with our token, height clamped', () => {
    expect(frameSizeFromMessage(msg(), 'tok')).toEqual({ height: 500, width: 300 })
    expect(frameSizeFromMessage(msg({ height: 12 }), 'tok')?.height).toBe(120)
    expect(frameSizeFromMessage(msg({ height: 5000 }), 'tok')?.height).toBe(1200)
    expect(frameSizeFromMessage(msg({ height: 500.7 }), 'tok')?.height).toBe(501)
  })

  it('sanitizes width to 0 when missing or hostile', () => {
    expect(frameSizeFromMessage(msg({ width: undefined }), 'tok')?.width).toBe(0)
    expect(frameSizeFromMessage(msg({ width: 'wide' }), 'tok')?.width).toBe(0)
    expect(frameSizeFromMessage(msg({ width: Infinity }), 'tok')?.width).toBe(0)
    expect(frameSizeFromMessage(msg({ width: -10 }), 'tok')?.width).toBe(0)
  })

  it('rejects wrong type, wrong token, and hostile shapes', () => {
    expect(frameSizeFromMessage(msg({ type: 'other' }), 'tok')).toBeNull()
    expect(frameSizeFromMessage(msg({ token: 'stolen' }), 'tok')).toBeNull()
    expect(frameSizeFromMessage(msg({ height: 'tall' }), 'tok')).toBeNull()
    expect(frameSizeFromMessage(msg({ height: Infinity }), 'tok')).toBeNull()
    expect(frameSizeFromMessage(msg({ height: -5 }), 'tok')).toBeNull()
    expect(frameSizeFromMessage(null, 'tok')).toBeNull()
    expect(frameSizeFromMessage('str', 'tok')).toBeNull()
  })
})

describe('intentFromMessage', () => {
  const msg = (over: Record<string, unknown> = {}) => ({
    type: 'hermes-inline-preview-intent',
    token: 'tok',
    prompt: 'get-price eth',
    ...over
  })

  it('accepts our intent with our token, trimmed', () => {
    expect(intentFromMessage(msg(), 'tok')).toBe('get-price eth')
    expect(intentFromMessage(msg({ prompt: '  hi  ' }), 'tok')).toBe('hi')
  })

  it('caps runaway prompts to a sentence-sized budget', () => {
    expect(intentFromMessage(msg({ prompt: 'x'.repeat(9000) }), 'tok')).toHaveLength(500)
  })

  it('rejects wrong token, wrong type, empty, and hostile shapes', () => {
    expect(intentFromMessage(msg({ token: 'stolen' }), 'tok')).toBeNull()
    expect(intentFromMessage(msg({ type: 'hermes-inline-preview-size' }), 'tok')).toBeNull()
    expect(intentFromMessage(msg({ prompt: '   ' }), 'tok')).toBeNull()
    expect(intentFromMessage(msg({ prompt: 42 }), 'tok')).toBeNull()
    expect(intentFromMessage(null, 'tok')).toBeNull()
  })
})

describe('withInlineChrome intent wiring', () => {
  it('injects hermes.send and the data-hermes-send click bridge', () => {
    const framed = withInlineChrome('<html><body><h1>w</h1></body></html>', 'tok', '')

    expect(framed).toContain('window.hermes={send:send}')
    expect(framed).toContain('data-hermes-send')
    expect(framed).toContain('hermes-inline-preview-intent')
  })
})
