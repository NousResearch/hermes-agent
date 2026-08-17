import { describe, expect, it } from 'vitest'

import { directiveFrameHeight, frameHeightFromMessage, withMeasurement } from './inline-preview-directive'

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

describe('withMeasurement', () => {
  it('injects the measuring script before </body>', () => {
    const doc = '<html><body><h1>hi</h1></body></html>'
    const framed = withMeasurement(doc, 'tok')

    expect(framed.indexOf('<script>')).toBeGreaterThan(framed.indexOf('<h1>'))
    expect(framed.indexOf('</script>')).toBeLessThan(framed.indexOf('</body>'))
    expect(framed).toContain('"tok"')
  })

  it('appends when there is no body close tag', () => {
    const framed = withMeasurement('<h1>fragment</h1>', 'tok')

    expect(framed.startsWith('<h1>fragment</h1>')).toBe(true)
    expect(framed).toContain('postMessage')
  })
})

describe('frameHeightFromMessage', () => {
  const msg = (over: Record<string, unknown> = {}) => ({
    type: 'hermes-inline-preview-size',
    token: 'tok',
    height: 500,
    ...over
  })

  it('accepts our message with our token, clamped', () => {
    expect(frameHeightFromMessage(msg(), 'tok')).toBe(500)
    expect(frameHeightFromMessage(msg({ height: 12 }), 'tok')).toBe(120)
    expect(frameHeightFromMessage(msg({ height: 5000 }), 'tok')).toBe(1200)
    expect(frameHeightFromMessage(msg({ height: 500.7 }), 'tok')).toBe(501)
  })

  it('rejects wrong type, wrong token, and hostile shapes', () => {
    expect(frameHeightFromMessage(msg({ type: 'other' }), 'tok')).toBeNull()
    expect(frameHeightFromMessage(msg({ token: 'stolen' }), 'tok')).toBeNull()
    expect(frameHeightFromMessage(msg({ height: 'tall' }), 'tok')).toBeNull()
    expect(frameHeightFromMessage(msg({ height: Infinity }), 'tok')).toBeNull()
    expect(frameHeightFromMessage(msg({ height: -5 }), 'tok')).toBeNull()
    expect(frameHeightFromMessage(null, 'tok')).toBeNull()
    expect(frameHeightFromMessage('str', 'tok')).toBeNull()
  })
})
