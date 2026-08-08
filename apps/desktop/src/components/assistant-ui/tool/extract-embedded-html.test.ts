import { describe, it, expect } from 'vitest'
import { extractEmbeddedHtml } from './fallback-model'

describe('extractEmbeddedHtml', () => {
  it('returns empty string for null/undefined', () => {
    expect(extractEmbeddedHtml(null)).toBe('')
    expect(extractEmbeddedHtml(undefined)).toBe('')
    expect(extractEmbeddedHtml('not an object')).toBe('')
  })

  it('returns empty string when no embedded HTML is present', () => {
    expect(extractEmbeddedHtml({ result: 'plain text' })).toBe('')
    expect(extractEmbeddedHtml({ content: [{ type: 'text', text: 'hello' }] })).toBe('')
    expect(extractEmbeddedHtml({ content: 'just a string' })).toBe('')
    expect(extractEmbeddedHtml({ content: [] })).toBe('')
    expect(extractEmbeddedHtml({})).toBe('')
  })

  it('extracts HTML from _embedded_html field (primary path)', () => {
    const result = {
      result: '{"balance":2.5}',
      _embedded_html: '<!DOCTYPE html><html><body>Balance Card</body></html>',
    }
    expect(extractEmbeddedHtml(result)).toBe('<!DOCTYPE html><html><body>Balance Card</body></html>')
  })

  it('extracts HTML from a content array with a resource item (secondary path)', () => {
    const result = {
      content: [
        { type: 'text', text: '{"balance":2.5}' },
        {
          type: 'resource',
          resource: {
            uri: 'ui://example/balance-card',
            mimeType: 'text/html',
            text: '<!DOCTYPE html><html><body>Balance Card</body></html>',
          },
        },
      ],
    }
    expect(extractEmbeddedHtml(result)).toBe('<!DOCTYPE html><html><body>Balance Card</body></html>')
  })

  it('prefers _embedded_html over content array', () => {
    const result = {
      _embedded_html: '<html>from field</html>',
      content: [
        {
          type: 'resource',
          resource: { mimeType: 'text/html', text: '<html>from content</html>' },
        },
      ],
    }
    expect(extractEmbeddedHtml(result)).toBe('<html>from field</html>')
  })

  it('returns empty string for non-HTML resources', () => {
    const result = {
      content: [
        {
          type: 'resource',
          resource: {
            uri: 'file://some/file.json',
            mimeType: 'application/json',
            text: '{"data": 1}',
          },
        },
      ],
    }
    expect(extractEmbeddedHtml(result)).toBe('')
  })

  it('returns empty string when _embedded_html is empty', () => {
    expect(extractEmbeddedHtml({ _embedded_html: '' })).toBe('')
    expect(extractEmbeddedHtml({ _embedded_html: '   ' })).toBe('')
  })

  it('returns empty string when resource text is empty', () => {
    const result = {
      content: [
        {
          type: 'resource',
          resource: {
            uri: 'ui://example/card',
            mimeType: 'text/html',
            text: '',
          },
        },
      ],
    }
    expect(extractEmbeddedHtml(result)).toBe('')
  })
})