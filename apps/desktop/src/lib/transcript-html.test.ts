import { expect, it } from 'vitest'

import { DEFAULT_TRANSCRIPT_OPTIONS, type TranscriptDocument } from './transcript-document'
import { renderHtml } from './transcript-html'

const doc: TranscriptDocument = {
  version: 1,
  session_id: 's',
  title: 'Demo',
  exported_at: 'now',
  options: { ...DEFAULT_TRANSCRIPT_OPTIONS, include_tools: true },
  warnings: [],
  messages: [{ role: 'user', speaker: 'Alice', text: 'Visit https://example.com' }]
}

it('links safe URLs in the HTML transcript', () => {
  const page = new DOMParser().parseFromString(renderHtml(doc), 'text/html')
  expect(page.querySelector('a')?.getAttribute('href')).toBe('https://example.com')
})

it('renders safe self-contained cards, disclosures and attachment references', () => {
  const value: TranscriptDocument = {
    ...doc,
    options: { ...doc.options, include_links: false },
    warnings: ['warning <x>'],
    messages: [
      {
        role: 'tool',
        speaker: 'Tool <x>',
        kind: 'aside',
        timestamp: 'today',
        text: '<script>bad</script> https://example.com',
        reasoning: '秘密',
        tool_calls: [{ name: 'run', arguments: '<code>' }],
        attachments: [{ kind: 'file', name: 'Notes', reference: 'C:/secret', local: true }]
      }
    ]
  }

  const page = new DOMParser().parseFromString(renderHtml(value), 'text/html')
  expect(page.querySelector('meta[http-equiv="Content-Security-Policy"]')?.getAttribute('content')).toContain(
    "default-src 'none'"
  )
  expect(page.querySelector('script')).toBeNull()
  expect(page.querySelector('a')).toBeNull()
  expect(page.body.textContent).toContain('Aside — /btw')
  expect(page.body.textContent).toContain('<script>bad</script> https://example.com')
  expect(page.body.textContent).toContain('may be unavailable')
  expect(page.querySelectorAll('details').length).toBeGreaterThanOrEqual(2)
  expect(page.querySelector('details')?.hasAttribute('open')).toBe(false)
  expect(page.querySelector('style')?.textContent).toContain('@media print')
})

it('embeds only bounded valid inline raster images', () => {
  const png =
    'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGNgYGD4DwABBAEAX+XDSwAAAABJRU5ErkJggg=='

  const value: TranscriptDocument = {
    ...doc,
    messages: [
      {
        role: 'user',
        speaker: 'U',
        text: '',
        attachments: [
          { kind: 'image', name: 'safe', data_url: png },
          { kind: 'image', name: 'bad', data_url: 'data:image/svg+xml;base64,PHN2Zz4=' }
        ]
      }
    ]
  }

  const page = new DOMParser().parseFromString(renderHtml(value), 'text/html')
  expect(page.querySelectorAll('img')).toHaveLength(1)
  expect(page.querySelector('img')?.getAttribute('src')).toBe(png)
  expect(page.body.textContent).toContain('image not embedded')
})
