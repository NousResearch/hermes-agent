import { expect, it } from 'vitest'

import { DEFAULT_TRANSCRIPT_OPTIONS, type TranscriptDocument } from './transcript-document'
import { renderTranscript } from './transcript-render'

const doc: TranscriptDocument = {
  version: 1,
  session_id: 'sample',
  title: 'Sample',
  exported_at: '2026-11-03T18:00:00Z',
  options: { ...DEFAULT_TRANSCRIPT_OPTIONS },
  warnings: [],
  messages: [
    {
      role: 'user',
      speaker: 'User',
      text: '**Hello** world',
      attachments: [
        {
          kind: 'image',
          name: 'Raster',
          data_url:
            'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGNQLdrxHwAEXgJPUgh8kgAAAABJRU5ErkJggg=='
        }
      ]
    }
  ]
}

it('connects HTML exports to the self-contained image-capable renderer', () => {
  const page = new DOMParser().parseFromString(renderTranscript(doc, 'html').content, 'text/html')
  expect(page.querySelectorAll('article.turn')).toHaveLength(1)
  expect(page.querySelectorAll('img')).toHaveLength(1)
  expect(page.querySelector('meta[http-equiv="Content-Security-Policy"]')).not.toBeNull()
})

it('connects Markdown exports to readable message formatting and inline raster images', () => {
  const output = renderTranscript(doc, 'md').content
  expect(output).toContain('**Hello** world')
  expect(output).not.toContain('<pre>**Hello** world</pre>')
  expect(output).toContain('![Raster](data:image/png;base64,')
})
