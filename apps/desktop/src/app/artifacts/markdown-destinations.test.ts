import { describe, expect, it } from 'vitest'

import type { SessionInfo } from '@/types/hermes'

import { collectArtifactsForSession } from './artifact-utils'

const session: SessionInfo = {
  ended_at: null,
  id: 's',
  input_tokens: 0,
  is_active: false,
  last_active: 1000,
  message_count: 1,
  model: null,
  output_tokens: 0,
  preview: null,
  source: null,
  started_at: 1000,
  title: 'Session',
  tool_call_count: 0
}

function values(content: string): string[] {
  return collectArtifactsForSession(session, [{ content, role: 'assistant', timestamp: 2000 }]).map(a => a.value)
}

describe('Markdown destinations on the Artifacts page', () => {
  it('keeps the whole CommonMark destination: angle brackets, spaces, parentheses, titles', () => {
    expect(
      values(
        [
          '![chart](</home/u/My Pictures/q3 chart.png>)',
          '[report](<~/Reports/Q3 (final).pdf>)',
          '![logo](/srv/out/logo.png "Company logo")',
          '[notes](/srv/out/notes_(v2).md)'
        ].join('\n')
      ).sort()
    ).toEqual(
      ['/home/u/My Pictures/q3 chart.png', '/srv/out/logo.png', '/srv/out/notes_(v2).md', '~/Reports/Q3 (final).pdf'].sort()
    )
  })

  it('classifies a spaced angle-bracket Windows image as an image labelled by its file name', () => {
    const [artifact] = collectArtifactsForSession(session, [
      { content: 'Saved ![render](<C:\\Users\\me\\My Renders\\frame 01.webp>)', role: 'assistant', timestamp: 2000 }
    ])

    expect(artifact).toMatchObject({ kind: 'image', label: 'frame 01.webp', value: 'C:\\Users\\me\\My Renders\\frame 01.webp' })
  })
})
