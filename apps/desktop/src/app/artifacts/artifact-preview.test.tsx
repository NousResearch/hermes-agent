import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'

import { artifactPreviewDescriptor, ArtifactPreviewThumbnail } from './artifact-preview'
import type { ArtifactRecord } from './artifact-utils'

function makeArtifact(value: string, kind: ArtifactRecord['kind'] = 'file'): ArtifactRecord {
  return {
    href: value,
    id: value,
    kind,
    label: value.split('/').pop() || value,
    previewable: false,
    profile: 'default',
    sessionId: 'session-1',
    sessionTitle: 'Artifacts',
    timestamp: 1,
    value
  }
}

afterEach(() => cleanup())

describe('artifactPreviewDescriptor', () => {
  it.each([
    ['https://example.com/report.pdf?download=1', 'file', 'document', 'PDF'],
    ['./src/widget.tsx', 'file', 'code', 'TSX'],
    ['/tmp/data.yaml', 'file', 'code', 'YAML'],
    ['/tmp/table.xlsx', 'file', 'spreadsheet', 'XLSX'],
    ['/tmp/slides.pptx', 'file', 'presentation', 'PPTX'],
    ['/tmp/clip.webm', 'file', 'video', 'WEBM'],
    ['/tmp/sound.flac', 'file', 'audio', 'FLAC'],
    ['/tmp/archive.zip', 'file', 'archive', 'ZIP'],
    ['/tmp/photo.avif', 'image', 'image', 'AVIF'],
    ['https://example.com/docs', 'link', 'web', 'WEB']
  ] as const)('classifies %s as %s', (value, kind, expectedKind, badge) => {
    const descriptor = artifactPreviewDescriptor(makeArtifact(value, kind))

    expect(descriptor.kind).toBe(expectedKind)
    expect(descriptor.badge).toBe(badge)
  })

  it('renders only a typed miniature with caller-localized accessibility text', () => {
    render(
      <ArtifactPreviewThumbnail
        ariaLabel="deployment-report.md preview"
        artifact={makeArtifact('/tmp/deployment-report.md')}
      />
    )

    expect(screen.getByRole('img', { name: 'deployment-report.md preview' })).toBeTruthy()
    expect(screen.getByText('MD')).toBeTruthy()
  })
})
