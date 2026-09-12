import { describe, expect, it } from 'vitest'

import { decodePreviewTabs } from './preview'

describe('persisted preview migration', () => {
  it('upgrades a pre-PDF remote tab from binary to pdf', () => {
    const source = '/remote/.hermes/desktop-attachments/spec.pdf'

    const [restored] = decodePreviewTabs(
      JSON.stringify([
        {
          id: `file:file://${source}`,
          target: {
            binary: true,
            kind: 'file',
            label: 'spec.pdf',
            large: true,
            path: source,
            previewKind: 'binary',
            source,
            url: `file://${source}`
          }
        }
      ])
    )

    expect(restored?.target.previewKind).toBe('pdf')
  })

  it('leaves a persisted non-PDF binary tab unchanged', () => {
    const source = '/work/archive.zip'

    const [restored] = decodePreviewTabs(
      JSON.stringify([
        {
          id: `file:file://${source}`,
          target: {
            binary: true,
            kind: 'file',
            label: 'archive.zip',
            path: source,
            previewKind: 'binary',
            source,
            url: `file://${source}`
          }
        }
      ])
    )

    expect(restored?.target.previewKind).toBe('binary')
  })

  it.each(['report.pdf#notes', 'report.pdf?draft'])('treats %s as a literal filesystem path', sourceName => {
    const source = `/work/${sourceName}`

    const [restored] = decodePreviewTabs(
      JSON.stringify([
        {
          id: `file:file://${encodeURI(source)}`,
          target: {
            binary: true,
            kind: 'file',
            label: sourceName,
            path: source,
            previewKind: 'binary',
            source,
            url: `file:///work/${encodeURIComponent(sourceName)}`
          }
        }
      ])
    )

    expect(restored?.target.previewKind).toBe('binary')
  })

  it('does not overwrite a non-binary PDF preview kind', () => {
    const source = '/work/spec.pdf'

    const [restored] = decodePreviewTabs(
      JSON.stringify([
        {
          id: `file:file://${source}`,
          target: {
            kind: 'file',
            label: 'spec.pdf',
            path: source,
            previewKind: 'text',
            source,
            url: `file://${source}`
          }
        }
      ])
    )

    expect(restored?.target.previewKind).toBe('text')
  })

  // #95459's own reproduction: the owner is a RUNTIME session id, minted fresh
  // per restart (gateway/session_lifecycle.py). A persisted owner therefore
  // names a session that no longer exists, and openPreview's immutability rule
  // (existingOwner ?? ownerSessionId) would weld it in place forever — the
  // restored tab could never be re-owned by the bot's new runtime id, so the
  // exact reported repro still failed after the ownership fix.
  it('drops the persisted owner so a restored tab is re-stampable', () => {
    const [restored] = decodePreviewTabs(
      JSON.stringify([
        {
          id: 'url:browser-1',
          ownerSessionId: 'runtime-before-restart',
          target: { kind: 'url', label: 'Browser', source: 'https://example.com', url: 'https://example.com' }
        }
      ])
    )

    expect(restored?.ownerSessionId).toBeUndefined()
  })

  it('keeps the tab itself when the owner is dropped', () => {
    const restored = decodePreviewTabs(
      JSON.stringify([
        {
          id: 'url:browser-1',
          ownerSessionId: 'runtime-before-restart',
          target: { kind: 'url', label: 'Browser', source: 'https://example.com', url: 'https://example.com' }
        }
      ])
    )

    expect(restored).toHaveLength(1)
    expect(restored[0].id).toBe('url:browser-1')
    expect(restored[0].target.url).toBe('https://example.com')
  })

  it('rejects a row whose persisted owner is not a string', () => {
    const restored = decodePreviewTabs(
      JSON.stringify([
        {
          id: 'url:browser-1',
          ownerSessionId: 42,
          target: { kind: 'url', label: 'Browser', source: 'https://example.com', url: 'https://example.com' }
        }
      ])
    )

    expect(restored).toHaveLength(0)
  })
})
