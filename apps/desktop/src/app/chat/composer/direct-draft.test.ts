import { describe, expect, it } from 'vitest'

import { createDirectDraftAdmission, directDraftIsValid } from './direct-draft'

describe('exact direct draft admission', () => {
  it('matches the gateway canonical UTF-8 digest for exact Unicode and whitespace', async () => {
    const admission = await createDirectDraftAdmission({ text: '  Unicode Ω\nsecond line  ', attachments: [] })

    expect(admission).toEqual({
      attachmentManifest: [],
      contextText: '  Unicode Ω\nsecond line  ',
      payloadDigest: '95219f53e0234f302ac3a97eae0aa7d1d28f78096e0a15cfbcaae626b6f9d15c',
      sourceText: '  Unicode Ω\nsecond line  '
    })
  })

  it('binds every ordered relayed attachment integrity field without paths', async () => {
    const admission = await createDirectDraftAdmission({
      text: 'inspect',
      attachments: [
        {
          id: 'file-1',
          kind: 'file',
          mediaType: 'text/plain',
          name: 'notes.txt',
          occurrenceId: 'occ-1',
          order: 0,
          provenance: { authorizedAt: 1234, kind: 'composer', occurrenceId: 'occ-1', sourceId: 'file-1' },
          refText: '@file:attachments/notes.txt',
          runtimeSessionId: 'runtime-1',
          sha256: 'a'.repeat(64),
          size: 5,
          storedName: 'notes-2.txt'
        }
      ]
    })

    expect(admission.contextText).toBe('@file:attachments/notes.txt\n\ninspect')
    expect(admission.attachmentManifest).toEqual([
      {
        id: 'file-1',
        kind: 'file',
        mediaType: 'text/plain',
        name: 'notes.txt',
        occurrenceId: 'occ-1',
        order: 0,
        refText: '@file:attachments/notes.txt',
        runtimeSessionId: 'runtime-1',
        sha256: 'a'.repeat(64),
        size: 5,
        storedName: 'notes-2.txt',
        sourceId: 'file-1'
      }
    ])
    expect(JSON.stringify(admission)).not.toMatch(/[A-Z]:\/|\\\\server|private/i)
    expect(admission.payloadDigest).toMatch(/^[0-9a-f]{64}$/)
  })

  it('uses the normal image-only fallback and rejects malformed or oversized drafts', async () => {
    await expect(
      createDirectDraftAdmission({
        text: '',
        attachments: [
          {
            id: 'image-1',
            kind: 'image',
            mediaType: 'image/png',
            name: 'image.png',
            occurrenceId: null,
            order: 0,
            provenance: { authorizedAt: 1, kind: 'composer', occurrenceId: null, sourceId: 'image-1' },
            runtimeSessionId: 'runtime-1',
            sha256: 'b'.repeat(64),
            size: 4,
            storedName: 'upload-1.png'
          }
        ]
      })
    ).resolves.toMatchObject({ contextText: 'What do you see in this image?' })
    expect(directDraftIsValid({ text: 'x'.repeat(64_001) })).toBe(false)
    expect(directDraftIsValid({ text: 'ok', attachments: [] })).toBe(true)
    expect(directDraftIsValid({ text: '', attachments: [] })).toBe(false)
  })
})
