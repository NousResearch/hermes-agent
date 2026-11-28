import { describe, expect, it } from 'vitest'

import { parseTranscriptDocument } from './transcript-document'

describe('transcript document', () => {
  it('accepts preserved tool identities and rejects unsafe or excluded identities', () => {
    const doc = {
      version: 1, session_id: 's', title: 'T', exported_at: 'now',
      options: { include_tools: true, include_reasoning: false, include_images: true, include_videos: true,
        include_files: true, include_links: true, include_asides: true, collapse_details: true },
      messages: [
        { role: 'assistant', speaker: 'assistant', text: '', tool_calls: [
          { name: 'shell', arguments: '{}', id: 'same', call_id: 'provider', uid: 'uid-one' },
          { name: 'shell', arguments: '{}' }
        ] },
        { role: 'tool', speaker: 'shell', text: 'result', tool_call_id: 'provider', tool_call_uid: 'uid-one' }
      ], warnings: []
    }

    expect(parseTranscriptDocument(JSON.stringify(doc)).messages).toEqual(doc.messages)

    for (const bad of ['', 'x\n', 'x'.repeat(513), 5, null]) {
      const copy = structuredClone(doc) as any

      copy.messages[0].tool_calls[0].uid = bad
      expect(() => parseTranscriptDocument(JSON.stringify(copy))).toThrow()
      copy.messages[0].tool_calls[0].uid = 'uid-one'
      copy.messages[1].tool_call_id = bad
      expect(() => parseTranscriptDocument(JSON.stringify(copy))).toThrow()
    }

    const excluded = structuredClone(doc) as any

    excluded.options.include_tools = false
    excluded.messages = [{ role: 'assistant', speaker: 'assistant', text: '', tool_call_id: 'leak' }]
    expect(() => parseTranscriptDocument(JSON.stringify(excluded))).toThrow()
  })
  it('rejects noncanonical roles without normalizing them', () => {
    const doc = {
      version: 1,
      session_id: 's',
      title: 'T',
      exported_at: 'now',
      options: {
        include_tools: false,
        include_reasoning: false,
        include_images: true,
        include_videos: true,
        include_files: true,
        include_links: true,
        include_asides: true,
        collapse_details: true
      },
      messages: [{ role: 'tool ', speaker: 'shell', text: 'private' }],
      warnings: []
    }

    for (const role of ['tool ', ' Tool', 'TOOL', 'unknown', 'assistant ', 'system ']) {
      doc.messages[0].role = role
      expect(() => parseTranscriptDocument(JSON.stringify(doc)), role).toThrow()
    }
  })
  it('accepts canonical turns including system-speaker notices and permitted tools', () => {
    const doc = {
      version: 1,
      session_id: 's',
      title: 'T',
      exported_at: 'now',
      options: {
        include_tools: true,
        include_reasoning: false,
        include_images: true,
        include_videos: true,
        include_files: true,
        include_links: true,
        include_asides: true,
        collapse_details: true
      },
      messages: [
        { role: 'user', speaker: 'user', text: 'hello' },
        { role: 'assistant', speaker: 'assistant', text: 'hi' },
        { role: 'tool', speaker: 'shell', text: 'result' },
        { role: 'assistant', speaker: 'system', kind: 'notice', text: 'notice' }
      ],
      warnings: []
    }

    expect(parseTranscriptDocument(JSON.stringify(doc)).messages).toEqual(doc.messages)
  })
  it('bounds a gateway response before attempting to parse it', () => {
    expect(() => parseTranscriptDocument(' '.repeat(32 * 1024 * 1024 + 1))).toThrow('Transcript too large')
  })
  it('rejects system prompt turns in readable transcripts', () => {
    const doc = {
      version: 1,
      session_id: 's',
      title: 'T',
      exported_at: '2026-01-01T00:00:00Z',
      options: {
        include_tools: false,
        include_reasoning: false,
        include_images: true,
        include_videos: true,
        include_files: true,
        include_links: true,
        include_asides: true,
        collapse_details: true
      },
      messages: [{ role: 'system', speaker: 'system', text: 'private prompt' }],
      warnings: []
    }

    expect(() => parseTranscriptDocument(JSON.stringify(doc))).toThrow()
  })
  it('rejects excluded tool turns from an untrusted backend', () => {
    const doc = {
      version: 1,
      session_id: 's',
      title: 'T',
      exported_at: '2026-01-01T00:00:00Z',
      options: {
        include_tools: false,
        include_reasoning: false,
        include_images: true,
        include_videos: true,
        include_files: true,
        include_links: true,
        include_asides: true,
        collapse_details: true
      },
      messages: [{ role: 'tool', speaker: 'tool', text: 'secret' }],
      warnings: []
    }

    expect(() => parseTranscriptDocument(JSON.stringify(doc))).toThrow()
  })
})
