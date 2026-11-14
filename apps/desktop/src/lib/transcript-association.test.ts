import { strFromU8, unzipSync } from 'fflate'
import { expect, it } from 'vitest'

import { transcriptLayout } from './transcript-association'
import { DEFAULT_TRANSCRIPT_OPTIONS, type TranscriptDocument } from './transcript-document'
import type { TranscriptTurn } from './transcript-document'
import { renderDocx } from './transcript-docx'
import { renderHtml } from './transcript-html'
import { renderMarkdown } from './transcript-markdown'
import { renderOdt } from './transcript-odt'
import { renderTranscript } from './transcript-render'

const part = (s: string, name: string) => strFromU8(unzipSync(Uint8Array.from(atob(s), c => c.charCodeAt(0)))[name])

it('cross-references a uniquely identified interleaved result without changing chronology in every format', () => {
  const doc: TranscriptDocument = { version: 1, session_id: 's', title: 't', exported_at: 'now', warnings: [], options: { ...DEFAULT_TRANSCRIPT_OPTIONS, include_tools: true }, messages: [
    { role: 'assistant', speaker: 'Hermes', text: 'CALLING', tool_calls: [{ name: 'run', arguments: 'ARGS', id: 'id', uid: 'unique' }] },
    { role: 'user', speaker: 'User', text: 'INTERVENING' },
    { role: 'tool', speaker: 'run', text: 'RESULT', tool_call_id: 'id', tool_call_uid: 'unique' }
  ] }
  for (const output of [renderHtml(doc), renderMarkdown(doc), part(renderDocx(doc), 'word/document.xml'), part(renderOdt(doc), 'content.xml')]) {
    expect(output).toContain('Assistant turn 1')
    expect(output).toContain('Result for assistant turn 1, call 1')
    expect(output.indexOf('CALLING')).toBeLessThan(output.indexOf('INTERVENING'))
    expect(output.indexOf('INTERVENING')).toBeLessThan(output.indexOf('RESULT'))
  }
  const xml = new DOMParser().parseFromString(part(renderDocx(doc), 'word/document.xml'), 'application/xml')
  const ns = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
  const resultHeading = [...xml.getElementsByTagNameNS(ns, 'p')].find(p => p.textContent?.startsWith('Tool result —'))
  expect(resultHeading?.textContent).toContain('Result for assistant turn 1, call 1')
})

it('rejects duplicate and conflicting identities, including cross-lineage collisions', () => {
  const assistant = (calls: NonNullable<TranscriptTurn['tool_calls']>): TranscriptTurn => ({ role: 'assistant', speaker: 'a', text: '', tool_calls: calls })
  const result = (id?: string, uid?: string): TranscriptTurn => ({ role: 'tool', speaker: 'run', text: 'literal', ...(id ? { tool_call_id: id } : {}), ...(uid ? { tool_call_uid: uid } : {}) })
  const call = (id?: string, uid?: string) => ({ name: 'run', arguments: '{}', ...(id ? { id } : {}), ...(uid ? { uid } : {}) })
  const fallback = 'Unassociated tool result (original position)'
  const cases: Array<[TranscriptTurn[], string[]]> = [
    [[assistant([call('same', 'u1'), call('same', 'u2')]), result('same', 'u2'), result('same', 'u1')], ['Result for assistant turn 1, call 2', 'Result for assistant turn 1, call 1']],
    [[assistant([call('same'), call('same')]), result('same')], [fallback]],
    [[assistant([call('same')]), assistant([call('same')]), result('same')], [fallback]],
    [[assistant([call('x', 'u')]), result('other', 'u')], [fallback]],
    [[assistant([call('x', 'u')]), result('x', 'wrong')], [fallback]],
    [[assistant([call('x')]), result()], [fallback]],
    [[assistant([call('x', 'u')]), result('x', 'u'), result('x')], [fallback, fallback]],
    [[assistant([call('x', 'u')]), result(undefined, 'u'), result(undefined, 'u')], [fallback, fallback]],
    [[assistant([call('x')]), result('x')], ['Result for assistant turn 1, call 1']],
    [[assistant([{ name: 'run', arguments: '', id: 'response-item', call_id: 'call-key|item' }]), result('call-key')], ['Result for assistant turn 1, call 1']],
    [[assistant([call('x', 'duplicate'), call('y', 'duplicate')]), result('x', 'duplicate')], [fallback]],
    [[assistant([call('x', 'duplicate'), call('y', 'duplicate')]), result('x')], [fallback]],
    [[assistant([call(undefined, 'u'), call('x', 'v')]), result('x', 'u')], [fallback]],
    [[result('x'), assistant([call('x')])], [fallback]]
  ]
  for (const [messages, expected] of cases) {
    const doc: TranscriptDocument = { version: 1, session_id: 's', title: 't', exported_at: 'now', warnings: [], options: { ...DEFAULT_TRANSCRIPT_OPTIONS, include_tools: true }, messages }
    expect(transcriptLayout(doc).associations.filter((_, i) => messages[i].role === 'tool')).toEqual(expected)
  }
})

it('renders multi-call links and honest fallback without altering JSON', () => {
  const doc: TranscriptDocument = { version: 1, session_id: 's', title: 't', exported_at: 'now', warnings: [], options: { ...DEFAULT_TRANSCRIPT_OPTIONS, include_tools: true }, messages: [
    { role: 'assistant', speaker: 'a', text: 'A', tool_calls: [{ name: 'run', arguments: 'FIRST', id: 'same', uid: 'one' }, { name: 'run', arguments: 'SECOND', id: 'same', uid: 'two' }] },
    { role: 'tool', speaker: 'run', text: 'RESULT-TWO', tool_call_id: 'same', tool_call_uid: 'two' },
    { role: 'tool', speaker: 'run', text: 'RESULT-ONE', tool_call_id: 'same', tool_call_uid: 'one' },
    { role: 'tool', speaker: 'run', text: 'UNKNOWN' }
  ] }
  const original = JSON.stringify(doc)
  for (const output of [renderHtml(doc), renderMarkdown(doc), part(renderDocx(doc), 'word/document.xml'), part(renderOdt(doc), 'content.xml')]) {
    expect(output).toContain('Call 1')
    expect(output).toContain('Call 2')
    expect(output).toContain('Result for assistant turn 1, call 2')
    expect(output).toContain('Result for assistant turn 1, call 1')
    expect(output).toContain('Unassociated tool result (original position)')
    expect(output.indexOf('RESULT-TWO')).toBeLessThan(output.indexOf('RESULT-ONE'))
    expect(output.indexOf('RESULT-ONE')).toBeLessThan(output.indexOf('UNKNOWN'))
  }
  expect(JSON.stringify(doc)).toBe(original)
  expect(JSON.parse(renderTranscript(doc, 'json').content)).toEqual(doc)
})
