import { parseSlashCommand } from '@hermes/shared'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { ComposerAttachment } from '@/store/composer'

import { expandPastedSlashArgs, hasNonPasteAttachment, isPastedAttachment, trimSlashCommandText } from './slash-paste'

const PASTE_PATH_A = '/home/user/.hermes/composer-pastes/pasted_content_2026-11-01_12-00-00-000_aaaaaa.txt'
const PASTE_PATH_B = '/home/user/.hermes/composer-pastes/pasted_content_2026-11-01_12-00-01-000_bbbbbb.txt'

function makePasteChip(path = PASTE_PATH_A, overrides: Partial<ComposerAttachment> = {}): ComposerAttachment {
  return {
    id: `file:${path}`,
    kind: 'file',
    label: 'Pasted content (4.9 KB)',
    path,
    refText: `@file:${path}`,
    titlePreview: 'preview-1000-chars-must-never-be-used',
    ...overrides
  }
}

describe('expandPastedSlashArgs', () => {
  const readFileText = vi.fn()

  beforeEach(() => {
    readFileText.mockReset()
    Object.defineProperty(window, 'hermesDesktop', {
      configurable: true,
      value: { readFileText }
    })
  })

  afterEach(() => {
    vi.restoreAllMocks()
  })

  it('recognizes only composer-paste file chips as paste attachments', () => {
    expect(isPastedAttachment(makePasteChip())).toBe(true)
    expect(
      isPastedAttachment({ id: 'file:/tmp/notes.txt', kind: 'file', label: 'notes.txt', path: '/tmp/notes.txt' })
    ).toBe(false)
    expect(isPastedAttachment({ id: 'image:1', kind: 'image', label: 'shot.png', path: PASTE_PATH_A })).toBe(false)
    expect(isPastedAttachment({ id: 'folder:1', kind: 'folder', label: 'src', path: '/work/src' })).toBe(false)
    expect(isPastedAttachment({ id: 'url:1', kind: 'url', label: 'https://example.com' })).toBe(false)
    expect(isPastedAttachment({ id: 'term:1', kind: 'terminal', label: 'zsh:1-10', detail: 'ls' })).toBe(false)
    expect(hasNonPasteAttachment([makePasteChip()])).toBe(false)
    expect(
      hasNonPasteAttachment([
        makePasteChip(),
        { id: 'file:/tmp/notes.txt', kind: 'file', label: 'notes.txt', path: '/tmp/notes.txt' }
      ])
    ).toBe(true)
  })

  it('inserts a newline before the paste when the command text does not end in whitespace so the command name stays intact', async () => {
    const body = '  line one of pasted spec\nline two with trailing spaces  '
    readFileText.mockResolvedValueOnce({
      binary: false,
      byteSize: body.length,
      path: PASTE_PATH_A,
      text: body,
      truncated: false
    })

    const result = await expandPastedSlashArgs('/goal', [makePasteChip()])

    expect(readFileText).toHaveBeenCalledExactlyOnceWith(PASTE_PATH_A)
    expect(result).toEqual({
      ok: true,
      text: `/goal\n${body}`
    })
    expect(parseSlashCommand(result.ok ? result.text : '').name).toBe('goal')
  })

  it('concatenates two paste chips in chip order with a newline between them without trimming either body', async () => {
    const first = 'first paste block'
    const second = '  second paste block  '
    readFileText
      .mockResolvedValueOnce({ path: PASTE_PATH_A, text: first })
      .mockResolvedValueOnce({ path: PASTE_PATH_B, text: second })

    const result = await expandPastedSlashArgs('/goal', [makePasteChip(PASTE_PATH_A), makePasteChip(PASTE_PATH_B)])

    expect(readFileText.mock.calls).toEqual([[PASTE_PATH_A], [PASTE_PATH_B]])
    expect(result).toEqual({
      ok: true,
      text: `/goal\n${first}\n${second}`
    })
  })

  it.each([
    ['a real file chip', { id: 'file:/tmp/a.txt', kind: 'file' as const, label: 'a.txt', path: '/tmp/a.txt' }],
    ['an image chip', { id: 'img:1', kind: 'image' as const, label: 'a.png', path: '/tmp/a.png' }],
    ['a folder chip', { id: 'dir:1', kind: 'folder' as const, label: 'dir', path: '/tmp/dir' }],
    ['a url chip', { id: 'url:1', kind: 'url' as const, label: 'https://example.com' }],
    ['a terminal chip', { id: 'term:1', kind: 'terminal' as const, label: 'zsh:1-5', detail: 'echo hi' }]
  ])('refuses without reading disk when %s is present', async (_label, nonPasteChip) => {
    const result = await expandPastedSlashArgs('/goal', [makePasteChip(), nonPasteChip])

    expect(result).toEqual({ ok: false, reason: 'non-paste-attachment' })
    expect(readFileText).not.toHaveBeenCalled()
  })

  it.each([
    ['truncated: true', () => Promise.resolve({ path: PASTE_PATH_A, text: 'cut at 512 KiB', truncated: true })],
    ['binary: true', () => Promise.resolve({ binary: true, path: PASTE_PATH_A, text: 'raw', truncated: false })],
    [
      '{ ok: false }',
      () => Promise.resolve({ error: 'ENOENT', message: 'File does not exist', ok: false as const, path: PASTE_PATH_A })
    ],
    ['a thrown error', () => Promise.reject(new Error('IPC read failed'))]
  ])('refuses when readFileText returns %s and never falls back to titlePreview', async (_label, mockImpl) => {
    readFileText.mockImplementationOnce(mockImpl)

    const result = await expandPastedSlashArgs('/goal', [makePasteChip()])

    expect(result).toEqual({ ok: false, reason: 'read-failed' })
  })
})

describe('trimSlashCommandText', () => {
  it('trims a one-line command at both ends', () => {
    expect(trimSlashCommandText('  /goal ship it  ')).toBe('/goal ship it')
  })

  it('keeps the trailing whitespace of a multiline paste body', () => {
    expect(trimSlashCommandText('  /goal\nbody with trailing space   \n')).toBe('/goal\nbody with trailing space   \n')
  })
})
