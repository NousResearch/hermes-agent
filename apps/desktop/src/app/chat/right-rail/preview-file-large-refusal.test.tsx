import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { LocalFilePreview } from './preview-file'

/**
 * The large-file refusal has to describe the file it is refusing.
 *
 * Regression: the gate fired on the size the pane had just read, while the
 * sentence printed `target.byteSize` — the size captured when the tab was
 * opened. Opening a tab on a file at 129 KB and letting it grow past 512 KB
 * produced "suggestions.json is 129 KB. Hermes will only show the first
 * 512 KB." for a 3.4 MB file: the two numbers in one dialog contradicting each
 * other.
 */

const { readDesktopFileText } = vi.hoisted(() => ({ readDesktopFileText: vi.fn() }))

vi.mock('@/lib/desktop-fs', () => ({
  desktopFileDiff: vi.fn(async () => ''),
  desktopFsCacheKey: () => 'local:',
  desktopGitRoot: vi.fn(async () => null),
  isDesktopFsRemoteMode: () => false,
  readDesktopFileDataUrl: vi.fn(async () => ''),
  readDesktopFileText: (...args: unknown[]) => readDesktopFileText(...args),
  writeDesktopFileText: vi.fn()
}))

const FILE = '/mnt/data/projects/calibre-tag-audit/suggestions.json'
// 132,096 B is what `formatBytes` renders as "129 KB"; 3,595,108 B = the same
// file once the run that rewrote it finished.
const OPENED_AT_BYTES = 132_096
const MEASURED_BYTES = 3_595_108

function fileTarget(overrides: Partial<Parameters<typeof LocalFilePreview>[0]['target']> = {}) {
  return {
    kind: 'file' as const,
    label: 'suggestions.json',
    path: FILE,
    previewKind: 'text' as const,
    source: FILE,
    url: `file://${FILE}`,
    ...overrides
  }
}

function readResult(byteSize: number) {
  return {
    binary: false,
    byteSize,
    language: 'json',
    path: FILE,
    text: '{"books":{"1":{"title":"A Game Of Thrones"',
    truncated: byteSize > 512 * 1024
  }
}

async function refusalSentence() {
  return screen.findByText(/Hermes will only show the first 512 KB/)
}

describe('LocalFilePreview large-file refusal', () => {
  afterEach(() => {
    cleanup()
    readDesktopFileText.mockReset()
  })

  it('names the size it measured, not the size recorded when the tab was opened', async () => {
    readDesktopFileText.mockResolvedValue(readResult(MEASURED_BYTES))

    render(<LocalFilePreview reloadKey={0} target={fileTarget({ byteSize: OPENED_AT_BYTES })} />)

    const sentence = await refusalSentence()

    expect(sentence.textContent).toMatch(/is 3(\.\d+)? MB\./)
    expect(sentence.textContent).not.toContain('129 KB')
  })

  it('still names the recorded size when the file is refused before it is read', async () => {
    render(<LocalFilePreview reloadKey={0} target={fileTarget({ byteSize: MEASURED_BYTES, large: true })} />)

    const sentence = await refusalSentence()

    expect(readDesktopFileText).not.toHaveBeenCalled()
    expect(sentence.textContent).toMatch(/is 3(\.\d+)? MB\./)
  })
})
