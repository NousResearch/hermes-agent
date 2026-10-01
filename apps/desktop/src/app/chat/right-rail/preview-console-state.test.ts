import { describe, expect, it } from 'vitest'

import {
  createPreviewConsoleState,
  PREVIEW_CONSOLE_MAX_ENTRIES,
  PREVIEW_CONSOLE_MAX_MESSAGE_CHARS,
  PREVIEW_CONSOLE_MAX_SOURCE_CHARS,
  PREVIEW_CONSOLE_MAX_TOTAL_CHARS
} from './preview-console-state'

type PreviewLogs = ReturnType<ReturnType<typeof createPreviewConsoleState>['$logs']['get']>

const retainedChars = (logs: PreviewLogs) =>
  logs.reduce((total, entry) => total + entry.message.length + (entry.source?.length ?? 0), 0)

describe('preview console retention', () => {
  it('truncates one oversized message and source before retaining them', () => {
    const state = createPreviewConsoleState()

    state.append({
      level: 0,
      message: 'x'.repeat(PREVIEW_CONSOLE_MAX_MESSAGE_CHARS * 4),
      source: 's'.repeat(PREVIEW_CONSOLE_MAX_SOURCE_CHARS * 4)
    })

    const [entry] = state.$logs.get()

    expect(entry.message).toHaveLength(PREVIEW_CONSOLE_MAX_MESSAGE_CHARS)
    expect(entry.message).toMatch(/… \[truncated\]$/)
    expect(entry.source).toHaveLength(PREVIEW_CONSOLE_MAX_SOURCE_CHARS)
    expect(entry.source).toMatch(/… \[truncated\]$/)
  })

  it('does not split a UTF-16 surrogate pair at the truncation boundary', () => {
    const state = createPreviewConsoleState()
    const marker = '\n… [truncated]'
    const prefixChars = PREVIEW_CONSOLE_MAX_MESSAGE_CHARS - marker.length
    // The high surrogate begins exactly at the final retained code-unit slot;
    // enough tail follows to force truncation rather than exercising pass-through.
    const message = `${'x'.repeat(prefixChars - 1)}😀${'tail'.repeat(10)}`

    state.append({ level: 0, message })

    const [entry] = state.$logs.get()
    const beforeMarker = entry.message.slice(0, -marker.length)

    expect(message.length).toBeGreaterThan(PREVIEW_CONSOLE_MAX_MESSAGE_CHARS)
    expect(entry.message).toHaveLength(PREVIEW_CONSOLE_MAX_MESSAGE_CHARS)
    expect(entry.message.endsWith(marker)).toBe(true)
    expect(beforeMarker.endsWith('\ufffd')).toBe(true)
  })

  it('evicts oldest large entries and their selections to stay under the tab budget', () => {
    const state = createPreviewConsoleState()

    state.append({ level: 0, message: 'first'.repeat(PREVIEW_CONSOLE_MAX_MESSAGE_CHARS) })
    const selectedId = state.$logs.get()[0].id
    state.toggleSelection(selectedId)

    for (let index = 0; index < 20; index++) {
      state.append({ level: 0, message: `${index}`.repeat(PREVIEW_CONSOLE_MAX_MESSAGE_CHARS) })
    }

    const logs = state.$logs.get()

    expect(retainedChars(logs)).toBeLessThanOrEqual(PREVIEW_CONSOLE_MAX_TOTAL_CHARS)
    expect(logs.at(-1)?.id).toBe(21)
    expect(logs.some(entry => entry.id === selectedId)).toBe(false)
    expect(state.$selectedLogIds.get().has(selectedId)).toBe(false)
  })

  it('preserves the existing 200-entry cap for small logs', () => {
    const state = createPreviewConsoleState()

    for (let index = 0; index < PREVIEW_CONSOLE_MAX_ENTRIES + 10; index++) {
      state.append({ level: 0, message: `line-${index}` })
    }

    expect(state.$logs.get()).toHaveLength(PREVIEW_CONSOLE_MAX_ENTRIES)
    expect(state.$logs.get()[0].message).toBe('line-10')
    expect(retainedChars(state.$logs.get())).toBeLessThanOrEqual(PREVIEW_CONSOLE_MAX_TOTAL_CHARS)
  })
})
