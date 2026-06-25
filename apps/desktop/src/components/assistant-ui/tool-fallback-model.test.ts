import { describe, expect, it } from 'vitest'

import {
  buildToolView,
  clampForDisplay,
  countDiffLineStats,
  inlineDiffFromResult,
  MAX_TOOL_RENDER_CHARS,
  type ToolPart
} from './tool-fallback-model'

const part = (overrides: Partial<ToolPart>): ToolPart => ({
  args: {},
  isError: false,
  result: {},
  toolCallId: 'call_1',
  toolName: 'vision_analyze',
  type: 'tool-call',
  ...overrides
})

describe('buildToolView image handling', () => {
  // vision_analyze reports the input image as a local path; an <img> pointed at
  // a bare path resolves against the renderer origin and 404s, so we render the
  // tool codicon instead of a broken image.
  it('drops bare filesystem paths', () => {
    expect(buildToolView(part({ args: { path: '/Users/me/shot.png' } }), '').imageUrl).toBe('')
    expect(buildToolView(part({ result: { image_path: '/tmp/out.jpg' } }), '').imageUrl).toBe('')
  })

  it('keeps fetchable data URLs', () => {
    const dataUrl = 'data:image/png;base64,AAAA'

    expect(buildToolView(part({ result: { image_url: dataUrl } }), '').imageUrl).toBe(dataUrl)
  })

  it('keeps remote http(s) image URLs', () => {
    const url = 'https://example.com/pic.webp'

    expect(buildToolView(part({ result: { url } }), '').imageUrl).toBe(url)
  })
})

describe('buildToolView terminal exit-code status', () => {
  const terminal = (result: Record<string, unknown>) =>
    buildToolView(part({ result, toolName: 'terminal' }), '')

  // A non-zero exit code with real output is not a failure (grep no-match,
  // diff differences, piped commands surfacing the last stage's code, etc.) —
  // it should render as success so the card isn't painted red.
  it('treats non-zero exit with output as success', () => {
    expect(terminal({ exit_code: 7, output: 'node ... 5174 (LISTEN)' }).status).toBe('success')
    expect(terminal({ exit_code: 1, stdout: 'partial results' }).status).toBe('success')
  })

  // No output + non-zero exit is a genuine failure worth flagging.
  it('treats non-zero exit with no output as error', () => {
    expect(terminal({ exit_code: 127, output: '' }).status).toBe('error')
    expect(terminal({ exit_code: 1 }).status).toBe('error')
  })

  it('treats zero exit as success', () => {
    expect(terminal({ exit_code: 0, output: 'done' }).status).toBe('success')
  })

  // Explicit error signals still win regardless of output presence.
  it('keeps explicit error signals red even with output', () => {
    expect(terminal({ error: 'boom', exit_code: 0, output: 'partial' }).status).toBe('error')
    expect(buildToolView(part({ isError: true, result: { output: 'x' }, toolName: 'terminal' }), '').status).toBe(
      'error'
    )
  })
})

describe('buildToolView file edit diffs', () => {
  const patchDiff = '--- a/src/demo.ts\n+++ b/src/demo.ts\n@@ -1 +1 @@\n-old\n+new'

  it('reads inline_diff and diff fields from patch results', () => {
    expect(inlineDiffFromResult({ inline_diff: patchDiff })).toBe(patchDiff)
    expect(inlineDiffFromResult({ diff: patchDiff })).toBe(patchDiff)
  })

  it('suppresses raw patch args when a diff is available', () => {
    const view = buildToolView(
      part({
        args: { context: 'src/demo.ts', mode: 'replace', new_string: 'new', path: 'src/demo.ts' },
        result: { diff: patchDiff, success: true },
        toolName: 'patch'
      }),
      patchDiff
    )

    expect(view.title).toBe('demo.ts')
    expect(view.subtitle).toBe('src/demo.ts')
    expect(view.detail).toBe('')
    expect(view.inlineDiff).toBe(patchDiff)
  })

  it('shows path subtitle instead of patch args JSON while pending', () => {
    const view = buildToolView(
      part({
        args: { context: 'src/demo.ts', mode: 'replace', new_string: 'new', path: 'src/demo.ts' },
        result: undefined,
        toolName: 'patch'
      }),
      ''
    )

    expect(view.title).toBe('demo.ts')
    expect(view.subtitle).toBe('src/demo.ts')
    expect(view.detail).toBe('')
  })
})

describe('clampForDisplay', () => {
  it('passes short payloads through untouched', () => {
    expect(clampForDisplay('hello')).toBe('hello')
    expect(clampForDisplay('x'.repeat(MAX_TOOL_RENDER_CHARS))).toHaveLength(MAX_TOOL_RENDER_CHARS)
  })

  it('truncates oversized payloads and reports the omitted count', () => {
    const oversized = 'x'.repeat(MAX_TOOL_RENDER_CHARS + 5_000)
    const clamped = clampForDisplay(oversized)

    expect(clamped.length).toBeLessThan(oversized.length)
    expect(clamped.startsWith('x'.repeat(MAX_TOOL_RENDER_CHARS))).toBe(true)
    expect(clamped).toContain('5,000 more characters truncated')
    expect(clamped).toContain('Copy')
  })
})

// A large tool result (e.g. a 100KB read_file during a `/learn` run) must not
// be serialized into the rendered rawResult at full size — that JSON.stringify
// payload is what floods the renderer when many rows stack up.
describe('buildToolView caps serialized result size', () => {
  it('clamps rawResult for an oversized result', () => {
    const huge = 'y'.repeat(MAX_TOOL_RENDER_CHARS * 3)
    const view = buildToolView(part({ result: { content: huge }, toolName: 'read_file' }), '')

    expect(view.rawResult.length).toBeLessThanOrEqual(MAX_TOOL_RENDER_CHARS + 200)
    expect(view.rawResult).toContain('truncated')
  })
})

describe('countDiffLineStats', () => {
  it('counts added and removed lines', () => {
    expect(
      countDiffLineStats(`--- a/x\n+++ b/x\n@@\n-old\n+new\n context\n+another`)
    ).toEqual({ added: 2, removed: 1 })
  })
})
