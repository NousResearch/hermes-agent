import { afterEach, describe, expect, it } from 'vitest'

import { setRuntimeI18nLocale } from '@/i18n'

import {
  buildToolView,
  clampForDisplay,
  countDiffLineStats,
  inlineDiffFromResult,
  MAX_TOOL_RENDER_CHARS,
  type ToolPart
} from './fallback-model'

const part = (overrides: Partial<ToolPart>): ToolPart => ({
  args: {},
  isError: false,
  result: {},
  toolCallId: 'call_1',
  toolName: 'vision_analyze',
  type: 'tool-call',
  ...overrides
})

afterEach(() => {
  setRuntimeI18nLocale('en')
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
  const terminal = (result: Record<string, unknown>) => buildToolView(part({ result, toolName: 'terminal' }), '')

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

describe('buildToolView browser_navigate title', () => {
  it('shows failed title when navigate returns success=false', () => {
    const view = buildToolView(
      part({
        toolName: 'browser_navigate',
        args: { url: 'https://hermes-agent.nousresearch.com/docs' },
        result: { success: false, error: 'Command timed out after 60 seconds' }
      }),
      ''
    )

    expect(view.status).toBe('error')
    expect(view.title).toBe('Failed to open hermes-agent.nousresearch.com')
  })

  it('shows opened title on success', () => {
    const view = buildToolView(
      part({
        toolName: 'browser_navigate',
        args: { url: 'https://hermes-agent.nousresearch.com/docs' },
        result: { success: true, url: 'https://hermes-agent.nousresearch.com/docs', title: 'Docs' }
      }),
      ''
    )

    expect(view.status).toBe('success')
    expect(view.title).toBe('Opened hermes-agent.nousresearch.com')
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

describe('buildToolView title actions', () => {
  it('marks the pending action separately from the rest of the title', () => {
    const read = buildToolView(part({ args: { path: '/tmp/demo.txt' }, result: undefined, toolName: 'read_file' }), '')

    const web = buildToolView(
      part({ args: { url: 'https://example.com/docs' }, result: undefined, toolName: 'web_extract' }),
      ''
    )

    const terminal = buildToolView(
      part({ args: { command: 'npm test -- --runInBand' }, result: undefined, toolName: 'terminal' }),
      ''
    )

    const code = buildToolView(
      part({ args: { code: 'print("hello")' }, result: undefined, toolName: 'execute_code' }),
      ''
    )

    expect(read.title).toBe('Reading demo.txt')
    expect(read.titleAction).toEqual({ prefix: '', text: 'Reading', suffix: ' demo.txt' })
    expect(web.title).toBe('Reading example.com/docs')
    expect(web.titleAction).toEqual({ prefix: '', text: 'Reading', suffix: ' example.com/docs' })
    expect(terminal.title).toBe('Running npm test -- --runInBand')
    expect(terminal.titleAction).toEqual({ prefix: '', text: 'Running', suffix: ' npm test -- --runInBand' })
    expect(code.title).toBe('Scripting print("hello")')
    expect(code.titleAction).toEqual({ prefix: '', text: 'Scripting', suffix: ' print("hello")' })
  })

  it('does not mark completed tool titles as pending actions', () => {
    const view = buildToolView(part({ args: { url: 'https://example.com/docs' }, toolName: 'web_extract' }), '')

    expect(view.title).toBe('Read example.com/docs')
    expect(view.titleAction).toBeUndefined()
  })

  it('uses the filename for completed read_file rows', () => {
    const view = buildToolView(
      part({ args: { path: './package.json' }, result: { content: '1|{"name":"demo"}' }, toolName: 'read_file' }),
      ''
    )

    expect(view.title).toBe('Read package.json')
    expect(view.subtitle).toBe('')
    expect(view.titleAction).toBeUndefined()
  })

  it('adds a compact line range to line-scoped read_file rows', () => {
    const view = buildToolView(
      part({
        args: { limit: 10, offset: 25, path: './src/main.ts' },
        result: { content: '25|function toggleDock() {\n26|  dock.classList.toggle("hidden");\n34|}' },
        toolName: 'read_file'
      }),
      ''
    )

    expect(view.title).toBe('Read main.ts L25-34')
    expect(view.subtitle).toBe('')
  })

  it('uses the requested positive offset/limit for read_file row line ranges', () => {
    const view = buildToolView(
      part({
        args: { limit: 5, offset: 1, path: './package.json' },
        result: {
          content:
            '1|{\n2|  "name": "bb-rainbows",\n3|  "private": true,\n4|  "version": "0.0.1",\n5|  "type": "module",\n6|  "description": "extra"'
        },
        toolName: 'read_file'
      }),
      ''
    )

    expect(view.title).toBe('Read package.json L1-5')
  })

  it('uses inherited backend context for live read_file rows', () => {
    const view = buildToolView(
      part({
        args: { context: 'package.json L1-5', path: './package.json' },
        result: undefined,
        toolName: 'read_file'
      }),
      ''
    )

    expect(view.title).toBe('Reading package.json L1-5')
    expect(view.titleAction).toEqual({ prefix: '', text: 'Reading', suffix: ' package.json L1-5' })
  })

  it('uses returned line numbers for negative-offset read_file rows', () => {
    const view = buildToolView(
      part({
        args: { limit: 2, offset: -2, path: './src/main.ts' },
        result: { content: '99|lastLine();\n100|done();' },
        toolName: 'read_file'
      }),
      ''
    )

    expect(view.title).toBe('Read main.ts L99-100')
  })

  it('renders compact terminal titles for session 20260624_231846_bdbd1e commands', () => {
    const rows = [
      [
        'cd /Users/brooklyn/www/bb-rainbows && pnpm run lint 2>&1 | tail -20; echo "lint_exit=${PIPESTATUS[0]}"',
        'Ran pnpm run lint'
      ],
      [
        'cd /Users/brooklyn/www/bb-rainbows && pnpm run build 2>&1 | tail -20; echo "build_exit=${PIPESTATUS[0]}"',
        'Ran pnpm run build'
      ],
      [
        'which node pnpm corepack; node -v; echo "---"; corepack --version 2>&1; echo "---pnpm via corepack---"; pnpm --version 2>&1 | tail -5',
        'Ran which node pnpm corepack + 3 commands'
      ],
      [
        'echo "--- proto pnpm direct ---"; ~/.proto/tools/node/24.11.0/bin/pnpm --version 2>&1 | tail -3; echo "--- proto node ---"; ls ~/.proto/tools/node/ 2>&1; echo "--- corepack cache ---"; ls ~/.cache/node/corepack/v1/pnpm/ 2>&1',
        'Ran ~/.proto/tools/node/24.11.0/bin/pnpm --version + 2 commands'
      ],
      [
        'cd /Users/brooklyn/www/bb-rainbows && COREPACK_ENABLE_DOWNLOAD_PROMPT=0 corepack pnpm@10.20.0 --version 2>&1 | tail -3',
        'Ran COREPACK_ENABLE_DOWNLOAD_PROMPT=0 corepack pnpm@10.20.0 --version'
      ],
      [
        'cd /Users/brooklyn/www/bb-rainbows && COREPACK_ENABLE_DOWNLOAD_PROMPT=0 corepack use pnpm@10.20.0 2>&1 | tail -10; echo "exit=$?"',
        'Ran COREPACK_ENABLE_DOWNLOAD_PROMPT=0 corepack use pnpm@10.20.0'
      ]
    ] as const

    for (const [command, expectedTitle] of rows) {
      const view = buildToolView(
        part({ args: { command }, result: { output: 'ok', exit_code: 0 }, toolName: 'terminal' }),
        ''
      )

      expect(view.title).toBe(expectedTitle)
    }
  })

  it('retains the full terminal command for the renderer', () => {
    const view = buildToolView(
      part({ args: { command: 'cd /tmp && pnpm run lint' }, result: { output: 'done' }, toolName: 'terminal' }),
      ''
    )

    expect(view.command).toBe('cd /tmp && pnpm run lint')
    expect(view.title).toBe('done')
  })

  it('uses the runtime locale for title text and action placement', () => {
    setRuntimeI18nLocale('ja')

    const read = buildToolView(part({ args: { path: '/tmp/demo.txt' }, result: undefined, toolName: 'read_file' }), '')

    const web = buildToolView(
      part({ args: { url: 'https://example.com/docs' }, result: undefined, toolName: 'web_extract' }),
      ''
    )

    expect(read.title).toBe('demo.txt を読み取り中')
    expect(read.titleAction).toEqual({ prefix: 'demo.txt を', text: '読み取り中', suffix: '' })
    expect(web.title).toBe('example.com/docs を読み取り中')
    expect(web.titleAction).toEqual({ prefix: 'example.com/docs を', text: '読み取り中', suffix: '' })
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
    expect(countDiffLineStats(`--- a/x\n+++ b/x\n@@\n-old\n+new\n context\n+another`)).toEqual({ added: 2, removed: 1 })
  })
})
