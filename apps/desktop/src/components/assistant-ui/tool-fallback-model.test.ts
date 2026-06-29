import { describe, expect, it } from 'vitest'

import { buildToolView, type ToolPart } from './tool-fallback-model'

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

  // curl genuine failures (DNS/connect/timeout) no longer carry exit_code_meaning
  // (the backend stopped tagging them benign), so an empty-output curl failure
  // is flagged red instead of the prior false-green.
  it('treats a curl connect failure with no output as error', () => {
    expect(terminal({ exit_code: 7, output: '' }).status).toBe('error')
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

  // Backend-tagged benign nonzero exits (exit_code_meaning) are not failures
  // even with empty output — grep prints nothing when it finds nothing.
  it('treats backend-tagged benign nonzero exit (grep no-match) as success', () => {
    expect(
      terminal({ exit_code: 1, exit_code_meaning: 'No matches found (not an error)', output: '' }).status
    ).toBe('success')
  })

  // A user interrupt returns 130 with no tag — render it neutral, not red.
  it('treats user interrupt (exit 130) with no output as success', () => {
    expect(terminal({ exit_code: 130, output: '' }).status).toBe('success')
  })

  // SIGPIPE (141) — a downstream reader closed the pipe (`… | head` under
  // pipefail). Benign signal death, rendered neutral.
  it('treats SIGPIPE (exit 141) as success', () => {
    expect(terminal({ exit_code: 141, output: 'first lines...' }).status).toBe('success')
    expect(terminal({ exit_code: 141, output: '' }).status).toBe('success')
  })

  // The benign tag never overrides a populated error field (error wins).
  it('keeps a populated error field red even when exit_code_meaning is set', () => {
    expect(
      terminal({ error: 'grep: invalid option', exit_code: 1, exit_code_meaning: 'No matches found', output: '' })
        .status
    ).toBe('error')
  })
})
