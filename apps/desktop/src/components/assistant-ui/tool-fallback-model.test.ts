import { describe, expect, it } from 'vitest'

import { buildToolView, toolCopyPayload, type ToolPart } from './tool-fallback-model'

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

describe('buildToolView secret redaction', () => {
  it('redacts token-shaped values from terminal display and copy payloads', () => {
    const fakeToken = 'fake_plex_token_for_tool_card_123456789'

    const tool = part({
      args: {
        command: `PLEX_TOKEN='${fakeToken}' curl 'http://localhost:32400/library?X-Plex-Token=${fakeToken}'`
      },
      result: {
        output: `Authorization: Bearer ${fakeToken}`,
        stderr: `token=${fakeToken}`,
        stdout: `http://localhost:32400/status?X-Plex-Token=${fakeToken}`
      },
      toolName: 'terminal'
    })

    const view = buildToolView(tool, '')
    const copy = toolCopyPayload(tool, view)
    const rendered = JSON.stringify({ view, copy })

    expect(rendered).not.toContain(fakeToken)
    expect(view.title).toContain('[REDACTED]')
    expect(view.detail).toContain('[REDACTED]')
    expect(view.rawArgs).toContain('[REDACTED]')
    expect(view.rawResult).toContain('[REDACTED]')
    expect(copy.text).toContain('[REDACTED]')
  })
})
