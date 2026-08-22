import { cleanup, render } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'

import { MarkdownTextContent } from './markdown-text'

// Render-level regression for the prose-safety preprocessor fixes that landed
// on 2026-08-22 (upstream #50871 lone-tilde strikethrough, #53953 unknown
// html-like tokens swallowing the message tail). The preprocessor unit tests
// in markdown-text.test.ts assert on the transformed string; these mount the
// REAL production component and assert on the rendered DOM, so a wrapper that
// stopped applying the preprocess cannot pass here.
afterEach(cleanup)

describe('markdown prose renders tilde ranges and unknown tags literally', () => {
  it.each([
    [
      'CJK range list',
      '分组：奇数 1~10,11~20，其余另列。',
      (text: string) => {
        expect(text).toContain('1~10,11~20')
      }
    ],
    [
      'approximation prefixes in one paragraph',
      '收益为 3~5 倍，成本约 ~¥0.089。',
      (text: string) => {
        expect(text).toContain('3~5')
        expect(text).toContain('~¥0.089')
      }
    ]
  ])('renders %s without a strikethrough element', (_label, source, assertText) => {
    const { container } = render(<MarkdownTextContent isRunning={false} text={source} />)

    expect(container.querySelectorAll('del, s').length, `no strikethrough for: ${source}`).toBe(0)
    assertText(container.textContent || '')
  })

  it('renders unknown html-like tokens as literal text and keeps the tail visible', () => {
    const text =
      'The proxy wraps calls in <tool_call> and results in <observation> blocks. ' +
      'Keep the rest visible after the tag. The message tail must NOT vanish.'
    const { container } = render(<MarkdownTextContent isRunning={false} text={text} />)

    expect(container.textContent).toContain('<tool_call>')
    expect(container.textContent).toContain('<observation>')
    expect(container.textContent).toContain('The message tail must NOT vanish.')
    expect(container.querySelector('tool_call')).toBeNull()
    expect(container.querySelector('observation')).toBeNull()
  })

  it('leaves math comparisons untouched', () => {
    const { container } = render(
      <MarkdownTextContent isRunning={false} text={'如果 a < b 且 2<3 则原样保留。'} />
    )

    expect(container.textContent).toContain('a < b')
    expect(container.textContent).toContain('2<3')
  })

  it('renders autolinks for urls (tilde-in-url is a streamdown remend quirk, asserted loosely)', () => {
    const { container } = render(
      <MarkdownTextContent isRunning={false} text={'Docs at https://example.com/a_b/c~d/page'} />
    )

    const link = container.querySelector('a')

    expect(link).not.toBeNull()
    // Streamdown's tailBoundedRemend escapes `~` even inside autolinks
    // (pre-existing library behaviour, unchanged by this fix) — assert the link
    // itself mounted and points at the host, not byte-exact URL round-trip.
    expect(link?.getAttribute('href')).toContain('example.com')
  })

  it('still renders intentional ~~strikethrough~~', () => {
    const { container } = render(<MarkdownTextContent isRunning={false} text={'保留 ~~deleted~~ 删除线。'} />)

    expect(container.querySelectorAll('del, s').length).toBeGreaterThan(0)
    expect(container.textContent).toContain('deleted')
  })
})
