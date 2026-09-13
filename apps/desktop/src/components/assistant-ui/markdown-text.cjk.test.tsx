import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'

import { MarkdownTextContent } from './markdown-text'

afterEach(() => cleanup())

// Regression for #92814: CommonMark's Unicode-punctuation delimiter rules
// reject a `**` closing run when the preceding char is closing punctuation
// and the following char is a CJK letter (e.g. the Korean particle 는 right
// after a closing quote), so the markers leak into the transcript as literal
// text. The @streamdown/cjk plugin is wired into MarkdownTextSurface's plugin
// table to make those delimiters accepted.
describe('MarkdownTextContent CJK emphasis', () => {
  it('bolds a quoted Korean span followed by a particle', async () => {
    render(<MarkdownTextContent isRunning={false} text={'**“공개 문서는 점검한다”**는 원칙입니다.'} />)

    const strong = await screen.findByText('“공개 문서는 점검한다”')
    expect(strong.getAttribute('data-streamdown')).toBe('strong')
    expect(strong.parentElement).not.toBeNull()
    expect(document.body.textContent).not.toContain('**')
  })

  it('bolds CJK spans ending with parentheses or full-stop punctuation', async () => {
    render(
      <MarkdownTextContent
        isRunning={false}
        text={
          '**한국어 구문(괄호 포함)**을 강조합니다.\n**日本語の文章（括弧付き）。**この文が続きます。\n**中文文本（带括号）。**这句话继续。'
        }
      />
    )

    expect(await screen.findByText('한국어 구문(괄호 포함)')).toBeTruthy()
    expect(await screen.findByText('日本語の文章（括弧付き）。')).toBeTruthy()
    expect(await screen.findByText('中文文本（带括号）。')).toBeTruthy()
    expect(document.body.textContent).not.toContain('**')
  })

  it('does not regress plain emphasis', async () => {
    render(<MarkdownTextContent isRunning={false} text="**bold** and normal text" />)

    const strong = await screen.findByText('bold')
    expect(strong.getAttribute('data-streamdown')).toBe('strong')
  })
})
