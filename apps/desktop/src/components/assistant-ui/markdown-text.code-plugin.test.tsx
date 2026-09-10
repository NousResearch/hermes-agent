import { cleanup, render } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

const { codeModuleFactory, streamdownRenders } = vi.hoisted(() => ({
  codeModuleFactory: vi.fn(),
  streamdownRenders: vi.fn()
}))

vi.mock('@assistant-ui/react-streamdown', () => ({
  StreamdownTextPrimitive: (props: { components: Record<string, unknown>; plugins: unknown }) => {
    streamdownRenders(props)

    return null
  },
  tailBoundedRemend: (text: string) => text
}))

vi.mock('@streamdown/code', () => {
  codeModuleFactory()

  return { code: { name: 'unused-code-plugin' } }
})

import { MarkdownTextContent } from './markdown-text'

describe('MarkdownTextContent code highlighting', () => {
  afterEach(() => {
    cleanup()
    codeModuleFactory.mockClear()
    streamdownRenders.mockClear()
  })

  it.each([
    ['ordinary prose', 'Just an ordinary prose answer.'],
    ['top-level fenced code', '```ts\nconst answer = 42\n```'],
    ['quoted fenced code', '> ```ts\n> const answer = 42\n> ```']
  ])('uses the custom SyntaxHighlighter without importing @streamdown/code for %s', async (_label, text) => {
    render(<MarkdownTextContent isRunning={false} text={text} />)

    await vi.dynamicImportSettled()

    expect(codeModuleFactory).not.toHaveBeenCalled()
    expect(streamdownRenders).toHaveBeenCalledOnce()
    expect(streamdownRenders).toHaveBeenLastCalledWith(
      expect.objectContaining({
        components: expect.objectContaining({ SyntaxHighlighter: expect.any(Function) }),
        plugins: expect.not.objectContaining({ code: expect.anything() })
      })
    )
  })
})
