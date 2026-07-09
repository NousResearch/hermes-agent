import { fireEvent, render } from '@testing-library/react'
import { createRef } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { CodeEditorApi } from './code-editor'

const originalCreateRange = document.createRange.bind(document)

function suppressCodeMirrorGeometryError(event: ErrorEvent) {
  if (String(event.error || event.message).includes('getClientRects is not a function')) {
    event.preventDefault()
  }
}

describe('CodeEditor', () => {
  beforeEach(() => {
    const rect = new DOMRect(0, 0, 0, 0)

    const rectList = {
      0: rect,
      item: (index: number) => (index === 0 ? rect : null),
      length: 1,
      [Symbol.iterator]: function* () {
        yield rect
      }
    } as DOMRectList

    Object.defineProperty(document, 'createRange', {
      configurable: true,
      value: vi.fn(() =>
        ({
          collapse: vi.fn(),
          commonAncestorContainer: document.body,
          getBoundingClientRect: vi.fn(() => rect),
          getClientRects: vi.fn(() => rectList),
          setEnd: vi.fn(),
          setStart: vi.fn()
        }) as unknown as Range
      )
    })
    vi.spyOn(console, 'error').mockImplementation((...args: unknown[]) => {
      if (String(args[0]).includes('getClientRects is not a function')) {
        return
      }

      throw new Error(args.map(String).join(' '))
    })
    window.addEventListener('error', suppressCodeMirrorGeometryError)
  })

  afterEach(() => {
    window.removeEventListener('error', suppressCodeMirrorGeometryError)
    vi.restoreAllMocks()
    Object.defineProperty(document, 'createRange', { configurable: true, value: originalCreateRange })
  })

  it('opens the native CodeMirror search panel with Ctrl/Cmd+F', async () => {
    const { CodeEditor } = await import('./code-editor')

    const rendered = render(
      <CodeEditor filePath="example.ts" initialValue="const browseros = true\nconsole.log(browseros)" onChange={vi.fn()} />
    )

    const content = rendered.container.querySelector('.cm-content')

    expect(content).toBeInstanceOf(HTMLElement)

    fireEvent.keyDown(content!, { ctrlKey: true, key: 'f' })

    expect(rendered.container.querySelector('.cm-search')).toBeTruthy()
  })

  it('opens the native CodeMirror search panel through the imperative API', async () => {
    const { CodeEditor } = await import('./code-editor')
    const apiRef = createRef<CodeEditorApi | null>()

    const rendered = render(
      <CodeEditor apiRef={apiRef} filePath="example.ts" initialValue="const browseros = true" onChange={vi.fn()} />
    )

    apiRef.current?.find()

    expect(rendered.container.querySelector('.cm-search')).toBeTruthy()
  })
})
