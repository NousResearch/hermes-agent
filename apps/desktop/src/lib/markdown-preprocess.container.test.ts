import { describe, expect, it } from 'vitest'

import { preprocessMarkdown } from './markdown-preprocess'

describe('preprocessMarkdown container fences', () => {
  it.each([
    ['blockquote', ['> ```ts', '> const value = 1;', '> ```'].join('\n')],
    ['unordered list', ['- ```ts', '  const value = 1;', '  ```'].join('\n')],
    ['ordered list', ['1. ```ts', '   const value = 1;', '   ```'].join('\n')]
  ])('keeps valid %s container fences intact', (_label, input) => {
    const output = preprocessMarkdown(input)

    expect(output).toContain('```ts')
    expect(output).toContain('const value = 1;')
    expect(output).toContain('```')
  })

  it.each([
    [
      'list then blockquote',
      ['- > ```ts', '  > const value = 1;', '  > ```'].join('\n'),
      ['- > ```ts', '  > const value = 1;', '- > ```'].join('\n')
    ],
    [
      'blockquote then list',
      ['> - ```ts', '>   const value = 1;', '>   ```'].join('\n'),
      ['> - ```ts', '>   const value = 1;', '> - ```'].join('\n')
    ],
    [
      'deeply nested containers',
      ['> - > 1. ```ts', '>   >   1. const value = 1;', '>   >   1. ```'].join('\n'),
      ['> - > 1. ```ts', '>   >   1. const value = 1;', '> - > 1. ```'].join('\n')
    ]
  ])('preserves %s fence marker and language', (_label, input, expected) => {
    expect(preprocessMarkdown(input)).toBe(expected)
  })
})
