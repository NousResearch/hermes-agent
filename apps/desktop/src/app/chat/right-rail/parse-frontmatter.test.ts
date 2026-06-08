import { describe, expect, it } from 'vitest'

import { parseFrontmatter } from './preview-file'

describe('parseFrontmatter', () => {
  it('returns null entries and original text when no frontmatter', () => {
    const result = parseFrontmatter('# Hello\n\nSome content')
    expect(result.entries).toBeNull()
    expect(result.body).toBe('# Hello\n\nSome content')
  })

  it('parses a simple frontmatter block', () => {
    const text = '---\ntitle: My Note\ntags: code, test\n---\n\n# Body'
    const result = parseFrontmatter(text)
    expect(result.entries).toEqual([
      ['title', 'My Note'],
      ['tags', 'code, test']
    ])
    expect(result.body).toBe('\n# Body')
  })

  it('handles Windows-style line endings (\\r\\n)', () => {
    const text = '---\r\ntitle: Windows Note\r\nauthor: Alice\r\n---\r\n\r\nBody text'
    const result = parseFrontmatter(text)
    expect(result.entries).toEqual([
      ['title', 'Windows Note'],
      ['author', 'Alice']
    ])
    expect(result.body).toBe('\r\nBody text')
  })

  it('handles multi-line values (indented continuation)', () => {
    const text = '---\ntitle: My Note\ndescription: This is a\n  multi-line description\n  that spans several lines\n---\n\nBody'
    const result = parseFrontmatter(text)
    expect(result.entries).toEqual([
      ['title', 'My Note'],
      ['description', 'This is a\nmulti-line description\nthat spans several lines']
    ])
  })

  it('handles empty frontmatter values', () => {
    const text = '---\ntitle:\ntags:\n---\n\nBody'
    const result = parseFrontmatter(text)
    expect(result.entries).toEqual([
      ['title', ''],
      ['tags', '']
    ])
  })

  it('returns null entries for malformed frontmatter (no closing ---)', () => {
    const text = '---\ntitle: Broken\nno closing delimiter'
    const result = parseFrontmatter(text)
    expect(result.entries).toBeNull()
    expect(result.body).toBe(text)
  })

  it('ignores frontmatter not at start of text', () => {
    const text = '# Title\n\n---\ntitle: Not frontmatter\n---\n\nBody'
    const result = parseFrontmatter(text)
    expect(result.entries).toBeNull()
    expect(result.body).toBe(text)
  })

  it('handles frontmatter with no body', () => {
    const text = '---\ntitle: Only Metadata\n---'
    const result = parseFrontmatter(text)
    expect(result.entries).toEqual([['title', 'Only Metadata']])
    expect(result.body).toBe('')
  })

  it('handles keys with underscores, dots, and hyphens', () => {
    const text = '---\nmy_key: value1\nsome.field: value2\ncamel-case: value3\n---\n\nBody'
    const result = parseFrontmatter(text)
    expect(result.entries).toEqual([
      ['my_key', 'value1'],
      ['some.field', 'value2'],
      ['camel-case', 'value3']
    ])
  })

  it('skips indented lines that are not continuation values', () => {
    const text = '---\ntitle: Test\n  - list item 1\n  - list item 2\nauthor: Bob\n---\n\nBody'
    const result = parseFrontmatter(text)
    expect(result.entries).toEqual([
      ['title', 'Test'],
      ['author', 'Bob']
    ])
  })

  it('preserves body content after frontmatter exactly', () => {
    const body = '\n# Heading\n\nParagraph with **bold** and `code`.\n\n```js\nconsole.log("hello")\n```'
    const text = `---\ntitle: Test\n---${body}`
    const result = parseFrontmatter(text)
    expect(result.body).toBe(body)
  })
})
