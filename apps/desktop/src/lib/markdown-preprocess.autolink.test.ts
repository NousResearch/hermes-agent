import { tailBoundedRemend } from '@assistant-ui/react-streamdown'
import remarkGfm from 'remark-gfm'
import remarkParse from 'remark-parse'
import { unified } from 'unified'
import { describe, expect, it } from 'vitest'

import { preprocessMarkdown } from './markdown-preprocess'

/**
 * The bare-URL autolinker wraps prose URLs in `<…>`. It must leave URLs that
 * markdown syntax already owns alone: wrapping the label of `[url](url)`
 * swallowed `](url)` into the href (#49822, #85234). Each case asserts the
 * hrefs the renderer's parser actually produces from the preprocessed text.
 */

type MdNode = { children?: MdNode[]; identifier?: string; type: string; url?: string }

// Every href the parser resolves, reference links included, in document order.
function hrefs(markdown: string, prepare = preprocessMarkdown): string[] {
  const tree = unified().use(remarkParse).use(remarkGfm).parse(prepare(markdown)) as MdNode
  const definitions = new Map<string, string>()
  const found: (() => string)[] = []

  const walk = (node: MdNode) => {
    if (node.type === 'definition') {
      definitions.set(node.identifier ?? '', node.url ?? '')
    } else if (node.type === 'link' || node.type === 'image') {
      found.push(() => node.url ?? '')
    } else if (node.type === 'linkReference' || node.type === 'imageReference') {
      found.push(() => definitions.get(node.identifier ?? '') ?? '')
    }

    node.children?.forEach(walk)
  }

  walk(tree)

  return found.map(resolve => resolve())
}

describe('preprocessMarkdown / raw-URL autolinking inside markdown links', () => {
  it.each([
    'https://example.com',
    'http://example.com',
    'https://example.com:8080',
    'https://user@example.com',
    'https://example.com/page',
    'https://example.com?q=test',
    'https://example.com#test',
    'https://www.metodic.io/articles/whats-moving-in-facilitation-july-2026'
  ])('keeps [url](url) intact for %s', url => {
    const input = `[${url}](${url})`

    expect(preprocessMarkdown(input)).toBe(input)
    expect(hrefs(input)).toEqual([url])
  })

  it.each([
    ['[see https://a.example/x for more](https://b.example/y)', ['https://b.example/y']],
    ['[http://[::1]:8080/status](https://example.com/status)', ['https://example.com/status']],
    ['[docs [https://example.com/path]](https://example.com/path)', ['https://example.com/path']],
    ['[a\\]https://example.com/x](https://example.com/y)', ['https://example.com/y']],
    ['[[inner](https://i.example) outer](https://o.example)', ['https://i.example', 'https://o.example']],
    ['![https://example.com/a.png](https://example.com/a.png)', ['https://example.com/a.png']],
    ['[https://x.example](https://x.example "https://x.example title")', ['https://x.example']],
    ['[Foo](https://en.wikipedia.org/wiki/Foo_(bar))', ['https://en.wikipedia.org/wiki/Foo_(bar)']],
    ['<https://example.com/a>', ['https://example.com/a']]
  ])('keeps link labels and targets intact: %s', (input, expected) => {
    expect(preprocessMarkdown(input)).toBe(input)
    expect(hrefs(input)).toEqual(expected)
  })

  it('keeps full and collapsed reference links intact', () => {
    const full = '[https://example.com/a][ref]\n\n[ref]: https://example.com/b'
    const collapsed = '[https://example.com/a][]\n\n[https://example.com/a]: https://example.com/b'

    expect(preprocessMarkdown(full)).toBe(full)
    expect(hrefs(full)).toEqual(['https://example.com/b'])
    expect(preprocessMarkdown(collapsed)).toBe(collapsed)
    expect(hrefs(collapsed)).toEqual(['https://example.com/b'])
  })

  it('never wraps URLs in inline code or code fences', () => {
    const input =
      'Try `[https://x.example](https://x.example)` or\n\n```md\n[https://x.example](https://x.example)\n```'

    expect(preprocessMarkdown(input)).toBe(input)
  })

  it('still autolinks bare URLs sharing a line with a markdown link', () => {
    const input = 'Docs https://a.example/x and [https://b.example/y](https://b.example/y).'

    expect(hrefs(input)).toEqual(['https://a.example/x', 'https://b.example/y'])
  })

  it('never leaks link syntax into an href while the link streams in', () => {
    const link = '[https://example.com/a](https://example.com/a)'
    const streamed = (text: string) => tailBoundedRemend(preprocessMarkdown(text))

    for (let end = 1; end <= link.length; end += 1) {
      for (const url of hrefs(`See ${link.slice(0, end)}`, streamed)) {
        expect(url).not.toMatch(/[\]()]|%5D/)
      }
    }
  })
})

describe('preprocessMarkdown / bare loopback URLs in prose', () => {
  // A loopback URL the model writes into a sentence is user-facing content —
  // "the dev server is at http://localhost:3000" must render the address,
  // not leave a hole (#121683). The only legitimate suppression is the fenced
  // preview hand-off (a whole fenced block whose body is just the URL),
  // owned by LOCAL_PREVIEW_ONLY_RE in normalizeFenceBlocks.
  it.each([
    ['Bare autolink: http://localhost:3000', 'http://localhost:3000'],
    ['With a path: http://localhost:3000/dashboard', 'http://localhost:3000/dashboard'],
    ['Loopback IP: http://127.0.0.1:8080', 'http://127.0.0.1:8080'],
    ['Wildcard bind: http://0.0.0.0:8000', 'http://0.0.0.0:8000'],
    ['IPv6 loopback: http://[::1]:8080/status', 'http://[::1]:8080/status'],
    ['Trailing slash: http://localhost:3000/ - panel', 'http://localhost:3000/'],
    [
      'Server at http://localhost:3000 and docs at https://example.com',
      ['http://localhost:3000', 'https://example.com']
    ]
  ])('preserves bare loopback URLs in prose: %s', (input, expected) => {
    expect(hrefs(input)).toEqual([expected].flat())
  })

  it('keeps a markdown-labeled loopback link intact', () => {
    const input = '[my dev server](http://localhost:3000)'

    expect(preprocessMarkdown(input)).toBe(input)
    expect(hrefs(input)).toEqual(['http://localhost:3000'])
  })

  it('still drops a fenced block whose whole body is a loopback URL (preview hand-off)', () => {
    const input = 'Before\n\n```\nhttp://localhost:3000\n```\n\nAfter'

    expect(preprocessMarkdown(input)).toBe('Before\n\n\nAfter')
    expect(hrefs(input)).toEqual([])
  })
})

describe('preprocessMarkdown / bare-URL trailing punctuation', () => {
  it.each([
    ['Visit https://example.com/a.', 'https://example.com/a'],
    ['Visit https://example.com/a, then', 'https://example.com/a'],
    ['(see https://example.com/a)', 'https://example.com/a'],
    ['(see https://example.com/a).', 'https://example.com/a'],
    ['(see https://example.com/a.)', 'https://example.com/a'],
    ['[https://example.com/a]', 'https://example.com/a'],
    ['See https://en.wikipedia.org/wiki/Foo_(bar).', 'https://en.wikipedia.org/wiki/Foo_(bar)'],
    ['(https://en.wikipedia.org/wiki/Foo_(bar))', 'https://en.wikipedia.org/wiki/Foo_(bar)'],
    ['Host http://[2001:db8::1]:8080/status.', 'http://[2001:db8::1]:8080/status']
  ])('%s links %s', (input, expected) => {
    expect(hrefs(input)).toEqual([expected])
  })
})

describe('preprocessMarkdown / CJK bare-URL boundaries', () => {
  it('leaves CJK punctuation and adjoining prose outside each bare URL', () => {
    const url = 'https://example.com/release/v2.1.0'

    const tails = [
      '（private，建库成功）',
      '，建库成功',
      '。',
      '。然后',
      '、次へ',
      '；下一项',
      '：详情',
      '！完成',
      '？確認',
      '）',
      '「説明」',
      '『説明』',
      '【说明】'
    ]

    for (const tail of tails) {
      const input = `已建库 ${url}${tail}`

      expect(preprocessMarkdown(input), input).toBe(`已建库 <${url}>${tail}`)
      expect(hrefs(input), input).toEqual([url])
    }

    for (const separator of ['。然后 ', '。', '，', '、', '；', '：']) {
      const input = `https://example.com/a${separator}https://example.com/b`

      expect(preprocessMarkdown(input), input).toBe(`<https://example.com/a>${separator}<https://example.com/b>`)
      expect(hrefs(input), input).toEqual(['https://example.com/a', 'https://example.com/b'])
    }

    expect(preprocessMarkdown(`${url}。约1~10项`)).toBe(`<${url}>。约1\\~10项`)
  })

  it('preserves internationalized URLs and the surrounding Markdown contracts', () => {
    const urls = [
      'https://example.com/路径/含中文',
      'https://example.com/search?q=中文&分类=文档#章节',
      'https://example.com/ＡＢＣ１２３/ｶﾀｶﾅ/～資料?ｑ=中文',
      'https://example.com/%EF%BC%88private%EF%BC%89',
      'https://en.wikipedia.org/wiki/Foo_(bar)',
      'https://example.com/a,b;c:d!e?f=g'
    ]

    for (const url of urls) {
      expect(hrefs(url), url).toEqual([url])
      expect(hrefs(`${url}。然后`), url).toEqual([url])
      expect(hrefs(`**资料: ${url}**。`), url).toEqual([url])
      expect(preprocessMarkdown(`**资料: ${url}**。`), url).toBe(`**资料: <${url}>**。`)
    }

    const explicitUrl = 'https://example.com/资料（新版），目录'

    const protectedInputs = [
      `[${explicitUrl}](${explicitUrl})`,
      `[资料][ref]\n\n[ref]: ${explicitUrl}`,
      `<${explicitUrl}>`,
      `\`${explicitUrl}\``,
      `\`\`\`js\nconst url = '${explicitUrl}'\n\`\`\``
    ]

    for (const input of protectedInputs) {
      expect(preprocessMarkdown(input), input).toBe(input)
    }

    expect(hrefs(protectedInputs[0])).toEqual([explicitUrl])
    expect(hrefs(protectedInputs[1])).toEqual([explicitUrl])
    expect(hrefs(protectedInputs[2])).toEqual([explicitUrl])
    expect(hrefs(protectedInputs[3])).toEqual([])
    expect(hrefs(protectedInputs[4])).toEqual([])
  })
})
