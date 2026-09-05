import { describe, expect, it } from 'vitest'

import { annotateSplitDepth, groupAnnotations } from './group'
import type { ComposerReadyAnnotation } from './pack'

function item(number: number, selector?: string): ComposerReadyAnnotation {
  return {
    imageDataUrl: '',
    note: `note ${number}`,
    number,
    prompt: `Comment ${number}`,
    identity: selector ? { css: {}, html: '', selector, tag: 'div', text: '' } : undefined
  }
}

describe('annotateSplitDepth', () => {
  it('splits at the shallowest region where the comments disagree', () => {
    expect(annotateSplitDepth(['body>main>div.header>h1', 'body>main>div.header>p', 'body>main>div.footer>a'])).toBe(3)
  })

  it('does not split siblings inside one component', () => {
    // Differ only at the leaf, so the ancestor paths are identical: the depth
    // runs to the end of the shared path and both land in the same group.
    const selectors = ['body>div.card>h1', 'body>div.card>p']

    expect(annotateSplitDepth(selectors)).toBe(2)
    expect(groupAnnotations(selectors.map((selector, index) => item(index + 1, selector)))).toHaveLength(1)
  })

  it('does not split when every comment is on the same element', () => {
    expect(annotateSplitDepth(['body>div.card', 'body>div.card'])).toBe(1)
  })

  it('separates a container comment from comments nested inside it', () => {
    const depth = annotateSplitDepth(['body>main', 'body>main>div.a>span', 'body>main>div.b>span'])

    expect(depth).toBe(2)
  })

  it('handles a single selector and an empty batch', () => {
    expect(annotateSplitDepth(['body>div.only'])).toBe(1)
    expect(annotateSplitDepth([])).toBe(0)
  })
})

describe('groupAnnotations', () => {
  it('gathers comments on the same region and separates different regions', () => {
    const groups = groupAnnotations([
      item(1, 'body>main>section.hero>h1'),
      item(2, 'body>main>section.pricing>button'),
      item(3, 'body>main>section.hero>p'),
      item(4, 'body>main>section.pricing>span')
    ])

    expect(groups).toHaveLength(2)
    expect(groups[0]?.label).toBe('section.hero')
    expect(groups[0]?.items.map(entry => entry.number)).toEqual([1, 3])
    expect(groups[1]?.label).toBe('section.pricing')
    expect(groups[1]?.items.map(entry => entry.number)).toEqual([2, 4])
  })

  it('produces groups whose subtrees do not overlap, so they can run in parallel', () => {
    const groups = groupAnnotations([
      item(1, 'body>main>section.hero>h1'),
      item(2, 'body>main>section.pricing>button'),
      item(3, 'body>main>section.faq>li')
    ])

    const keys = groups.map(group => group.key)
    const overlapping = keys.filter(key => keys.some(other => other !== key && other.startsWith(`${key}>`)))

    expect(keys).toHaveLength(3)
    expect(overlapping).toEqual([])
  })

  it('keeps area pins in their own trailing group rather than guessing a subtree', () => {
    const groups = groupAnnotations([item(1, 'body>main>div.a>h1'), item(2), item(3, 'body>main>div.b>h1'), item(4)])

    const loose = groups[groups.length - 1]

    expect(loose?.key).toBe('')
    expect(loose?.label).toBe('')
    expect(loose?.items.map(entry => entry.number)).toEqual([2, 4])
  })

  it('returns one group when every comment lands in the same region', () => {
    const groups = groupAnnotations([item(1, 'body>div.card>h1'), item(2, 'body>div.card>p')])

    expect(groups).toHaveLength(1)
  })

  it('orders groups by first appearance so the numbering still reads in click order', () => {
    const groups = groupAnnotations([
      item(1, 'body>main>div.b>h1'),
      item(2, 'body>main>div.a>h1'),
      item(3, 'body>main>div.b>p')
    ])

    expect(groups.map(group => group.label)).toEqual(['div.b', 'div.a'])
    expect(groups[0]?.items.map(entry => entry.number)).toEqual([1, 3])
  })

  it('survives a batch with no element comments at all', () => {
    const groups = groupAnnotations([item(1), item(2)])

    expect(groups).toHaveLength(1)
    expect(groups[0]?.items).toHaveLength(2)
  })
})

describe('groupAnnotations refinement', () => {
  // A normal page: header / main / footer part company at the top, so a single
  // split buries every section under `main`.
  const page = [
    ...['a.logo', 'ul.links>li', 'button.menu'].map((tail, index) => item(index + 1, `body>header.nav>${tail}`)),
    ...['h1', 'p.sub', 'a.cta', 'img.art'].map((tail, index) => item(index + 4, `body>main>section.hero>${tail}`)),
    ...['table', 'button.buy', 'span.note'].map((tail, index) => item(index + 8, `body>main>section.pricing>${tail}`)),
    ...['details:nth-of-type(1)', 'details:nth-of-type(4)'].map((tail, index) =>
      item(index + 11, `body>main>section.faq>${tail}`)
    ),
    ...['div.cols>ul', 'small.copy'].map((tail, index) => item(index + 13, `body>footer.foot>${tail}`))
  ]

  it('breaks up the branch that would otherwise swallow most of the batch', () => {
    const groups = groupAnnotations(page)
    const labels = groups.map(group => group.label)

    expect(labels).toContain('section.hero')
    expect(labels).toContain('section.pricing')
    expect(labels).toContain('section.faq')
    expect(labels).not.toContain('main')
  })

  it('leaves no group holding more than a third of the batch', () => {
    const groups = groupAnnotations(page)
    const ceiling = Math.max(2, Math.ceil(page.length / 3))

    for (const group of groups) {
      expect(group.items.length).toBeLessThanOrEqual(ceiling)
    }
  })

  it('loses and duplicates nothing while refining', () => {
    const numbers = groupAnnotations(page)
      .flatMap(group => group.items.map(entry => entry.number))
      .sort((a, b) => a - b)

    expect(numbers).toEqual(page.map(entry => entry.number))
  })

  it('keeps refined groups on non-overlapping subtrees', () => {
    const keys = groupAnnotations(page).map(group => group.key)
    const nested = keys.filter(key => keys.some(other => other !== key && other.startsWith(`${key}>`)))

    expect(nested).toEqual([])
  })

  it('stops instead of looping when an oversized group cannot divide further', () => {
    const identical = Array.from({ length: 9 }, (_, index) => item(index + 1, 'body>div.card>span'))
    const groups = groupAnnotations(identical)

    expect(groups).toHaveLength(1)
    expect(groups[0]?.items).toHaveLength(9)
  })
})
