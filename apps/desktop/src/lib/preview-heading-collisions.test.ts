import { expect, it } from 'vitest'

import { rehypePreviewHeadingIds } from './preview-markdown-links'

it('allocates unique anchors when authored headings collide with numbered duplicates', () => {
  const headings = ['Topic', 'Topic', 'Topic-1', 'Topic', 'Topic-1']
  const tree = {
    type: 'root',
    children: headings.map(value => ({
      type: 'element',
      tagName: 'h2',
      properties: {} as Record<string, unknown>,
      children: [{ type: 'text', value }]
    }))
  }

  rehypePreviewHeadingIds()(tree)

  const ids = tree.children.map(node => node.properties.id)

  expect(new Set(ids).size).toBe(headings.length)
  expect(ids).toEqual(['topic', 'topic-1', 'topic-1-1', 'topic-2', 'topic-1-2'])
})
