import { describe, expect, it } from 'vitest'

import { type MdastNode, remarkSoftBreaks } from './remark-soft-breaks'

const paragraph = (...children: MdastNode[]) => ({ type: 'root', children: [{ type: 'paragraph', children }] })

describe('remarkSoftBreaks', () => {
  it('turns single newlines in prose into line breaks', () => {
    const tree = paragraph({ type: 'text', value: '1\n2  \n3' })

    remarkSoftBreaks()(tree)

    expect(tree.children[0]!.children).toEqual([
      { type: 'text', value: '1' },
      { type: 'break' },
      { type: 'text', value: '2' },
      { type: 'break' },
      { type: 'text', value: '3' }
    ])
  })

  it('leaves code and math nodes untouched', () => {
    const code = { type: 'code', value: 'a\nb' }
    const math = { type: 'inlineMath', value: 'x\ny' }
    const tree: MdastNode = { type: 'root', children: [code, paragraph(math).children[0]!] }

    remarkSoftBreaks()(tree)

    expect(tree.children?.[0]).toEqual({ type: 'code', value: 'a\nb' })
    expect(tree.children?.[1]).toEqual({ type: 'paragraph', children: [{ type: 'inlineMath', value: 'x\ny' }] })
  })

  it('splits text nested inside emphasis', () => {
    const tree = paragraph({ type: 'emphasis', children: [{ type: 'text', value: 'a\nb' }] })

    remarkSoftBreaks()(tree)

    expect(tree.children[0]!.children[0]).toEqual({
      type: 'emphasis',
      children: [{ type: 'text', value: 'a' }, { type: 'break' }, { type: 'text', value: 'b' }]
    })
  })
})
