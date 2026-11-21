import { describe, expect, it } from 'vitest'

import { describesNothingNew, toolRows } from './derive-tools'

function redundant(name: string, description: string): boolean {
  const [tool] = toolRows(
    [{ categories: [], deprecated: false, description, facet: 'read', hints: [], name, slug: 'read' }],
    new Set()
  )

  return describesNothingNew(tool)
}

describe('tool description disclosure', () => {
  it('keeps a description that adds words beyond the name, in any script', () => {
    const cases = [
      ['中文资料读取', '按明确地址读取项目资料并保留段落结构。'],
      ['Read', 'Read 中文说明'],
      ['Поиск', 'Искать документы.'],
      ['읽기', '문서를 읽습니다.']
    ]

    for (const [name, description] of cases) {
      expect(redundant(name, description), `${name}: ${description}`).toBe(false)
    }
  })

  it('still hides empty, punctuation-only and name-only descriptions, in any script', () => {
    const cases = [
      ['Read file', ''],
      ['Read file', ' \n\t '],
      ['Read file', '...!? 。！？【】 🔍'],
      ['Read file', '\u2764\uFE0F \u{1F3F3}\uFE0F\u200D\u{1F308}'],
      ['Read file', '  READ_file! '],
      ['读取资料', '读取资料。'],
      ['Поиск файлов', 'ПОИСК_ФАЙЛОВ!'],
      ['Résumé', 'Re\u0301sume\u0301!']
    ]

    for (const [name, description] of cases) {
      expect(redundant(name, description), `${name}: ${description}`).toBe(true)
    }
  })
})
