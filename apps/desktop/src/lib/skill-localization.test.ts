import { describe, expect, it } from 'vitest'

import skillCatalog from '@/i18n/skill-catalog.zh.json'

import {
  skillCategoryLabel,
  skillCommandDescription,
  skillDescription,
  skillDisplayName,
  skillSearchText
} from './skill-localization'

// 对照表本身的自检：每条都带中文、标识与中文名对得上，分类可由条目到达。
const entries = Object.entries(skillCatalog.skills)
const han = /[\u3400-\u9fff]/

describe('skill catalog integrity', () => {
  it('covers every translated skill with a Chinese name and description', () => {
    expect(entries.length).toBeGreaterThan(0)

    for (const [name, entry] of entries) {
      expect(entry.displayName, name).toMatch(han)
      expect(entry.description, name).toMatch(han)
      // 中文名以 " · <原标识>" 结尾，列表里认得出 /name 怎么调。
      expect(entry.displayName, name).toMatch(new RegExp(` · ${name.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}$`))
    }
  })

  it('keeps every category label translated and reachable from its entries', () => {
    const categories = new Set(Object.keys(skillCatalog.categories))
    expect(categories.size).toBeGreaterThan(0)

    // 品牌名保持原文（GitHub 就是 GitHub），其余分类必须有中文。
    const brands = new Set(['GitHub'])

    for (const [category, label] of Object.entries(skillCatalog.categories)) {
      expect(label, category).not.toBe('')

      if (!brands.has(label)) {
        expect(label, category).toMatch(han)
      }
    }

    for (const [name, entry] of entries) {
      if (entry.category !== null) {
        expect(categories.has(entry.category), `${name}: ${entry.category}`).toBe(true)
      }
    }
  })
})

describe('skill display localization', () => {
  it('prefixes the Chinese name for zh and leaves other locales alone', () => {
    expect(skillDisplayName('pdf', 'zh')).toBe('PDF 处理 · pdf')
    expect(skillDisplayName('test-driven-development', 'zh')).toBe('测试驱动开发 · test-driven-development')
    expect(skillDisplayName('pdf', 'en')).toBe('pdf')
    // 桌面 locale 是 'zh'；'zh-CN' 从未在 languages.ts 里出现过，不该被当成命中。
    expect(skillDisplayName('pdf', 'zh-CN')).toBe('pdf')
    expect(skillDisplayName('no-such-skill', 'zh')).toBe('no-such-skill')
  })

  it('translates the description only while it still matches the source it was made from', () => {
    const source = 'PDF files: create, read, merge, fill, OCR, edit text.'
    expect(skillDescription('pdf', source, 'zh')).toBe('PDF 文件：创建、读取、合并、填写、OCR、编辑文本。')
    expect(skillDescription('pdf', source, 'en')).toBe(source)
    // 上游改了 description：译文已经对不上，回退原文而不是显示旧意思。
    expect(skillDescription('pdf', 'PDF files, now with stamping.', 'zh')).toBe('PDF files, now with stamping.')
    expect(skillDescription('no-such-skill', source, 'zh')).toBe(source)
  })

  it('translates a slash-completion row from its command text', () => {
    const entry = (skillCatalog.skills as Record<string, { description: string; sourceDescription: string }>)[
      'pdf'
    ]

    // 补全行的 key 是 `/pdf`，要还原成技能名才查得到译文。
    expect(skillCommandDescription('/pdf', entry.sourceDescription, 'zh')).toBe(entry.description)
    // 带参数（`/name arg`）取首个词；英文界面与未知技能一律原样返回。
    expect(skillCommandDescription('/pdf something', entry.sourceDescription, 'zh')).toBe(entry.description)
    expect(skillCommandDescription('/pdf', entry.sourceDescription, 'en')).toBe(entry.sourceDescription)
    expect(skillCommandDescription('/skills', '原样说明', 'zh')).toBe('原样说明')
    expect(skillCommandDescription('/pdf', '上游改过的说明', 'zh')).toBe('上游改过的说明')
  })

  it('labels categories in Chinese, falling back to the raw key', () => {
    expect(skillCategoryLabel('productivity', 'zh')).toBe('效率办公')
    expect(skillCategoryLabel('productivity', 'en')).toBe('productivity')
    expect(skillCategoryLabel('no-such-category', 'zh')).toBe('no-such-category')
  })

  it('searches Chinese copy without losing the English terms', () => {
    const skill = {
      name: 'pdf',
      description: 'PDF files: create, read, merge, fill, OCR, edit text.',
      category: 'productivity'
    }

    const zhText = skillSearchText(skill, 'zh')

    expect(zhText).toContain('合并') // 中文说明能搜到
    expect(zhText).toContain('merge') // 英文说明仍在
    expect(zhText).toContain('pdf') // 原标识仍在
    expect(skillSearchText(skill, 'en')).toContain('merge')
    expect(skillSearchText(skill, 'en')).not.toContain('合并')
  })
})
