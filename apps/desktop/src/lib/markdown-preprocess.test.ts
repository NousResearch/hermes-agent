import { describe, expect, it } from 'vitest'

import { preprocessMarkdown } from '@/lib/markdown-preprocess'

describe('preprocessMarkdown prose fences', () => {
  it('strips plaintext language labels while keeping prose fence titles', () => {
    const input = ['```text 【詳細ページ用】', 'タイトル：四街道市', '```'].join('\n')

    expect(preprocessMarkdown(input)).toBe('【詳細ページ用】\nタイトル：四街道市')
  })

  it('preserves line breaks inside Japanese plaintext prose fences', () => {
    const input = [
      '```text',
      '基本情報：',
      '工事内容：外壁塗装・付帯部塗装',
      '建物種別：戸建て',
      '築年数：未確認',
      '```'
    ].join('\n')

    expect(preprocessMarkdown(input)).toBe(
      ['基本情報：\\', '工事内容：外壁塗装・付帯部塗装\\', '建物種別：戸建て\\', '築年数：未確認'].join('\n')
    )
  })

  it('keeps real code fences intact', () => {
    const input = ['```ts', 'const value = 1', '```'].join('\n')

    expect(preprocessMarkdown(input)).toBe(input)
  })
})
