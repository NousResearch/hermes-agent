import { describe, expect, it } from 'vitest'

import { isLikelyProseCodeBlock } from '@/lib/markdown-code'

describe('isLikelyProseCodeBlock', () => {
  it('detects prose that Streamdown mislabels as an unknown language', () => {
    expect(
      isLikelyProseCodeBlock(
        'heads',
        [
          '- Pure white (`#ffffff`), roughness 0.55, no emissive',
          '- Black wireframe edges at 35% opacity',
          '',
          'Want the bunny gone, or want me to keep riffing on it?'
        ].join('\n')
      )
    ).toBe(true)
  })

  it('treats Japanese plaintext flow fences as prose instead of code cards', () => {
    const flow = `社長・担当者
  ↓
Discord #施工事例 に写真と数行メモを投稿
  ↓
Hermes Agent / ライ君 が内容を整理
  ↓
HPの施工事例ページにそのまま載せられる下書きを作成
  ↓
必要なら Googleドキュメント / Markdown / PDF / WordPress担当会社へ送る文章まで生成`

    expect(isLikelyProseCodeBlock('text', flow)).toBe(true)
  })

  it('keeps real code blocks', () => {
    expect(isLikelyProseCodeBlock('ts', 'const value = { bunny: true };\nreturn value')).toBe(false)
  })
})
