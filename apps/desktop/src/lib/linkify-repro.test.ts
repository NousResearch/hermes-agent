import { describe, expect, it } from 'vitest'
import { linkifyFilePaths } from '@/lib/linkify-file-paths'
import { preprocessMarkdown } from '@/lib/markdown-preprocess'

describe('reproduce the exact garbled message', () => {
  it('the exact message from the DB renders without NUL leak', () => {
    // [34438] 原始消息：粗体 + 反引号代码 + 中文混排
    const original =
      '这次同步比之前多了解了一处风险：**以后每次 `hermes update` 都可能在主仓库 main 分支触发自动换 bundle**，所以自用版被覆盖时，重新走这个流程即可。'
    const pre = preprocessMarkdown(original)
    const out = linkifyFilePaths(pre, '/Users/echo')
    expect(out.includes('\u0000')).toBe(false)
    // 输出不该含 \u0000N\u0000 模式
    const leaked = [...out.matchAll(/\u0000(\d+)\u0000/g)]
    expect(leaked).toEqual([])
  })

  it('bold-wrapped inline code with multiple masks', () => {
    const s = '**加粗 `x` 和 **`hermes update`** 还有 /tmp/a.ts 结尾'
    const pre = preprocessMarkdown(s)
    const out = linkifyFilePaths(pre, '/Users/echo')
    expect(out.includes('\u0000')).toBe(false)
  })
})
