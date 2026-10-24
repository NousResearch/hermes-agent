import { describe, expect, it } from 'vitest'

import { loadMathPlugin } from './math-plugin-loader'

describe('loadMathPlugin', () => {
  it('resolves the shared memoized math plugin once', async () => {
    const first = await loadMathPlugin()
    const second = await loadMathPlugin()

    // Same instance both times (module-level cache): the remark parser and
    // the KaTeX LRU must be shared, matching the old module-scope semantics.
    expect(second).toBe(first)
    expect(first.name).toBe('katex')
    expect(first.type).toBe('math')
  })
})
