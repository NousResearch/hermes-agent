import { beforeEach, describe, expect, it, vi } from 'vitest'

const loadStore = async () => {
  vi.resetModules()

  return import('./code-block-collapse')
}

describe('code block collapse preference', () => {
  beforeEach(() => {
    window.localStorage.clear()
  })

  it('defaults to compact (the pre-setting fold) and persists changes', async () => {
    const first = await loadStore()

    expect(first.$codeBlockCollapse.get()).toBe('compact')

    first.setCodeBlockCollapse('off')

    expect(window.localStorage.getItem('hermes.desktop.codeBlockCollapse')).toBe('off')
    expect((await loadStore()).$codeBlockCollapse.get()).toBe('off')
  })

  it('falls back to compact for an unknown stored value', async () => {
    window.localStorage.setItem('hermes.desktop.codeBlockCollapse', 'huge')

    expect((await loadStore()).$codeBlockCollapse.get()).toBe('compact')
  })

  it('folds tall blocks later than compact ones', async () => {
    const { CODE_BLOCK_COLLAPSE_LIMITS } = await loadStore()

    expect(CODE_BLOCK_COLLAPSE_LIMITS.tall.thresholdPx).toBeGreaterThan(CODE_BLOCK_COLLAPSE_LIMITS.compact.thresholdPx)
  })
})
