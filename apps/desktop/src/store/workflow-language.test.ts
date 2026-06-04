import { beforeEach, describe, expect, it, vi } from 'vitest'

const storage = new Map<string, string>()

vi.mock('@/lib/storage', () => ({
  persistString: (key: string, value: null | string) => {
    if (value === null) {
      storage.delete(key)
    } else {
      storage.set(key, value)
    }
  },
  storedString: (key: string) => storage.get(key) ?? null
}))

describe('workflow language preference', () => {
  beforeEach(() => {
    storage.clear()
    vi.resetModules()
  })

  it('defaults to English', async () => {
    const { $workflowLanguage } = await import('./workflow-language')

    expect($workflowLanguage.get()).toBe('en')
  })

  it('normalizes invalid values to English', async () => {
    storage.set('hermes.desktop.workflowLanguage.v1', 'fr')
    const { $workflowLanguage } = await import('./workflow-language')

    expect($workflowLanguage.get()).toBe('en')
  })

  it('persists Chinese selection', async () => {
    const { $workflowLanguage, WORKFLOW_LANGUAGE_STORAGE_KEY, setWorkflowLanguage } = await import('./workflow-language')

    setWorkflowLanguage('zh')

    expect($workflowLanguage.get()).toBe('zh')
    expect(storage.get(WORKFLOW_LANGUAGE_STORAGE_KEY)).toBe('zh')
  })
})
