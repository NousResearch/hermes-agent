import { beforeEach, describe, expect, it, vi } from 'vitest'

vi.mock('@/i18n/runtime', () => ({
  translateNow: (key: string) => `[zh] ${key}`
}))

const store = new Map<string, string>()
const localStorageMock = {
  getItem: vi.fn((key: string) => store.get(key) ?? null),
  setItem: vi.fn((key: string, value: string) => {
    store.set(key, value)
  }),
  removeItem: vi.fn((key: string) => {
    store.delete(key)
  }),
  clear: vi.fn(() => store.clear())
}

Object.defineProperty(globalThis, 'localStorage', { value: localStorageMock, configurable: true })
Object.defineProperty(window, 'localStorage', { value: localStorageMock, configurable: true })

import {
  $promptTemplates,
  addTemplate,
  resetToBuiltins
} from './prompt-templates'
import {
  applyPromptImportJob,
  normalizePromptText,
  parsePromptImportJob,
  tokenJaccard
} from './prompt-import'

describe('prompt-import', () => {
  beforeEach(() => {
    store.clear()
    localStorageMock.getItem.mockClear()
    localStorageMock.setItem.mockClear()
    resetToBuiltins()
  })

  it('tokenJaccard is 1 for identical bags', () => {
    expect(tokenJaccard('苏格拉底 追问 前提', '苏格拉底 追问 前提')).toBe(1)
  })

  it('normalizePromptText collapses whitespace', () => {
    expect(normalizePromptText('a  b\n\n\nc')).toBe('a b\n\nc')
  })

  it('parsePromptImportJob rejects empty merge', () => {
    expect(parsePromptImportJob({ v: 1, op: 'merge', templates: [] })).toBeNull()
  })

  it('parsePromptImportJob accepts replace clear', () => {
    expect(parsePromptImportJob({ v: 1, op: 'replace', templates: [] })).toEqual({
      v: 1,
      op: 'replace',
      source: undefined,
      folder: undefined,
      folderDescription: undefined,
      templates: []
    })
  })

  it('merge creates folder + template', () => {
    const result = applyPromptImportJob({
      op: 'merge',
      folder: 'Group',
      templates: [{ label: '反向采访', description: 'd', text: '请先采访我' }]
    })

    expect(result.created).toBe(1)
    expect(result.foldersEnsured).toContain('Group')
    const list = $promptTemplates.get()
    expect(list.some(n => n.kind === 'folder' && n.label === 'Group')).toBe(true)
    expect(list.some(n => n.kind === 'template' && n.label === '反向采访')).toBe(true)
  })

  it('skips exact text duplicate', () => {
    applyPromptImportJob({
      folder: 'Group',
      templates: [{ label: 'A', text: 'same body' }]
    })
    const second = applyPromptImportJob({
      folder: 'Group',
      templates: [{ label: 'B', text: 'same body' }]
    })

    expect(second.skippedDup).toBe(1)
    expect(second.created).toBe(0)
  })

  it('marks near-duplicate labels', () => {
    addTemplate('苏格拉底式提问', '', 'long unique body alpha')
    const result = applyPromptImportJob({
      folder: 'Group',
      templates: [{ label: '苏格拉底追问', text: 'totally different body beta' }]
    })

    expect(result.created).toBe(1)
    const created = $promptTemplates.get().find(n => n.text.includes('beta'))
    expect(created?.label).toMatch(/〔相近〕/)
  })

  it('replace clears then imports', () => {
    addTemplate('keep-me', '', 'x')
    applyPromptImportJob({
      op: 'replace',
      folder: 'Toolkit',
      templates: [{ label: 'only', text: 'new' }]
    })
    const labels = $promptTemplates.get().map(n => n.label)
    expect(labels).not.toContain('keep-me')
    expect(labels).toContain('Toolkit')
    expect(labels).toContain('only')
  })
})
