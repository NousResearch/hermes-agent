import { afterEach, describe, expect, it, vi } from 'vitest'

import { registry } from '@/contrib/registry'
import type { Contribution } from '@/contrib/types'
import { $previewTabs, closeRightRail, openPreview } from '@/store/preview'

import { FILE_OPENER_AREA, type FileOpenerContribution, tryOpenFileWithPlugin } from './file-openers'

function contribution(id: string, data: FileOpenerContribution, order = 0): Contribution {
  return { id, area: FILE_OPENER_AREA, data, order }
}

const disposers: Array<() => void> = []

afterEach(() => {
  disposers.splice(0).forEach(dispose => dispose())
  closeRightRail()
  window.localStorage.clear()
})

describe('local file opener contributions', () => {
  it('uses the first claiming contribution in registry order', () => {
    const openFirst = vi.fn()
    const openSecond = vi.fn()
    const fallback = vi.fn()

    tryOpenFileWithPlugin(
      '/vault/note.md',
      [
        contribution('first', { claims: () => true, open: openFirst }, 0),
        contribution('later', { claims: () => true, open: openSecond }, 10)
      ],
      fallback
    )

    expect(openFirst).toHaveBeenCalledWith('/vault/note.md')
    expect(openSecond).not.toHaveBeenCalled()
    expect(fallback).not.toHaveBeenCalled()
  })

  it('falls through when no contribution claims the path', () => {
    const fallback = vi.fn()

    tryOpenFileWithPlugin('/vault/note.md', [contribution('other', { claims: () => false, open: vi.fn() })], fallback)

    expect(fallback).toHaveBeenCalledOnce()
  })

  it('falls back when the selected opener rejects', async () => {
    const fallback = vi.fn()
    const warning = vi.spyOn(console, 'warn').mockImplementation(() => undefined)

    tryOpenFileWithPlugin(
      '/vault/note.md',
      [contribution('broken', { claims: () => true, open: () => Promise.reject(new Error('reader unavailable')) })],
      fallback
    )
    await vi.waitFor(() => expect(fallback).toHaveBeenCalledOnce())

    expect(warning).toHaveBeenCalled()
    warning.mockRestore()
  })

  it('routes openPreview through the registered file opener before creating a built-in tab', () => {
    const open = vi.fn()

    disposers.push(
      registry.register({
        ...contribution('vault-reader', { claims: path => path.endsWith('.md'), open }),
        source: 'plugin:hermes-vault'
      })
    )

    openPreview({ kind: 'file', label: 'note.md', path: '/vault/note.md', source: '/vault/note.md', url: 'file:///vault/note.md' })

    expect(open).toHaveBeenCalledWith('/vault/note.md')
    expect($previewTabs.get()).toHaveLength(0)
  })
})
