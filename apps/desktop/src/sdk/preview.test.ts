import { beforeEach, describe, expect, it } from 'vitest'

import { registerPreviewScriptRunner } from '@/app/chat/right-rail/preview-script-runner'
import { host } from '@/sdk'
import { $rightRailActiveTabId } from '@/store/layout'
import { closeRightRail, openPreview } from '@/store/preview'

function urlTarget(url: string) {
  return { kind: 'url' as const, label: 'Browser', source: url, url }
}

function fileTarget(path: string) {
  return {
    kind: 'file' as const,
    label: path,
    path,
    previewKind: 'text' as const,
    source: path,
    url: `file://${path}`
  }
}

describe('host.preview', () => {
  let unbind: (() => void) | undefined

  beforeEach(() => {
    unbind?.()
    unbind = undefined
    closeRightRail()
    window.localStorage.clear()
  })

  it('rejects when no preview page is open', async () => {
    expect(host.preview.isAvailable()).toBe(false)
    await expect(host.preview.executeJavaScript('1 + 1')).rejects.toThrow('No preview page is open.')
  })

  it('is unavailable on a file peek that never registered a guest runner', () => {
    openPreview(fileTarget('/work/notes.txt'))
    expect(host.preview.isAvailable()).toBe(false)
  })

  it('runs JS on the active Browser tab and returns the guest value', async () => {
    openPreview(urlTarget('https://example.com'))
    const tabId = $rightRailActiveTabId.get()!

    unbind = registerPreviewScriptRunner(tabId, async code => ({ ran: code }))

    expect(host.preview.isAvailable()).toBe(true)
    await expect(host.preview.executeJavaScript('document.title')).resolves.toEqual({ ran: 'document.title' })
  })

  it('rejects a non-string without throwing synchronously', async () => {
    const pending = host.preview.executeJavaScript(1 as never)

    expect(pending).toBeInstanceOf(Promise)
    await expect(pending).rejects.toThrow('host.preview.executeJavaScript requires a string')
  })

  it('forwards a guest throw as a rejection', async () => {
    openPreview(urlTarget('https://example.com'))
    unbind = registerPreviewScriptRunner($rightRailActiveTabId.get()!, async () => {
      throw new Error('guest exploded')
    })

    await expect(host.preview.executeJavaScript('x')).rejects.toThrow('guest exploded')
  })
})
