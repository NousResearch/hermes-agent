import { afterEach, expect, it, vi } from 'vitest'

import { activePreviewInput, registerPreviewInput } from '@/app/chat/right-rail/preview-input'
import { activePreviewNav, registerPreviewNav } from '@/app/chat/right-rail/preview-nav'
import { readActivePreview, registerPreviewPageReader } from '@/app/chat/right-rail/preview-reader'
import { activePreviewScriptRunner, registerPreviewScriptRunner } from '@/app/chat/right-rail/preview-script-runner'

import * as preview from './preview'
import { $selectedStoredSessionId } from './session'

const disposers: Array<() => void> = []
afterEach(() => {
  disposers.splice(0).forEach(dispose => dispose())
  preview.closeRightRail()
  preview.dropPreviewTabsForProfile('tool-owner')
  preview.dropPreviewTabsForProfile('tool-other')
  preview.setPreviewScope('default')
  $selectedStoredSessionId.set(null)
})

const viewerTarget = {
  kind: 'url',
  label: 'Viewer',
  source: 'viewer',
  url: 'https://viewer.example/view#ticket=fixture',
  transient: true,
  browserContext: 'isolated'
} as const

it('keeps a viewer mounted but excludes it from another session’s agent readers and input', async () => {
  preview.setPreviewScope('tool-owner')
  $selectedStoredSessionId.set('sess-owner')

  const tab = preview.openPreview(viewerTarget)

  const reader = vi.fn(async () => ({ text: 'Private viewer', title: 'Viewer', url: tab.target.url }))
  const runner = vi.fn(async () => null)
  const input = { focus: vi.fn(), send: vi.fn() }
  const nav = { back: vi.fn(), forward: vi.fn(), reload: vi.fn() }
  disposers.push(
    registerPreviewPageReader(tab.id, reader),
    registerPreviewScriptRunner(tab.id, runner),
    registerPreviewInput(tab.id, input),
    registerPreviewNav(tab.id, nav)
  )
  expect((await readActivePreview())?.text).toBe('Private viewer')
  expect(activePreviewScriptRunner()).toBe(runner)
  expect(activePreviewInput()).toBe(input)
  expect(activePreviewNav()).toBe(nav)

  // Another profile's chat on screen: the viewer stays in the window (and
  // mounted), but it is not that session's tab.
  preview.setPreviewScope('tool-other')
  $selectedStoredSessionId.set('sess-other')
  expect(preview.$previewTabs.get()[0]).toBe(tab)
  expect(await readActivePreview()).toBeNull()
  expect(activePreviewScriptRunner()).toBeNull()
  expect(activePreviewInput()).toBeNull()
  expect(activePreviewNav()).toBeNull()
  expect(reader).toHaveBeenCalledOnce()
  preview.closeAgentPreview('sess-other', ['Viewer'])
  preview.closeAgentPreview('sess-other', [])
  expect(preview.$previewTabs.get()[0]).toBe(tab)

  preview.setPreviewScope('tool-owner')
  $selectedStoredSessionId.set('sess-owner')
  expect(activePreviewScriptRunner()).toBe(runner)
  expect(activePreviewInput()).toBe(input)
  expect(activePreviewNav()).toBe(nav)
})

it('excludes a foreign viewer from the tab inventory when reading an ordinary tab', async () => {
  preview.setPreviewScope('tool-owner')
  $selectedStoredSessionId.set('sess-owner')
  preview.openPreview({ ...viewerTarget, label: 'Private', source: 'private' })
  preview.setPreviewScope('tool-other')
  $selectedStoredSessionId.set('sess-other')

  const tab = preview.openPreview({
    kind: 'file',
    label: 'Public',
    source: '/work/public.txt',
    url: 'file:///work/public.txt'
  })

  const read = await readActivePreview()
  expect(read?.url).toBe(tab.target.url)
  expect(read?.tabs).toBeUndefined()
  expect(JSON.stringify(read)).not.toContain('ticket')
})
