import { afterEach, expect, it, vi } from 'vitest'

import { activePreviewInput, registerPreviewInput } from '@/app/chat/right-rail/preview-input'
import { activePreviewNav, registerPreviewNav } from '@/app/chat/right-rail/preview-nav'
import { readActivePreview, registerPreviewPageReader } from '@/app/chat/right-rail/preview-reader'
import { activePreviewScriptRunner, registerPreviewScriptRunner } from '@/app/chat/right-rail/preview-script-runner'

import * as preview from './preview'

const disposers: Array<() => void> = []
afterEach(() => {
  disposers.splice(0).forEach(dispose => dispose())
  preview.closeRightRail()
  preview.dropPreviewTabsForProfile('tool-owner')
  preview.dropPreviewTabsForProfile('tool-other')
  preview.setPreviewScope('default')
})

it('keeps a viewer mounted but excludes it from another profile’s agent readers and input', async () => {
  preview.setPreviewScope('tool-owner')

  const tab = preview.openPreview({
    kind: 'url',
    label: 'Viewer',
    source: 'viewer',
    url: 'https://viewer.example/view#ticket=fixture',
    transient: true,
    browserContext: 'isolated'
  })

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

  preview.setPreviewScope('tool-other')
  expect(preview.$previewTabs.get()[0]).toBe(tab)
  expect(await readActivePreview()).toBeNull()
  expect(activePreviewScriptRunner()).toBeNull()
  expect(activePreviewInput()).toBeNull()
  expect(activePreviewNav()).toBeNull()
  expect(reader).toHaveBeenCalledOnce()
  expect(preview.closeAgentPreviews('Viewer')).toBe(false)
  preview.closeAgentPreviews()
  expect(preview.$previewTabs.get()[0]).toBe(tab)

  preview.setPreviewScope('tool-owner')
  expect(activePreviewScriptRunner()).toBe(runner)
  expect(activePreviewInput()).toBe(input)
  expect(activePreviewNav()).toBe(nav)
})

it('excludes a foreign viewer from the tab inventory when reading an ordinary tab', async () => {
  preview.setPreviewScope('tool-owner')
  preview.openPreview({
    kind: 'url',
    label: 'Private',
    source: 'private',
    url: 'https://viewer.example/view#ticket=fixture',
    transient: true,
    browserContext: 'isolated'
  })
  preview.setPreviewScope('tool-other')

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
