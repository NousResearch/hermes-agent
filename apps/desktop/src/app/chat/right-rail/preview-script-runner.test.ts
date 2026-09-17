import { afterEach, describe, expect, it } from 'vitest'

import {
  activePreviewScriptRunner,
  registerPreviewScriptRunner
} from '@/app/chat/right-rail/preview-script-runner'
import { $rightRailActiveTabId } from '@/store/layout'
import { $previewTabs, type PreviewTab } from '@/store/preview'

const tabs: PreviewTab[] = [
  { id: 'url:a', target: { kind: 'url', label: 'A', source: 'about:blank', url: 'about:blank' } },
  { id: 'url:b', target: { kind: 'url', label: 'B', source: 'about:blank', url: 'about:blank' } }
]

let unregisterA: (() => void) | undefined
let unregisterB: (() => void) | undefined

afterEach(() => {
  unregisterA?.()
  unregisterB?.()
  unregisterA = undefined
  unregisterB = undefined
  $previewTabs.set([])
  $rightRailActiveTabId.set(null)
})

describe('preview script runners (multi-tab)', () => {
  it('unregistering one tab leaves the sibling runner intact', async () => {
    $previewTabs.set(tabs)
    $rightRailActiveTabId.set('url:b')

    unregisterA = registerPreviewScriptRunner('url:a', async () => 'a')
    unregisterB = registerPreviewScriptRunner('url:b', async () => 'b')

    unregisterA()
    unregisterA = undefined

    expect(await activePreviewScriptRunner()?.('')).toBe('b')
  })
})
