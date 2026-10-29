import { cleanup, render } from '@testing-library/react'
import { createElement } from 'react'
import { afterEach, expect, it, vi } from 'vitest'

import { $previewTabs, closeRightRailTab, openPreview, setPreviewScope } from '@/store/preview'
import { $selectedStoredSessionId } from '@/store/session'

import { PreviewTilePane } from './preview'

vi.mock('./preview-pane', () => ({
  PreviewPane: ({ tabId }: { tabId: string }) => createElement('div', { 'data-pane': tabId })
}))

afterEach(() => {
  cleanup()
  setPreviewScope('default')
})

// A parked body of another bot's chat keeps rendering while the rail shows the
// next profile's bucket: rendering nothing would unmount its guest and the page
// would reload when the user comes back to that bot.
it("keeps rendering a hidden tab whose profile is out of view, and nothing once it is closed", () => {
  $selectedStoredSessionId.set('sess-a')
  openPreview({ kind: 'url', label: 'A', source: 'http://localhost:4321', url: 'http://localhost:4321' })
  const tabId = $previewTabs.get().at(-1)!.id

  setPreviewScope('bot-b')
  $selectedStoredSessionId.set('sess-b')
  expect($previewTabs.get().some(tab => tab.id === tabId)).toBe(false)

  const { container, rerender } = render(createElement(PreviewTilePane, { tabId }))

  expect(container.querySelector(`[data-pane="${tabId}"]`)).not.toBeNull()

  setPreviewScope('default')
  closeRightRailTab(tabId)
  rerender(createElement(PreviewTilePane, { tabId }))
  expect(container.querySelector(`[data-pane="${tabId}"]`)).toBeNull()
})
