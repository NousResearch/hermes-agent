import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import type { ReactNode } from 'react'
import { afterEach, beforeAll, expect, it, vi } from 'vitest'

vi.mock('./right-rail/preview', () => ({
  PreviewTilePane: () => null
}))

vi.mock('./right-rail/preview-console-store', () => ({
  forgetPreviewConsole: () => undefined
}))

import type { MenuKit } from '@/components/ui/actions-menu'
import { registry } from '@/contrib/registry'
import { $previewTabs, closeRightRail, openPreview } from '@/store/preview'

import { watchPreviewTiles } from './preview-tile'

// The zone menu calls the registered pane's `tabMenuPrefix(kit)` each time it
// opens. A stand-in kit renders each row as a button so a click reaches the
// same `onSelect` the native menu fires.
const Item = ({ children, onSelect }: { children?: ReactNode; onSelect?: (event: Event) => void }) => (
  <button onClick={() => onSelect?.(new Event('select'))} type="button">
    {children}
  </button>
)

const Passthrough = ({ children }: { children?: ReactNode }) => <>{children}</>

const kit = {
  copyAppearance: 'menu-item',
  Item,
  Label: Passthrough,
  Separator: () => null,
  Sub: Passthrough,
  SubContent: Passthrough,
  SubTrigger: Passthrough
} as unknown as MenuKit

beforeAll(() => {
  watchPreviewTiles()
})

afterEach(() => {
  cleanup()
  closeRightRail()
})

/** Open the zone menu for `tabId` the way the strip does, click the pin row,
 *  and close the menu again. Returns the label the row offered. */
function clickPinRow(tabId: string): string {
  const pane = registry.getArea('panes').find(entry => entry.id === `preview-tile:${tabId}`)
  const prefix = (pane?.data as { tabMenuPrefix?: (kit: MenuKit) => ReactNode } | undefined)?.tabMenuPrefix

  render(<>{prefix?.(kit)}</>)
  const row = screen.getByRole('button', { name: /pin/i })
  const label = row.textContent ?? ''

  fireEvent.click(row)
  cleanup()

  return label
}

it('offers Unpin after a pin and Pin after an unpin, toggling the stored flag each time', () => {
  openPreview({ kind: 'url', label: 'Browser', source: 'https://example.com', url: 'https://example.com' })
  const tabId = $previewTabs.get()[0]!.id
  const pinned = () => $previewTabs.get().find(tab => tab.id === tabId)?.pinned

  expect(clickPinRow(tabId)).toBe('Pin to workspace')
  expect(pinned()).toBe(true)

  expect(clickPinRow(tabId)).toBe('Unpin from workspace')
  expect(pinned()).toBe(false)

  expect(clickPinRow(tabId)).toBe('Pin to workspace')
  expect(pinned()).toBe(true)
})
