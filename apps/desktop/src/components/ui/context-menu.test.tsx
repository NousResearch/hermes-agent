import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'

import { ContextMenu, ContextMenuContent, ContextMenuItem, ContextMenuTrigger } from './context-menu'

afterEach(cleanup)

describe('ContextMenuContent', () => {
  it('clears the fixed titlebar layer', () => {
    render(
      <ContextMenu>
        <ContextMenuTrigger>Open</ContextMenuTrigger>
        <ContextMenuContent>
          <ContextMenuItem>Item</ContextMenuItem>
        </ContextMenuContent>
      </ContextMenu>
    )

    fireEvent.contextMenu(screen.getByText('Open'))

    expect(screen.getByText('Item').closest('[data-slot="context-menu-content"]')?.className).toContain(
      'z-(--z-over-modal-content)'
    )
  })
})
