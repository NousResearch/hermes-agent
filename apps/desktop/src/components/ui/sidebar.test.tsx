import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'

import { SidebarContent, SidebarGroupContent } from './sidebar'

describe('Sidebar overflow guards', () => {
  it('clips horizontal overflow in the sidebar scroller', () => {
    render(
      <SidebarContent data-testid="sidebar-content">
        <div style={{ width: '200%' }}>wide child</div>
      </SidebarContent>
    )

    expect(screen.getByTestId('sidebar-content').className).toContain('overflow-x-hidden')
    expect(screen.getByTestId('sidebar-content').className).toContain('min-w-0')
  })

  it('keeps section bodies shrinkable inside the sidebar width', () => {
    render(
      <SidebarGroupContent data-testid="sidebar-group-content">
        section body
      </SidebarGroupContent>
    )

    expect(screen.getByTestId('sidebar-group-content').className).toContain('min-w-0')
  })
})
