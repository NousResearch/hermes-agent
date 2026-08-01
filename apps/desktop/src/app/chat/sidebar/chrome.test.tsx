import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'

import { SidebarRowLink } from './chrome'

afterEach(cleanup)

describe('SidebarRowLink', () => {
  it('wraps long session titles instead of forcing a single truncated line', () => {
    render(<SidebarRowLink>Fix background process exit error during Hermes auto-update</SidebarRowLink>)

    const title = screen.getByText('Fix background process exit error during Hermes auto-update')
    expect(title.className).toContain('line-clamp-2')
    expect(title.className).not.toContain('truncate')
  })
})
