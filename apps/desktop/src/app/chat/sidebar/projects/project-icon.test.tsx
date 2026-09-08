import { cleanup, render, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'

import { ProjectIcon, projectIconComponent } from './project-icon'

afterEach(cleanup)

describe('ProjectIcon', () => {
  it('keeps rendering legacy Codicon project values', () => {
    const { container } = render(<ProjectIcon name="folder-library" size={16} />)

    expect(container.querySelector('.codicon-folder-library')).toBeTruthy()
  })

  it('renders a namespaced Lucide icon', async () => {
    const { container } = render(<ProjectIcon name="lucide:briefcase-business" size={16} />)

    await waitFor(() =>
      expect(container.querySelector('svg[data-project-icon="lucide:briefcase-business"]')).toBeTruthy()
    )
    const glyph = container.querySelector('svg[data-project-icon="lucide:briefcase-business"]')

    expect(glyph?.getAttribute('aria-hidden')).toBe('true')
  })

  it('falls back when a persisted Lucide name is unknown', () => {
    const { container } = render(<ProjectIcon fallback="folder-library" name="lucide:not-a-real-icon" size={16} />)

    expect(container.querySelector('.codicon-folder-library')).toBeTruthy()
  })

  it('adapts project icons for palette rows', async () => {
    const Icon = projectIconComponent('lucide:briefcase-business', 'folder-library')

    const { container } = render(<Icon aria-label="Project" />)

    await waitFor(() => expect(screen.getByLabelText('Project')).toBeTruthy())
    expect(container.querySelector('svg[data-project-icon="lucide:briefcase-business"]')).toBeTruthy()
  })
})
