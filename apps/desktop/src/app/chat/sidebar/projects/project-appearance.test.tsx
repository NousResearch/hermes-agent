import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { ProjectAppearancePicker } from './project-appearance'

afterEach(cleanup)

describe('ProjectAppearancePicker', () => {
  it('finds and selects any Lucide icon by name', async () => {
    const onIcon = vi.fn()

    render(
      <ProjectAppearancePicker color={null} icon={null} noColorLabel="No color" onColor={vi.fn()} onIcon={onIcon} />
    )

    fireEvent.change(screen.getByRole('textbox', { name: 'Search' }), { target: { value: 'briefcase business' } })
    fireEvent.click(await screen.findByRole('button', { name: 'briefcase-business' }))

    expect(onIcon).toHaveBeenCalledWith('lucide:briefcase-business')
  })

  it('keeps the selected Lucide icon visible when the search is cleared', async () => {
    render(
      <ProjectAppearancePicker
        color="#ff0000"
        icon="lucide:zodiac-virgo"
        noColorLabel="No color"
        onColor={vi.fn()}
        onIcon={vi.fn()}
      />
    )

    expect(await screen.findByRole('button', { name: 'zodiac-virgo' })).toBeTruthy()
  })

  it('caps broad searches so typing does not mount the full dynamic catalogue', async () => {
    const { container } = render(
      <ProjectAppearancePicker color={null} icon={null} noColorLabel="No color" onColor={vi.fn()} onIcon={vi.fn()} />
    )

    fireEvent.change(screen.getByRole('textbox', { name: 'Search' }), { target: { value: 'a' } })
    await screen.findByRole('button', { name: 'a-arrow-down' })

    expect(container.querySelectorAll('[data-slot="project-icon-results"] button')).toHaveLength(36)
  })
})
