import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { DescriptionSection } from './drawer'

vi.mock('./ui', async importOriginal => {
  const actual = await importOriginal<Record<string, unknown>>()

  return {
    ...actual,
    useKanban: () => ({
      cancelEdit: 'Cancel edit',
      description: 'Description',
      editDescription: 'Edit description',
      noDescription: 'No description',
      save: 'Save'
    })
  }
})

afterEach(cleanup)

describe('Kanban task description', () => {
  it('renders stored Markdown as formatted content', () => {
    render(
      <DescriptionSection
        body={'# Human review snapshot\n\n**Accountable human:** Mark\n\n- Approve\n- Hold'}
        onSave={vi.fn()}
      />
    )

    expect(screen.getByRole('heading', { level: 1, name: 'Human review snapshot' })).toBeTruthy()
    expect(screen.getByText('Accountable human:')).toBeTruthy()
    expect(screen.getAllByRole('listitem')).toHaveLength(2)
    expect(screen.queryByText(/^# Human review snapshot$/)).toBeNull()
  })

  it('keeps the original Markdown available in edit mode', () => {
    const body = '# Human review snapshot\n\n**Accountable human:** Mark'
    render(<DescriptionSection body={body} onSave={vi.fn()} />)

    fireEvent.click(screen.getByLabelText('Edit description'))

    expect((screen.getByRole('textbox') as HTMLTextAreaElement).value).toBe(body)
  })

  it('does not expose unsafe HTML or link protocols', () => {
    const { container } = render(
      <DescriptionSection
        body={
          '<script>alert(1)</script><img src=x onerror=alert(1)>\n\n[safe](https://example.com/path)\n\n[x](javascript:alert(1))\n\n[y](file:///etc/passwd)'
        }
        onSave={vi.fn()}
      />
    )

    const safeLink = screen.getByRole('link', { name: 'safe' })
    expect(safeLink.getAttribute('href')).toBe('https://example.com/path')
    expect(safeLink.classList.contains('ref')).toBe(true)
    expect(safeLink.getAttribute('rel')).toBe('noopener noreferrer')
    expect(safeLink.getAttribute('target')).toBe('_blank')
    expect(container.querySelector('script')).toBeNull()
    expect(container.querySelector('[onerror]')).toBeNull()
    expect(container.querySelector('a[href^="javascript:"]')).toBeNull()
    expect(container.querySelector('a[href^="file:"]')).toBeNull()
  })
})
