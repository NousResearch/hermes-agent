import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'

import { SegmentedControl } from '@/components/ui/segmented-control'

import { ListRow, ToggleRow } from './primitives'

afterEach(cleanup)

it('connects a row title and description to a grouped control without leaking row props', () => {
  const { container } = render(
    <ListRow
      action={({ titleId, descriptionId }) => (
        <SegmentedControl
          ariaDescribedBy={descriptionId}
          ariaLabelledBy={titleId}
          onChange={vi.fn()}
          options={[
            { id: 'on', label: 'On' },
            { id: 'off', label: 'Off' }
          ]}
          value="on"
        />
      )}
      description="Choose when Hermes stays awake"
      id="keep-awake-row"
      title="Keep awake"
    />
  )

  const group = screen.getByRole('group', { name: 'Keep awake', description: 'Choose when Hermes stays awake' })
  expect(group).toBeTruthy()
  expect(container.querySelector('#keep-awake-row')).toBeTruthy()
  expect(group.hasAttribute('ariaLabelledBy')).toBe(false)
  expect(group.hasAttribute('ariaDescribedBy')).toBe(false)
})

it('keeps existing explicit ToggleRow names and plain ListRow actions', () => {
  render(
    <>
      <ToggleRow checked description="Allow alerts" label="Notifications" onChange={vi.fn()} />
      <ListRow action={<button type="button">Open</button>} title="Advanced" />
    </>
  )

  expect(screen.getByRole('switch', { name: 'Notifications' })).toBeTruthy()
  expect(screen.getByRole('button', { name: 'Open' })).toBeTruthy()
})

it('names representative appearance choices from their row titles and descriptions', () => {
  render(
    <>
      {[
        { title: 'Theme', description: 'Choose light or dark mode', options: ['Light', 'Dark'] },
        { title: 'UI scale', description: 'Currently 100 percent', options: ['90%', '100%'] },
        { title: 'Chat text size', description: 'Adjust conversation text', options: ['Small', 'Large'] }
      ].map(({ title, description, options }) => (
        <ListRow
          action={({ titleId, descriptionId }) => (
            <SegmentedControl
              ariaDescribedBy={descriptionId}
              ariaLabelledBy={titleId}
              onChange={vi.fn()}
              options={options.map(option => ({ id: option, label: option }))}
              value={options[0]}
            />
          )}
          description={description}
          key={title}
          title={title}
        />
      ))}
    </>
  )

  expect(screen.getByRole('group', { name: 'Theme', description: 'Choose light or dark mode' })).toBeTruthy()
  expect(screen.getByRole('group', { name: 'UI scale', description: 'Currently 100 percent' })).toBeTruthy()
  expect(screen.getByRole('group', { name: 'Chat text size', description: 'Adjust conversation text' })).toBeTruthy()
})
