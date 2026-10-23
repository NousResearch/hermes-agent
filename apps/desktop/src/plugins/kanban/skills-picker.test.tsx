import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { useState } from 'react'
import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest'

// eslint-disable-next-line no-restricted-imports
import { registerPluginLocales } from '@/i18n/plugin-i18n'

import { SkillsPicker } from './board'
import { en, KANBAN_LOCALES } from './i18n'

vi.mock('@/hermes', () => ({ setApiRequestProfile: vi.fn() }))

beforeAll(() => {
  Element.prototype.scrollIntoView = vi.fn()
})

let disposeLocales: () => void

beforeEach(() => {
  disposeLocales = registerPluginLocales('kanban', KANBAN_LOCALES)
})

afterEach(() => {
  cleanup()
  disposeLocales()
})

const OPTIONS = ['github-code-review', 'docs-writer', 'translation']

function Harness({ onChange }: { onChange: (next: string[]) => void }) {
  const [value, setValue] = useState<string[]>([])

  return (
    <SkillsPicker
      onChange={next => {
        setValue(next)
        onChange(next)
      }}
      options={OPTIONS}
      value={value}
    />
  )
}

describe('SkillsPicker', () => {
  it('picks several skills and toggles one back off', () => {
    const onChange = vi.fn()

    render(<Harness onChange={onChange} />)
    fireEvent.click(screen.getByRole('button', { name: en.skillsPlaceholder }))
    fireEvent.click(screen.getByText('translation'))
    fireEvent.click(screen.getByText('github-code-review'))

    expect(onChange).toHaveBeenLastCalledWith(['translation', 'github-code-review'])

    fireEvent.click(screen.getByText('translation'))

    expect(onChange).toHaveBeenLastCalledWith(['github-code-review'])
  })

  it('filters the list by what is typed', () => {
    render(<Harness onChange={vi.fn()} />)
    fireEvent.click(screen.getByRole('button', { name: en.skillsPlaceholder }))
    fireEvent.change(screen.getByPlaceholderText(en.searchSkills), { target: { value: 'docs' } })

    expect(screen.getByText('docs-writer')).toBeTruthy()
    expect(screen.queryByText('translation')).toBeNull()
  })

  it('is disabled when the assignee has no skills', () => {
    render(<SkillsPicker onChange={vi.fn()} options={[]} value={[]} />)

    expect((screen.getByRole('button', { name: en.skillsPlaceholder }) as HTMLButtonElement).disabled).toBe(true)
  })
})
