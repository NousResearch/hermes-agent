import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'

import QuestionFormRenderer from './question-form-embed'

afterEach(cleanup)

describe('QuestionFormRenderer', () => {
  it('renders display-only questions from valid JSON', () => {
    render(
      <QuestionFormRenderer
        code={JSON.stringify({
          title: 'Decisions',
          fields: [
            { id: 'tone', label: 'Pick tone', options: ['dense', 'spacious'], default: 'dense', required: true },
            { id: 'ship', label: 'Ship now?', type: 'checkbox', default: true }
          ]
        })}
      />
    )

    expect(screen.getByText('Decisions')).toBeTruthy()
    expect(screen.getByText('Pick tone')).toBeTruthy()
    expect(screen.getByText('Required')).toBeTruthy()
    expect(screen.getByText('Ship now?')).toBeTruthy()
  })

  it('falls back on invalid JSON', () => {
    render(<QuestionFormRenderer code='{"fields": []}' />)

    expect(screen.getByText('Invalid question-form block')).toBeTruthy()
  })

  it('renders attacker labels as text, not markup', () => {
    const { container } = render(
      <QuestionFormRenderer
        code={JSON.stringify({ fields: [{ id: 'x', label: '<script>alert(1)</script>', default: '<img src=x onerror=alert(1)>' }] })}
      />
    )

    expect(screen.getByText('<script>alert(1)</script>')).toBeTruthy()
    expect(container.querySelector('script')).toBeNull()
    expect(container.querySelector('img')).toBeNull()
  })
})
