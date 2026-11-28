import { cleanup, render } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'

import { I18nProvider } from '@/i18n'

import { ChoiceLabel } from './choice-row'

afterEach(cleanup)

const renderLabel = (locale: string, choice: string) =>
  render(
    <I18nProvider configClient={null} initialLocale={locale}>
      <ChoiceLabel choice={choice} />
    </I18nProvider>
  )

// The backend embeds an English "(Recommended)" marker in the first choice's
// wire string (`tools/clarify_tool.py::mark_recommended`); the card must strip
// it and show the locale's own label instead of leaking the embedded English
// (#139344).
describe('ChoiceLabel recommended marker', () => {
  it('renders the locale label instead of the embedded English marker', () => {
    const { container } = renderLabel('zh', '从检查点重试 (Recommended)')

    expect(container.textContent).toContain('从检查点重试')
    expect(container.textContent).toContain('（推荐）')
    expect(container.textContent).not.toContain('(Recommended)')
  })

  it('keeps the English label for the en locale', () => {
    const { container } = renderLabel('en', 'Retry from checkpoint (Recommended)')

    expect(container.textContent).toContain('Retry from checkpoint')
    expect(container.textContent).toContain('(Recommended)')
  })

  it('renders a choice without the marker unchanged', () => {
    const { container } = renderLabel('zh', '只是一个选项')

    expect(container.textContent).toContain('只是一个选项')
  })
})
