import { cleanup, render } from '@testing-library/react'
import { afterEach, expect, it } from 'vitest'

import { StatusRow } from './status-row'
import { StatusSection } from './status-section'
import statusStyles from './status-stack.css?inline'

afterEach(cleanup)

it('keeps row padding unchanged when it becomes the first section', () => {
  const group = (
    <StatusSection accessory={<button type="button">Manage</button>} label="Background tasks">
      <StatusRow>Task</StatusRow>
    </StatusSection>
  )

  const samples = [
    <StatusRow key="row">Task</StatusRow>,
    <div className="status-artifacts" key="artifacts">
      <StatusRow>index.html</StatusRow>
    </div>,
    group,
    <div key="wrapped">{group}</div>
  ]

  for (const sample of samples) {
    const fixture = (first: boolean) => (
      <>
        <style>{statusStyles}</style>
        <div data-slot="composer-status-stack">
          {!first && <div data-slot="status-stack-section">Earlier section</div>}
          <div data-slot="status-stack-section" data-testid="section">
            {sample}
          </div>
        </div>
      </>
    )

    const view = render(fixture(false))

    const padding = () =>
      [...view.getByTestId('section').querySelectorAll('.status-row, .status-section-header > *')].map(element => {
        const style = getComputedStyle(element)

        return { align: style.alignSelf, bottom: style.paddingBottom, top: style.paddingTop }
      })

    const later = padding()

    expect(later.length).toBeGreaterThan(0)
    view.rerender(fixture(true))
    expect(padding()).toEqual(later)
    view.unmount()
  }
})

it('balances vertical padding when an artifact is the only status row', () => {
  const { container } = render(
    <>
      <style>{statusStyles}</style>
      <div data-slot="composer-status-stack">
        <div data-slot="status-stack-section">
          <div className="status-artifacts">
            <StatusRow>index.html</StatusRow>
          </div>
        </div>
      </div>
    </>
  )

  const style = getComputedStyle(container.querySelector('[data-slot="status-row"]')!)

  expect(parseFloat(style.paddingTop)).toBeGreaterThan(0)
  expect(style.paddingTop).toBe(style.paddingBottom)
})
