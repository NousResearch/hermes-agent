import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest'

// Radix Select calls scrollIntoView / pointer-capture APIs jsdom lacks.
beforeAll(() => {
  Element.prototype.scrollIntoView = vi.fn()
  Element.prototype.hasPointerCapture = vi.fn(() => false)
  Element.prototype.releasePointerCapture = vi.fn()
})

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
})

async function renderField(value: unknown, onChange = vi.fn()) {
  const { ReasoningOverridesField } = await import('./reasoning-overrides-field')

  render(<ReasoningOverridesField onChange={onChange} value={value} />)

  return onChange
}

describe('ReasoningOverridesField', () => {
  it('renders the persisted {model: effort} map as editable rows', async () => {
    await renderField({ 'deepseek-v4-flash': 'medium', 'claude-opus': 'high' })

    expect((screen.getByLabelText('Model 1') as HTMLInputElement).value).toBe('deepseek-v4-flash')
    expect((screen.getByLabelText('Model 2') as HTMLInputElement).value).toBe('claude-opus')
    // The desktop surface had zero coverage for this key — users hand-edited YAML (#117908).
    expect(screen.getByTestId('reasoning-overrides-field')).toBeTruthy()
    expect(screen.getByText('Add model override')).toBeTruthy()
  })

  it('editing a model id re-emits the map under the new key', async () => {
    const onChange = await renderField({ 'deepseek-v4-flash': 'medium' })

    fireEvent.change(screen.getByLabelText('Model 1'), { target: { value: 'glm-5' } })

    expect(onChange.mock.calls.at(-1)?.[0]).toEqual({ 'glm-5': 'medium' })
  })

  it('removing a row drops its override from the emitted map', async () => {
    const onChange = await renderField({ 'deepseek-v4-flash': 'medium', 'claude-opus': 'high' })

    fireEvent.click(screen.getAllByLabelText('Remove')[0])

    expect(onChange.mock.calls.at(-1)?.[0]).toEqual({ 'claude-opus': 'high' })
  })

  it('adding a blank row does not persist an empty override', async () => {
    const onChange = await renderField({ 'deepseek-v4-flash': 'medium' })

    fireEvent.click(screen.getByText('Add model override'))

    // The empty row stays visible; only entries with a model id are emitted.
    expect(onChange.mock.calls.at(-1)?.[0]).toEqual({ 'deepseek-v4-flash': 'medium' })
    expect(screen.getAllByLabelText(/^Model \d+$/)).toHaveLength(2)
  })

  it('ignores non-object config values instead of crashing', async () => {
    await renderField('not-a-map')

    expect(screen.queryByTestId('reasoning-overrides-field')).toBeTruthy()
    expect(screen.queryAllByLabelText(/^Model \d+$/)).toHaveLength(0)
  })
})
