/** Interaction contracts adapted from webtecnica's #67523, including model-only drafts. */
import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { useState } from 'react'
import { afterEach, beforeAll, describe, expect, it, vi } from 'vitest'

import { DelegationModelProviderField } from './delegation-model-provider-field'
import { delegationModelsCopy } from './delegation-models-copy'

beforeAll(() => {
  Element.prototype.scrollIntoView = vi.fn()
  Element.prototype.hasPointerCapture = vi.fn(() => false)
  Element.prototype.releasePointerCapture = vi.fn()
})
afterEach(cleanup)

const copy = delegationModelsCopy('en')

const providers = [
  { slug: 'provider-a', name: 'Provider A', authenticated: true, models: ['model-a', 'model-b'] },
  { slug: 'custom:worker', name: 'Custom worker', authenticated: true, models: [] }
]

function Harness({ initial = { provider: '', model: '' }, onChange = vi.fn() }) {
  const [pair, setPair] = useState(initial)

  return <DelegationModelProviderField {...pair} copy={copy} onChange={(next, changed) => {
    setPair(next)
    onChange(next, changed)
  }} providers={providers} />
}

describe('guided delegation picker', () => {
  it('represents full inheritance without emitting a write on mount', () => {
    const onChange = vi.fn()
    render(<Harness onChange={onChange} />)
    expect(screen.getByText(copy.mainProvider)).toBeTruthy()
    expect(onChange).not.toHaveBeenCalled()
  })

  it('keeps a model-only override visible and editable', async () => {
    const onChange = vi.fn()
    render(<Harness initial={{ provider: '', model: 'inherited-provider-model' }} onChange={onChange} />)
    const model = screen.getByRole('textbox', { name: copy.model })
    expect((model as HTMLInputElement).value).toBe('inherited-provider-model')
    fireEvent.change(model, { target: { value: 'inherited-provider-model-next' } })
    expect(onChange).toHaveBeenLastCalledWith({ provider: '', model: 'inherited-provider-model-next' }, false)
  })

  it('uses the provider catalog and emits the selected pair to the local draft', async () => {
    const onChange = vi.fn()
    render(<Harness onChange={onChange} />)
    fireEvent.click(screen.getByRole('combobox', { name: copy.provider }))
    fireEvent.click(screen.getByRole('option', { name: 'Provider A' }))
    fireEvent.click(screen.getByRole('combobox', { name: copy.model }))
    fireEvent.click(screen.getByRole('option', { name: 'model-b' }))
    expect(onChange).toHaveBeenLastCalledWith({ provider: 'provider-a', model: 'model-b' }, false)
  })

  it('allows models not in a provider catalog', async () => {
    const onChange = vi.fn()
    render(<Harness initial={{ provider: 'provider-a', model: 'model-a' }} onChange={onChange} />)
    fireEvent.click(screen.getByRole('combobox', { name: copy.model }))
    fireEvent.click(screen.getByRole('option', { name: copy.customModel }))
    fireEvent.change(screen.getByRole('textbox', { name: copy.model }), { target: { value: 'private/model:tag' } })
    expect(onChange).toHaveBeenLastCalledWith({ provider: 'provider-a', model: 'private/model:tag' }, false)
  })

  it('resyncs controlled values when the owner changes', () => {
    const props = { copy, providers, onChange: vi.fn() }
    const view = render(<DelegationModelProviderField {...props} model="first" provider="" />)
    view.rerender(<DelegationModelProviderField {...props} model="second" provider="custom:worker" />)
    expect((screen.getByRole('textbox', { name: copy.model }) as HTMLInputElement).value).toBe('second')
    expect(props.onChange).not.toHaveBeenCalled()
  })
})
