import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { CUSTOM_ENDPOINT_FORM_AREA, type CustomEndpointFormProps } from '@/lib/custom-endpoint-form'

import { registry } from '../../contrib/registry'

import { CustomEndpointFormSlot } from './custom-endpoint-form-slot'

const disposers: Array<() => void> = []

const baseProps: CustomEndpointFormProps = {
  profile: 'default',
  draft: {
    id: '',
    name: 'My endpoint',
    baseUrl: 'http://127.0.0.1:8081/v1',
    model: 'm1',
    apiMode: '',
    contextLength: '',
    makeDefault: false,
    discoverModels: true,
    hasApiKey: false
  },
  saved: null,
  saving: false
}

afterEach(() => {
  cleanup()
  disposers.splice(0).forEach(dispose => dispose())
  vi.restoreAllMocks()
})

describe('CustomEndpointFormSlot', () => {
  it('renders nothing when no contribution is registered', () => {
    const { container } = render(<CustomEndpointFormSlot {...baseProps} />)
    expect(container.innerHTML).toBe('')
  })

  it('hands the draft and save snapshot to a data.render contribution', () => {
    const seen: CustomEndpointFormProps[] = []
    disposers.push(
      registry.register({
        area: CUSTOM_ENDPOINT_FORM_AREA,
        id: 'also-create',
        source: 'disk',
        data: {
          render: (props: CustomEndpointFormProps) => {
            seen.push(props)
            return <span>profiles:{props.profile ?? 'active'}</span>
          }
        }
      })
    )

    render(<CustomEndpointFormSlot {...baseProps} />)
    expect(screen.getByText('profiles:default')).toBeTruthy()
    expect(seen.at(-1)?.draft.name).toBe('My endpoint')
    expect(seen.at(-1)?.draft.hasApiKey).toBe(false)
    expect(seen.at(-1)?.saved).toBeNull()
  })

  it('re-renders contributions with the saved endpoint after a save', () => {
    disposers.push(
      registry.register({
        area: CUSTOM_ENDPOINT_FORM_AREA,
        id: 'also-create',
        source: 'disk',
        data: {
          render: (props: CustomEndpointFormProps) =>
            props.saved ? <span>saved:{props.saved.id}</span> : <span>unsaved</span>
        }
      })
    )

    const { rerender } = render(<CustomEndpointFormSlot {...baseProps} />)
    expect(screen.getByText('unsaved')).toBeTruthy()

    rerender(
      <CustomEndpointFormSlot
        {...baseProps}
        saved={{
          id: 'my-endpoint',
          name: 'My endpoint',
          base_url: 'http://127.0.0.1:8081/v1',
          model: 'm1',
          has_api_key: true,
          savedAt: 123
        }}
      />
    )
    expect(screen.getByText('saved:my-endpoint')).toBeTruthy()
  })

  it('contains a throwing contribution without taking the editor down', () => {
    vi.spyOn(console, 'error').mockImplementation(() => undefined)
    disposers.push(
      registry.register({
        area: CUSTOM_ENDPOINT_FORM_AREA,
        id: 'broken',
        source: 'disk',
        data: {
          render: () => {
            throw new Error('plugin render failed')
          }
        }
      })
    )
    disposers.push(
      registry.register({
        area: CUSTOM_ENDPOINT_FORM_AREA,
        id: 'healthy',
        source: 'disk',
        data: { render: () => <span>healthy</span> }
      })
    )

    render(<CustomEndpointFormSlot {...baseProps} />)
    expect(screen.getByText('healthy')).toBeTruthy()
  })
})
