import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { useState } from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { registry } from '../registry'

import { Slot } from './slot'

const disposers: Array<() => void> = []

afterEach(() => {
  cleanup()
  disposers.splice(0).forEach(dispose => dispose())
  vi.restoreAllMocks()
})

describe('Slot contribution render boundary', () => {
  it('hot-mounts and replaces hook-using plugin renders without changing the host hook order', () => {
    const area = 'test.plugin-hook-boundary'
    const id = 'stateful-plugin'

    function FirstPluginRender() {
      const [count, setCount] = useState(0)

      return <button onClick={() => setCount(value => value + 1)}>first:{count}</button>
    }

    function ReloadedPluginRender() {
      const [count, setCount] = useState(10)
      const [suffix] = useState('ready')

      return (
        <button onClick={() => setCount(value => value + 1)}>
          reloaded:{count}:{suffix}
        </button>
      )
    }

    render(<Slot area={area} />)

    act(() => {
      disposers.push(registry.register({ area, id, render: FirstPluginRender, source: 'disk' }))
    })

    fireEvent.click(screen.getByRole('button', { name: 'first:0' }))
    expect(screen.getByRole('button', { name: 'first:1' })).toBeTruthy()

    act(() => {
      disposers.push(registry.register({ area, id, render: ReloadedPluginRender, source: 'disk' }))
    })

    fireEvent.click(screen.getByRole('button', { name: 'reloaded:10:ready' }))
    expect(screen.getByRole('button', { name: 'reloaded:11:ready' })).toBeTruthy()
  })

  it('lets the contribution boundary contain a render-time plugin failure', () => {
    const area = 'test.plugin-error-boundary'
    const id = 'broken-plugin'

    vi.spyOn(console, 'error').mockImplementation(() => undefined)
    render(<Slot area={area} />)

    act(() => {
      disposers.push(
        registry.register({
          area,
          id,
          render: () => {
            throw new Error('plugin render failed')
          },
          source: 'disk'
        })
      )
    })

    expect(screen.getByRole('button', { name: id })).toBeTruthy()
  })
})
