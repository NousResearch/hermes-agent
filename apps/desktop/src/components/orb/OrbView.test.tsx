import { render, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'

import { defaultOrbConfig } from './orb-url'
import { OrbView } from './OrbView'

afterEach(() => {
  // Remove the WebGPU mock so tests don't leak it into each other.
  Reflect.deleteProperty(window.navigator, 'gpu')
})

describe('OrbView', () => {
  it('renders the standard loader fallback when WebGPU is unavailable', () => {
    // jsdom has no navigator.gpu.
    const { container } = render(<OrbView className="size-16" params={defaultOrbConfig()} />)

    expect(container.querySelector('canvas')).toBeNull()
    // The fallback Loader renders its SVG, not a broken canvas.
    expect(container.querySelector('svg')).not.toBeNull()
  })

  it('mounts a canvas when WebGPU is available, then degrades when init fails', async () => {
    Object.defineProperty(window.navigator, 'gpu', {
      configurable: true,
      value: { requestAdapter: async () => null }
    })

    const { container } = render(<OrbView className="size-16" params={defaultOrbConfig()} />)

    // Canvas mounts first; the (mocked, adapter-less) renderer then fails and
    // the component degrades back to the loader on its own.
    expect(container.querySelector('canvas')).not.toBeNull()
    await waitFor(() => {
      expect(container.querySelector('canvas')).toBeNull()
    })
    expect(container.querySelector('svg')).not.toBeNull()
  })
})
