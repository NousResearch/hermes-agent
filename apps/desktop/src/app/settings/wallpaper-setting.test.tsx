import { fireEvent, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { WallpaperColorControl } from './wallpaper-setting'

describe('WallpaperColorControl', () => {
  afterEach(() => vi.unstubAllGlobals())

  it('previews drag input once per frame and commits only the native change event', () => {
    const onChange = vi.fn()
    const onPreview = vi.fn()
    const frameCallbacks: FrameRequestCallback[] = []

    vi.stubGlobal('requestAnimationFrame', (callback: FrameRequestCallback) => {
      frameCallbacks.push(callback)

      return frameCallbacks.length
    })
    vi.stubGlobal('cancelAnimationFrame', vi.fn())

    render(
      <WallpaperColorControl
        disabled={false}
        label="Accent"
        onChange={onChange}
        onPreview={onPreview}
        value="#112233"
      />
    )

    const input = screen.getByLabelText('Accent') as HTMLInputElement

    fireEvent.input(input, { target: { value: '#445566' } })
    fireEvent.input(input, { target: { value: '#778899' } })

    expect(frameCallbacks).toHaveLength(1)
    expect(input.value).toBe('#778899')
    expect(onPreview).not.toHaveBeenCalled()
    expect(onChange).not.toHaveBeenCalled()

    frameCallbacks.shift()?.(performance.now())

    expect(onPreview).toHaveBeenCalledOnce()
    expect(onPreview).toHaveBeenLastCalledWith('#778899')

    fireEvent.change(input, { target: { value: '#778899' } })

    expect(onChange).toHaveBeenCalledOnce()
    expect(onChange).toHaveBeenCalledWith('#778899')
  })
})
