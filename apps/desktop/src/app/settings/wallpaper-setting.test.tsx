import { fireEvent, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { WallpaperColorControl, WallpaperSlider } from './wallpaper-setting'

describe('WallpaperSlider', () => {
  afterEach(() => vi.unstubAllGlobals())

  it('coalesces rapid input to one store update per animation frame and flushes the final value', () => {
    const onChange = vi.fn()
    const frameCallbacks: FrameRequestCallback[] = []
    const cancelAnimationFrame = vi.fn()

    vi.stubGlobal('requestAnimationFrame', (callback: FrameRequestCallback) => {
      frameCallbacks.push(callback)

      return frameCallbacks.length
    })
    vi.stubGlobal('cancelAnimationFrame', cancelAnimationFrame)

    render(
      <WallpaperSlider
        disabled={false}
        label="Opacity"
        max={100}
        min={0}
        onChange={onChange}
        value={20}
        valueLabel="20%"
      />
    )

    const input = screen.getByLabelText('Opacity')

    fireEvent.change(input, { target: { value: '30' } })
    fireEvent.change(input, { target: { value: '40' } })

    expect(frameCallbacks).toHaveLength(1)
    expect(onChange).not.toHaveBeenCalled()

    frameCallbacks.shift()?.(performance.now())

    expect(onChange).toHaveBeenCalledOnce()
    expect(onChange).toHaveBeenLastCalledWith(40)

    fireEvent.change(input, { target: { value: '55' } })
    fireEvent.pointerUp(input)

    expect(cancelAnimationFrame).toHaveBeenCalledOnce()
    expect(onChange).toHaveBeenCalledTimes(2)
    expect(onChange).toHaveBeenLastCalledWith(55)
  })
})

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
