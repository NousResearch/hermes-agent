import { act, cleanup, render } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { $petActivity, $petMotion, type PetInfo } from '@/store/pet'

import { PetSprite } from './pet-sprite'

const info: PetInfo = {
  enabled: true,
  displayName: 'V2 Test',
  frameH: 208,
  frameW: 192,
  framesPerState: 6,
  lookDirectionCount: 16,
  loopMs: 1100,
  mime: 'image/webp',
  scale: 1,
  spriteVersionNumber: 2,
  spritesheetBase64: 'AA==',
  stateRows: ['idle', 'running-right', 'running-left', 'waving', 'jumping', 'failed', 'waiting', 'running', 'review']
}

describe('PetSprite v2 gaze', () => {
  let nextFrame: FrameRequestCallback | undefined
  let drawImage: ReturnType<typeof vi.fn>

  beforeEach(() => {
    $petActivity.set({})
    $petMotion.set(null)
    drawImage = vi.fn()

    class ReadyImage {
      complete = true
      naturalWidth = 1536
      src = ''
    }

    vi.stubGlobal('Image', ReadyImage)
    vi.stubGlobal(
      'requestAnimationFrame',
      vi.fn((callback: FrameRequestCallback) => {
        nextFrame = callback

        return 1
      })
    )
    vi.stubGlobal('cancelAnimationFrame', vi.fn())
    vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue({
      clearRect: vi.fn(),
      drawImage,
      imageSmoothingEnabled: false
    } as unknown as CanvasRenderingContext2D)
    vi.spyOn(HTMLCanvasElement.prototype, 'getBoundingClientRect').mockReturnValue({
      bottom: 100,
      height: 100,
      left: 0,
      right: 100,
      top: 0,
      width: 100,
      x: 0,
      y: 0,
      toJSON: () => ({})
    })
  })

  afterEach(() => {
    cleanup()
    vi.restoreAllMocks()
    vi.unstubAllGlobals()
  })

  it('draws the fixed v2 cell nearest the pointer while idle', () => {
    render(<PetSprite info={info} lookAtPointer />)

    act(() => nextFrame?.(0))
    expect(drawImage).toHaveBeenLastCalledWith(expect.anything(), 6 * 192, 0, 192, 208, 0, 0, 192, 208)

    act(() => {
      window.dispatchEvent(new MouseEvent('mousemove', { clientX: 150, clientY: 50 }))
      nextFrame?.(16)
    })

    expect(drawImage).toHaveBeenLastCalledWith(expect.anything(), 4 * 192, 9 * 208, 192, 208, 0, 0, 192, 208)
  })

  it('keeps agent activity above cosmetic gaze', () => {
    $petActivity.set({ busy: true })
    render(<PetSprite info={info} lookAtPointer />)

    act(() => {
      window.dispatchEvent(new MouseEvent('mousemove', { clientX: 150, clientY: 50 }))
      nextFrame?.(0)
    })

    expect(drawImage).toHaveBeenLastCalledWith(expect.anything(), 0, 7 * 208, 192, 208, 0, 0, 192, 208)
  })

  it('never reads look rows from a v1 package', () => {
    render(<PetSprite info={{ ...info, lookDirectionCount: 0, spriteVersionNumber: 1 }} lookAtPointer />)

    act(() => {
      window.dispatchEvent(new MouseEvent('mousemove', { clientX: 150, clientY: 50 }))
      nextFrame?.(0)
    })

    expect(drawImage).toHaveBeenLastCalledWith(expect.anything(), 0, 0, 192, 208, 0, 0, 192, 208)
  })
})
