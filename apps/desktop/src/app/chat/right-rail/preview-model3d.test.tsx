import { cleanup, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

const viewer = vi.hoisted(() => ({
  destroy: vi.fn(),
  setBackground: vi.fn(),
  instances: [] as { host: HTMLElement; load: ReturnType<typeof vi.fn>; params: Record<string, (() => void) | unknown> }[]
}))

// jsdom has no WebGL: stand in for Online3DViewer's embedded viewer.
vi.mock('online-3d-viewer', () => ({
  EmbeddedViewer: class {
    Destroy = viewer.destroy
    LoadModelFromFileList = vi.fn()
    GetViewer = () => ({ SetBackgroundColor: viewer.setBackground })

    constructor(host: HTMLElement, params: Record<string, unknown>) {
      viewer.instances.push({ host, load: this.LoadModelFromFileList, params })
    }
  },
  RGBAColor: class {
    constructor(
      public r: number,
      public g: number,
      public b: number,
      public a: number
    ) {}
  }
}))

import { Model3dPreview } from './preview-model3d'

const DATA_URL = 'data:application/octet-stream;base64,AAEC'

describe('Model3dPreview', () => {
  beforeEach(() => {
    viewer.instances.length = 0
    viewer.destroy.mockClear()
    viewer.setBackground.mockClear()
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => ({ blob: async () => new Blob([new Uint8Array([0, 1, 2])]) }))
    )
  })

  afterEach(() => {
    cleanup()
    vi.unstubAllGlobals()
  })

  it('hands the file bytes to the viewer under the original file name', async () => {
    render(<Model3dPreview dataUrl={DATA_URL} name="part.stl" />)

    await waitFor(() => expect(viewer.instances).toHaveLength(1))

    const [{ load }] = viewer.instances
    const files = load.mock.calls[0]?.[0] as File[]

    expect(files).toHaveLength(1)
    expect(files[0]?.name).toBe('part.stl')
    expect(files[0]?.size).toBe(3)
  })

  it('clears the loading state once the model is loaded', async () => {
    render(<Model3dPreview dataUrl={DATA_URL} name="part.stl" />)
    await waitFor(() => expect(viewer.instances).toHaveLength(1))
    expect(screen.getByRole('status')).toBeTruthy()
    ;(viewer.instances[0]?.params.onModelLoaded as () => void)()
    await waitFor(() => expect(screen.queryByRole('status')).toBeNull())
  })

  it('shows an error when the viewer cannot parse the model', async () => {
    render(<Model3dPreview dataUrl={DATA_URL} name="broken.stl" />)
    await waitFor(() => expect(viewer.instances).toHaveLength(1))
    ;(viewer.instances[0]?.params.onModelLoadFailed as () => void)()
    expect(await screen.findByText(/load this 3D model/i)).toBeTruthy()
  })

  it('repaints the viewer background when the app theme changes', async () => {
    render(<Model3dPreview dataUrl={DATA_URL} name="part.obj" />)
    await waitFor(() => expect(viewer.instances).toHaveLength(1))
    expect(viewer.setBackground).not.toHaveBeenCalled()

    window.document.documentElement.setAttribute('data-hermes-theme', 'other')
    await waitFor(() => expect(viewer.setBackground).toHaveBeenCalledTimes(1))
    window.document.documentElement.removeAttribute('data-hermes-theme')
  })

  it('destroys the viewer on unmount', async () => {
    const { unmount } = render(<Model3dPreview dataUrl={DATA_URL} name="part.stl" />)

    await waitFor(() => expect(viewer.instances).toHaveLength(1))
    unmount()
    expect(viewer.destroy).toHaveBeenCalledTimes(1)
  })
})
