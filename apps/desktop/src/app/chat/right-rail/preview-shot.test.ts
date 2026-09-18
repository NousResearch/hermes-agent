import { beforeEach, describe, expect, it, vi } from 'vitest'

import { $rightRailActiveTabId } from '@/store/layout'
import { closeRightRail, openPreview, type PreviewTarget } from '@/store/preview'
import { $connection } from '@/store/session'

import { registerPreviewCamera, screenshotActivePreview } from './preview-shot'

function urlTarget(url: string): PreviewTarget {
  return { kind: 'url', label: 'Browser', source: url, url }
}

function fileTarget(path: string): PreviewTarget {
  return { kind: 'file', label: path, path, previewKind: 'text', source: path, url: `file://${path}` }
}

const SHOT = { height: 800, path: '/tmp/composer-images/preview_1.png', width: 1200 }

describe('screenshotActivePreview (desktop_preview action=screenshot)', () => {
  // All URL targets share the singleton Browser tab id, so a camera registered
  // in one test would answer the next — unregister whatever a test installed.
  let cleanups: Array<() => void> = []

  const register = (tabId: string, camera: Parameters<typeof registerPreviewCamera>[1]) => {
    cleanups.push(registerPreviewCamera(tabId, camera))
  }

  beforeEach(() => {
    for (const cleanup of cleanups) {
      cleanup()
    }

    cleanups = []
    closeRightRail()
    $connection.set(null)
    window.localStorage.clear()
  })

  it('fails closed when nothing is open', async () => {
    expect(await screenshotActivePreview()).toMatchObject({ success: false })
  })

  it('answers the path and the host, never the URL or its token', async () => {
    const accessUrl = 'https://preview.example.com/app/page?access_token=s3cret#frag'

    openPreview(urlTarget(accessUrl), 'tool-result')
    register($rightRailActiveTabId.get()!, async () => ({ ...SHOT, title: 'Dashboard', url: accessUrl }))

    const result = await screenshotActivePreview()

    expect(result).toEqual({
      height: 800,
      host: 'preview.example.com',
      kind: 'url',
      path: SHOT.path,
      success: true,
      title: 'Dashboard',
      width: 1200
    })
    expect(JSON.stringify(result)).not.toMatch(/s3cret|access_token|\/app\/page/)
  })

  it('points a file peek at read_file instead of photographing it', async () => {
    openPreview(fileTarget('/work/notes.md'), 'tool-result')

    const result = await screenshotActivePreview()

    expect(result.success).toBe(false)
    expect(result).toMatchObject({ error: expect.stringContaining('read_file') })
  })

  it('reports a camera failure without a path', async () => {
    openPreview(urlTarget('https://example.com'), 'tool-result')
    register($rightRailActiveTabId.get()!, async () => {
      throw new Error('preview capture was empty')
    })

    expect(await screenshotActivePreview()).toEqual({ error: 'preview capture was empty', success: false })
  })

  it('refuses on a remote gateway before anything is written', async () => {
    const camera = vi.fn(async () => ({ ...SHOT, title: 'Page', url: 'https://example.com' }))

    openPreview(urlTarget('https://example.com'), 'tool-result')
    register($rightRailActiveTabId.get()!, camera)
    $connection.set({
      baseUrl: 'http://localhost',
      connectionId: 'remote-fixture',
      isFullscreen: false,
      logs: [],
      mode: 'remote',
      nativeOverlayWidth: 0,
      profile: 'writer',
      token: '',
      windowButtonPosition: null,
      wsUrl: ''
    })

    const result = await screenshotActivePreview()

    expect(result.success).toBe(false)
    expect(result).toMatchObject({ error: expect.stringContaining('remote gateway') })
    expect(camera).not.toHaveBeenCalled()
  })
})
