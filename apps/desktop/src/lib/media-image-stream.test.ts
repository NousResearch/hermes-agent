import { afterEach, describe, expect, it, vi } from 'vitest'

import { $connection } from '@/store/session'

import { downloadGatewayMediaFile, resolveMediaDisplaySrc } from './media'

const path = '/srv/exports/写真 full size.png'

afterEach(() => {
  vi.unstubAllGlobals()
  $connection.set(null)
})

function remote(api: ReturnType<typeof vi.fn>) {
  vi.stubGlobal('window', { hermesDesktop: { api } })
  $connection.set({ mode: 'remote', connectionId: 'studio', profile: 'images', token: 'private-token' } as never)
}

describe('large remote image delivery', () => {
  it('streams a size-rejected image without placing credentials in the URL', async () => {
    remote(
      vi.fn().mockRejectedValue(new Error('Error invoking remote method: Error: 413: {"detail":"File too large"}'))
    )

    const source = new URL(await resolveMediaDisplaySrc(path))

    expect(source.protocol).toBe('hermes-media:')
    expect(source.hostname).toBe('remote')
    expect(decodeURIComponent(source.pathname.slice(1))).toBe(path)
    expect(source.searchParams.get('connectionId')).toBe('studio')
    expect(source.searchParams.get('profile')).toBe('images')
    expect(source.href).not.toContain('private-token')
  })

  it('keeps the originating gateway when the size rejection arrives after a switch', async () => {
    let reject!: (error: Error) => void
    remote(
      vi.fn(
        () =>
          new Promise((_resolve, onReject) => {
            reject = onReject
          })
      )
    )

    const pending = resolveMediaDisplaySrc(path)
    $connection.set({ mode: 'local', connectionId: 'local' } as never)
    reject(new Error('413: File too large'))

    expect(new URL(await pending).searchParams.get('connectionId')).toBe('studio')
  })

  it.each(['401: Unauthorized', '403: Forbidden', '404: File not found', '502: Bad gateway', 'Timed out'])(
    'does not treat %s as permission to use another delivery path',
    async message => {
      const error = new Error(message)
      remote(vi.fn().mockRejectedValue(error))

      await expect(resolveMediaDisplaySrc(path)).rejects.toBe(error)
    }
  )

  it.each(['image.svg', 'page.html', 'report.pdf'])('does not stream active or non-raster content: %s', async name => {
    remote(vi.fn().mockRejectedValue(new Error('413: File too large')))

    await expect(resolveMediaDisplaySrc(`/srv/${name}`)).rejects.toThrow('413')
  })

  it('retains the existing data URL behavior for normal images and older backends', async () => {
    const dataUrl = 'data:image/png;base64,aGVsbG8='
    remote(vi.fn().mockResolvedValue({ dataUrl }))

    await expect(resolveMediaDisplaySrc(path)).resolves.toBe(dataUrl)
  })

  it('downloads a streamed image from its original gateway with the original filename', async () => {
    const saveGatewayFile = vi.fn().mockResolvedValue({ saved: true })
    vi.stubGlobal('window', { hermesDesktop: { saveGatewayFile } })
    $connection.set({ mode: 'remote', connectionId: 'different', profile: 'other' } as never)

    await downloadGatewayMediaFile(
      `hermes-media://remote/${encodeURIComponent(path)}?connectionId=studio&profile=images`
    )

    expect(saveGatewayFile).toHaveBeenCalledWith({
      connectionId: 'studio',
      path,
      profile: 'images',
      suggestedName: '写真 full size.png'
    })
  })
})
