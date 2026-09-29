import { randomUUID } from 'node:crypto'

import type { ScreenshotResult, ScreenshotWindow } from './command-screenshot-types'

interface ScreenshotSource {
  id: string
  thumbnail: { isEmpty(): boolean; toPNG(): Uint8Array }
}

interface CaptureDependencies {
  hasScreenPermission(): boolean
  getSources(options: {
    types: ['window']
    thumbnailSize: { width: number; height: number }
    fetchWindowIcons: false
  }): Promise<ScreenshotSource[]>
}

/** A physical gesture grants one renderer one capture, not an arbitrary capture IPC. */
export function createScreenshotCapture({ getSources, hasScreenPermission }: CaptureDependencies) {
  let pending: { id: string; owner: number; window: ScreenshotWindow; expires: number } | null = null
  let generation = 0
  let busy = false

  return {
    request(owner: number, window: ScreenshotWindow): string | null {
      if (busy || (pending && pending.expires > Date.now())) {
        return null
      }

      const id = randomUUID()
      pending = { id, owner, window, expires: Date.now() + 5000 }

      return id
    },
    clear() {
      generation += 1
      pending = null
    },
    async take(owner: number, id: unknown): Promise<ScreenshotResult> {
      if (!pending || pending.id !== id || pending.owner !== owner) {
        return { ok: false, reason: 'expired' }
      }

      if (pending.expires <= Date.now()) {
        pending = null

        return { ok: false, reason: 'expired' }
      }

      const { window } = pending
      pending = null

      if (!hasScreenPermission()) {
        return { ok: false, reason: 'screen-permission' }
      }

      const capturedGeneration = generation
      busy = true

      try {
        // Retina detail, bounded for very large windows. No display fallback:
        // sharing the desktop would expose content the gesture did not select.
        const scale = Math.min(2, 4096 / Math.max(window.width, window.height))

        const sources = await getSources({
          types: ['window'],
          thumbnailSize: { width: Math.ceil(window.width * scale), height: Math.ceil(window.height * scale) },
          fetchWindowIcons: false
        })

        if (capturedGeneration !== generation) {
          return { ok: false, reason: 'expired' }
        }

        const source = sources.find(
          item => item.id.split(':')[0] === 'window' && item.id.split(':')[1] === String(window.windowId)
        )

        if (!source || source.thumbnail.isEmpty()) {
          return { ok: false, reason: 'unavailable' }
        }

        return { ok: true, png: source.thumbnail.toPNG() }
      } catch {
        return { ok: false, reason: hasScreenPermission() ? 'unavailable' : 'screen-permission' }
      } finally {
        busy = false
      }
    }
  }
}
