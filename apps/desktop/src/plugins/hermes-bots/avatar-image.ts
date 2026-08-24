/**
 * Image avatars: the device-upload path and the gateway's `image.generate`
 * backend (probe + generation). Kept apart from `avatar.tsx` so the roster's
 * render path doesn't drag the picker's network surface along with it.
 */

import { atom, host } from '@hermes/plugin-sdk'

import { getPluginCtx } from './shared'

// ── image avatars: upload from device + generate via image.generate ─────────

/** Downscale to a small square so plugin storage stays light. */
export function normalizeAvatarImage(dataUrl: string, edge = 256): Promise<string> {
  return new Promise(resolve => {
    const img = new Image()

    img.onload = () => {
      try {
        const canvas = document.createElement('canvas')
        canvas.width = edge
        canvas.height = edge
        const ctx2d = canvas.getContext('2d')!
        const side = Math.min(img.width, img.height)
        ctx2d.drawImage(img, (img.width - side) / 2, (img.height - side) / 2, side, side, 0, 0, edge, edge)
        resolve(canvas.toDataURL('image/png'))
      } catch {
        resolve(dataUrl)
      }
    }

    img.onerror = () => resolve(dataUrl)
    img.src = dataUrl
  })
}

export function pickImageFromDevice(): Promise<null | string> {
  return new Promise(resolve => {
    const input = document.createElement('input')
    input.type = 'file'
    input.accept = 'image/png,image/jpeg,image/webp,image/gif'

    input.onchange = () => {
      const file = input.files?.[0]

      if (!file) {
        return resolve(null)
      }

      if (file.size > 15_000_000) {
        host.notify({
          kind: 'error',
          message: getPluginCtx()?.i18n?.t('avatar.imageTooLarge') ?? 'Image too large (max 15MB).'
        })

        return resolve(null)
      }

      const reader = new FileReader()
      reader.onload = () => resolve(typeof reader.result === 'string' ? reader.result : null)
      reader.onerror = () => resolve(null)
      reader.readAsDataURL(file)
    }

    input.click()
  })
}

/** Cached probe: does the gateway have an image backend? A `false` answer
 *  is re-checked on every dialog open — the gateway may have been restarted
 *  (picking up image.generate) or a backend enabled since the last probe.
 *  Only `true` is sticky. */
export const $imagenAvailable = atom<boolean | null>(null)
let imagenProbeInflight: Promise<void> | null = null

export function probeImagen() {
  if (imagenProbeInflight) {
    return imagenProbeInflight
  }

  imagenProbeInflight = host
    .request<{ available?: boolean }>('image.generate', {
      probe: true
    })
    .then(res => $imagenAvailable.set(Boolean(res?.available)))
    .catch(() => $imagenAvailable.set(false))
    .finally(() => {
      imagenProbeInflight = null
    })

  return imagenProbeInflight
}

/** `image.generate`'s reply. `image_data` is a data URL (works over remote
 *  gateways); `image` is the raw backend URL fallback. */
export interface GeneratedImage {
  error?: string
  image?: string
  image_data?: string
  success?: boolean
}

export async function generateAvatarImage(
  bot: string,
  title?: string,
  description?: string
): Promise<string | undefined> {
  const who = [title || bot, description].filter(Boolean).join(' — ')

  const res = await host.request<GeneratedImage>('image.generate', {
    prompt:
      `Cute minimal robot avatar for an AI agent named "${who}". ` +
      'Friendly simple mascot face, bold flat vector style, solid color background, centered, no text.',
    aspect_ratio: 'square'
  })

  if (!res?.success) {
    throw new Error(res?.error || 'generation failed')
  }

  // image_data (data URL) works over local AND remote gateways; the raw
  // backend URL is the fallback when the gateway couldn't inline it.
  return res.image_data || res.image
}

/** The roster backfill draws the live SVG at 160x160. Pets are 96x104
 *  and uploads are 256. Use that to tell a still face-copy from a real picture. */
export function isBackfilledFacePng(dataUrl: null | string | undefined) {
  if (!dataUrl || typeof dataUrl !== 'string' || !dataUrl.startsWith('data:image/png;base64,')) {
    return false
  }

  try {
    const bin = atob(dataUrl.slice('data:image/png;base64,'.length).slice(0, 48))

    if (bin.length < 24) {
      return false
    }

    const w = (bin.charCodeAt(16) << 24) | (bin.charCodeAt(17) << 16) | (bin.charCodeAt(18) << 8) | bin.charCodeAt(19)
    const h = (bin.charCodeAt(20) << 24) | (bin.charCodeAt(21) << 16) | (bin.charCodeAt(22) << 8) | bin.charCodeAt(23)

    return w === 160 && h === 160
  } catch {
    return false
  }
}
