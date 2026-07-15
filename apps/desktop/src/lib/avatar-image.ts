// Downscale a picked image to a small square data URL for use as a profile
// avatar. Cover-crops to a centered square then re-encodes at `size`×`size`, so
// the file we upload is tiny (~10-50 KB) regardless of the source resolution.
// WebP keeps it small; we fall back to PNG if the platform can't encode WebP.

const AVATAR_SIZE = 256
const AVATAR_QUALITY = 0.85

export async function downscaleAvatar(dataUrl: string, size = AVATAR_SIZE): Promise<string> {
  if (!dataUrl.startsWith('data:image/')) {
    throw new Error('Selected file is not an image.')
  }

  const image = await loadImage(dataUrl)

  const canvas = document.createElement('canvas')
  canvas.width = size
  canvas.height = size

  const ctx = canvas.getContext('2d')

  if (!ctx) {
    throw new Error('Could not process image.')
  }

  // Center cover-crop: scale the shorter side to fill the square.
  const side = Math.min(image.naturalWidth, image.naturalHeight)
  const sx = (image.naturalWidth - side) / 2
  const sy = (image.naturalHeight - side) / 2
  ctx.drawImage(image, sx, sy, side, side, 0, 0, size, size)

  const webp = canvas.toDataURL('image/webp', AVATAR_QUALITY)

  // toDataURL returns a PNG when the requested type is unsupported; only trust
  // the WebP result when the platform actually produced one.
  if (webp.startsWith('data:image/webp')) {
    return webp
  }

  return canvas.toDataURL('image/png')
}

function loadImage(src: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const image = new Image()
    image.onload = () => resolve(image)
    image.onerror = () => reject(new Error('Could not load the selected image.'))
    image.src = src
  })
}
