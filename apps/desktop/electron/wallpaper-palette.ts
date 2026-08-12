import type { NativeImage } from 'electron'

const SAMPLE_EDGE = 48
const CHANNEL_BUCKET_SHIFT = 4
const MIN_ACCENT_SATURATION = 0.12

export interface WallpaperPalette {
  accent: string
  dominant: string
}

interface ColorBucket {
  blue: number
  count: number
  green: number
  red: number
}

function normalizedChannel(value: number): number {
  return Math.min(255, Math.max(0, Math.round(value)))
}

function rgbToHex([red, green, blue]: [number, number, number]): string {
  return `#${[red, green, blue].map(channel => normalizedChannel(channel).toString(16).padStart(2, '0')).join('')}`
}

function bucketColor(bucket: ColorBucket): [number, number, number] {
  return [bucket.red / bucket.count, bucket.green / bucket.count, bucket.blue / bucket.count]
}

function saturationAndLightness([red, green, blue]: [number, number, number]): {
  lightness: number
  saturation: number
} {
  const max = Math.max(red, green, blue) / 255
  const min = Math.min(red, green, blue) / 255
  const chroma = max - min
  const lightness = (max + min) / 2
  const saturation = chroma === 0 ? 0 : chroma / (1 - Math.abs(2 * lightness - 1))

  return {
    lightness,
    saturation: Number.isFinite(saturation) ? saturation : 0
  }
}

function accentScore(bucket: ColorBucket): number {
  const { lightness, saturation } = saturationAndLightness(bucketColor(bucket))
  const population = Math.sqrt(bucket.count)
  const usefulLightness = 0.52 + 0.48 * (1 - Math.abs(lightness - 0.5) * 2)

  return population * (0.16 + saturation ** 1.7 * 3.2) * usefulLightness
}

/** Quantize Electron's BGRA bitmap into one surface tint and one accent. */
export function extractWallpaperPaletteFromBgra(pixels: ArrayLike<number>): WallpaperPalette | null {
  const buckets = new Map<number, ColorBucket>()

  for (let index = 0; index + 3 < pixels.length; index += 4) {
    const alpha = Number(pixels[index + 3])

    if (!Number.isFinite(alpha) || alpha < 128) {
      continue
    }

    const blue = normalizedChannel(Number(pixels[index]))
    const green = normalizedChannel(Number(pixels[index + 1]))
    const red = normalizedChannel(Number(pixels[index + 2]))

    const key =
      ((red >> CHANNEL_BUCKET_SHIFT) << 8) | ((green >> CHANNEL_BUCKET_SHIFT) << 4) | (blue >> CHANNEL_BUCKET_SHIFT)

    const bucket = buckets.get(key)

    if (bucket) {
      bucket.red += red
      bucket.green += green
      bucket.blue += blue
      bucket.count += 1
    } else {
      buckets.set(key, { blue, count: 1, green, red })
    }
  }

  if (buckets.size === 0) {
    return null
  }

  const ranked = [...buckets.values()].sort((left, right) => right.count - left.count)
  const dominantBucket = ranked[0]
  let accentBucket = dominantBucket
  let bestScore = -1

  for (const bucket of ranked) {
    const { saturation } = saturationAndLightness(bucketColor(bucket))

    if (saturation < MIN_ACCENT_SATURATION) {
      continue
    }

    const score = accentScore(bucket)

    if (score > bestScore) {
      bestScore = score
      accentBucket = bucket
    }
  }

  return {
    accent: rgbToHex(bucketColor(accentBucket)),
    dominant: rgbToHex(bucketColor(dominantBucket))
  }
}

/** Sample an app-owned native image in the main process; no renderer pixels. */
export function extractWallpaperPalette(image: NativeImage): WallpaperPalette | null {
  if (image.isEmpty()) {
    return null
  }

  const sample = image.resize({ height: SAMPLE_EDGE, quality: 'good', width: SAMPLE_EDGE })
  const { height, width } = sample.getSize()
  const bitmap = sample.toBitmap({ scaleFactor: 1 })

  if (width <= 0 || height <= 0 || bitmap.length < width * height * 4) {
    return null
  }

  return extractWallpaperPaletteFromBgra(bitmap.subarray(0, width * height * 4))
}
