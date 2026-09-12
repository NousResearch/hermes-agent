export interface ContrastBitmap {
  /** BGRA bytes, as returned by Electron NativeImage.toBitmap() on Windows. */
  data: Uint8Array
  height: number
  width: number
}

export interface ContrastProbe {
  centerX: number
  footY: number
  fromY: number
  maximumScanWidth?: number
  minimumSpan?: number
  petTopY?: number
  petWidth: number
  toY: number
}

export interface DetectedVisualSurface {
  confidence: number
  left: number
  right: number
  y: number
}

const CONTRAST_THRESHOLD = 16
const CORE_COVERAGE = 0.5
const EXTENSION_COVERAGE = 0.35
const ROW_SAMPLE_OFFSET = 2
const SAMPLE_STEP = 2
const EXTENSION_CHUNK_PX = 16

const clamp = (value: number, min: number, max: number): number => Math.max(min, Math.min(max, value))

const luminance = ({ data, width }: ContrastBitmap, x: number, y: number): number => {
  const offset = (y * width + x) * 4
  const blue = data[offset] ?? 0
  const green = data[offset + 1] ?? 0
  const red = data[offset + 2] ?? 0

  return red * 0.2126 + green * 0.7152 + blue * 0.0722
}

const verticalContrast = (bitmap: ContrastBitmap, x: number, y: number): number =>
  Math.abs(
    (luminance(bitmap, x, y - 2) + luminance(bitmap, x, y - 1)) / 2 -
      (luminance(bitmap, x, y) + luminance(bitmap, x, y + 1)) / 2
  )

/** A one-pixel rule can disappear when averaged into the two regions around an edge. */
const thinLineContrast = (bitmap: ContrastBitmap, x: number, y: number): number =>
  Math.abs(luminance(bitmap, x, y) - (luminance(bitmap, x, y - 1) + luminance(bitmap, x, y + 1)) / 2)

const horizontalFeatureContrast = (bitmap: ContrastBitmap, x: number, y: number): number =>
  Math.max(verticalContrast(bitmap, x, y), thinLineContrast(bitmap, x, y))

function coverage(
  bitmap: ContrastBitmap,
  y: number,
  left: number,
  right: number,
  maskedLeft: number,
  maskedRight: number
): number {
  let eligible = 0
  let strong = 0

  for (let x = left; x < right; x += SAMPLE_STEP) {
    if (x >= maskedLeft && x <= maskedRight) {
      continue
    }

    eligible += 1

    if (horizontalFeatureContrast(bitmap, x, y) >= CONTRAST_THRESHOLD) {
      strong += 1
    }
  }

  return eligible > 0 ? strong / eligible : 0
}

function averageContrast(
  bitmap: ContrastBitmap,
  y: number,
  left: number,
  right: number,
  maskedLeft: number,
  maskedRight: number
): number {
  let eligible = 0
  let total = 0

  for (let x = left; x < right; x += SAMPLE_STEP) {
    if (x >= maskedLeft && x <= maskedRight) {
      continue
    }

    eligible += 1
    total += horizontalFeatureContrast(bitmap, x, y)
  }

  return eligible > 0 ? total / eligible : 0
}

/**
 * Find the nearest strong horizontal visual boundary beneath the pet. A wide,
 * mostly continuous luminance change qualifies; short text strokes and icons
 * do not. Near the current foot line, the pet's own pixels are masked out so
 * the visible parts of a real surface can reconnect across the sprite.
 */
export function detectHorizontalContrastSurface(
  bitmap: ContrastBitmap,
  probe: ContrastProbe
): DetectedVisualSurface | null {
  if (bitmap.width < 3 || bitmap.height < 5 || bitmap.data.length < bitmap.width * bitmap.height * 4) {
    return null
  }

  const centerX = clamp(Math.round(probe.centerX), 0, bitmap.width - 1)
  const petWidth = clamp(Math.round(probe.petWidth), 24, 256)

  const minimumSpan = clamp(Math.round(probe.minimumSpan ?? Math.max(120, petWidth * 2.25)), 24, bitmap.width)

  const maximumScanWidth = clamp(Math.round(probe.maximumScanWidth ?? 420), minimumSpan, bitmap.width)

  const scanWidth =
    probe.maximumScanWidth === undefined
      ? Math.max(minimumSpan, Math.min(maximumScanWidth, Math.round(petWidth * 5.5)))
      : maximumScanWidth

  // Shift the scan window at display edges instead of shortening it. This is
  // especially important when a selected support is re-traced across the full
  // frame while the pet is standing away from the display center.
  const scanLeft = clamp(Math.round(centerX - scanWidth / 2), 0, Math.max(0, bitmap.width - scanWidth))
  const scanRight = Math.min(bitmap.width, scanLeft + scanWidth)
  const coreLeft = clamp(Math.round(centerX - minimumSpan / 2), scanLeft, scanRight - minimumSpan)
  const coreRight = coreLeft + minimumSpan
  const startY = clamp(Math.round(probe.fromY), ROW_SAMPLE_OFFSET, bitmap.height - ROW_SAMPLE_OFFSET - 1)
  const endY = clamp(Math.round(probe.toY), startY, bitmap.height - ROW_SAMPLE_OFFSET - 1)

  for (let y = startY; y <= endY; y += 1) {
    const maskOwnPet = y >= (probe.petTopY ?? Number.NEGATIVE_INFINITY) && y <= probe.footY + 6
    const maskedLeft = maskOwnPet ? Math.round(centerX - petWidth * 0.6) : Number.POSITIVE_INFINITY
    const maskedRight = maskOwnPet ? Math.round(centerX + petWidth * 0.6) : Number.NEGATIVE_INFINITY
    const coreCoverage = coverage(bitmap, y, coreLeft, coreRight, maskedLeft, maskedRight)

    if (coreCoverage < CORE_COVERAGE) {
      continue
    }

    let surfaceY = y
    let surfaceStrength = averageContrast(bitmap, y, coreLeft, coreRight, maskedLeft, maskedRight)

    for (let candidateY = y + 1; candidateY <= Math.min(endY, y + 3); candidateY += 1) {
      const candidateMaskOwnPet =
        candidateY >= (probe.petTopY ?? Number.NEGATIVE_INFINITY) && candidateY <= probe.footY + 6

      const candidateMaskedLeft = candidateMaskOwnPet ? Math.round(centerX - petWidth * 0.6) : Number.POSITIVE_INFINITY

      const candidateMaskedRight = candidateMaskOwnPet ? Math.round(centerX + petWidth * 0.6) : Number.NEGATIVE_INFINITY

      if (
        coverage(bitmap, candidateY, coreLeft, coreRight, candidateMaskedLeft, candidateMaskedRight) < CORE_COVERAGE
      ) {
        continue
      }

      const candidateStrength = averageContrast(
        bitmap,
        candidateY,
        coreLeft,
        coreRight,
        candidateMaskedLeft,
        candidateMaskedRight
      )

      if (candidateStrength > surfaceStrength) {
        surfaceY = candidateY
        surfaceStrength = candidateStrength
      }
    }

    const surfaceMaskOwnPet =
      surfaceY >= (probe.petTopY ?? Number.NEGATIVE_INFINITY) && surfaceY <= probe.footY + 6

    const surfaceMaskedLeft = surfaceMaskOwnPet ? Math.round(centerX - petWidth * 0.6) : Number.POSITIVE_INFINITY

    const surfaceMaskedRight = surfaceMaskOwnPet ? Math.round(centerX + petWidth * 0.6) : Number.NEGATIVE_INFINITY

    const surfaceCoverage = coverage(bitmap, surfaceY, coreLeft, coreRight, surfaceMaskedLeft, surfaceMaskedRight)

    let left = coreLeft
    let right = coreRight

    while (left > scanLeft) {
      const nextLeft = Math.max(scanLeft, left - EXTENSION_CHUNK_PX)

      if (coverage(bitmap, surfaceY, nextLeft, left, surfaceMaskedLeft, surfaceMaskedRight) < EXTENSION_COVERAGE) {
        break
      }

      left = nextLeft
    }

    while (right < scanRight) {
      const nextRight = Math.min(scanRight, right + EXTENSION_CHUNK_PX)

      if (coverage(bitmap, surfaceY, right, nextRight, surfaceMaskedLeft, surfaceMaskedRight) < EXTENSION_COVERAGE) {
        break
      }

      right = nextRight
    }

    // The qualifying core may straddle the actual end of a surface. Trim weak
    // samples from both outer edges so callers receive the visible span, not
    // merely the minimum-width region that proved the row was a surface.
    while (left < right) {
      if (
        (left < surfaceMaskedLeft || left > surfaceMaskedRight) &&
        horizontalFeatureContrast(bitmap, left, surfaceY) >= CONTRAST_THRESHOLD
      ) {
        break
      }

      left += SAMPLE_STEP
    }

    while (right - SAMPLE_STEP > left) {
      const x = right - SAMPLE_STEP

      if (
        (x < surfaceMaskedLeft || x > surfaceMaskedRight) &&
        horizontalFeatureContrast(bitmap, x, surfaceY) >= CONTRAST_THRESHOLD
      ) {
        break
      }

      right -= SAMPLE_STEP
    }

    return { confidence: surfaceCoverage, left, right, y: surfaceY }
  }

  return null
}
