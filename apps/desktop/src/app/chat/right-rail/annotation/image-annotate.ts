/**
 * Brush-stroke math + compositing for image annotation.
 *
 * The drawing canvas is sized to the *displayed* image content box (the
 * object-contain area inside the <img> element). Strokes are recorded in
 * display pixels; compositing scales them up to the source image's natural
 * resolution so the exported PNG is full-quality.
 */

export interface BrushPoint {
  x: number
  y: number
}

export interface BrushStroke {
  points: BrushPoint[]
}

export const BRUSH_COLOR = '#ef4444'
export const BRUSH_WIDTH_PX = 4

/**
 * Geometry of an image rendered with object-contain inside an element box:
 * the content is scaled uniformly and centered, possibly with letterboxing.
 */
export interface ContainBox {
  height: number
  offsetX: number
  offsetY: number
  scale: number
  width: number
}

export function computeContainBox(
  elementWidth: number,
  elementHeight: number,
  naturalWidth: number,
  naturalHeight: number
): ContainBox {
  if (elementWidth <= 0 || elementHeight <= 0 || naturalWidth <= 0 || naturalHeight <= 0) {
    return { height: 0, offsetX: 0, offsetY: 0, scale: 1, width: 0 }
  }

  const scale = Math.min(elementWidth / naturalWidth, elementHeight / naturalHeight)
  const width = naturalWidth * scale
  const height = naturalHeight * scale

  return {
    height,
    offsetX: (elementWidth - width) / 2,
    offsetY: (elementHeight - height) / 2,
    scale,
    width
  }
}

/** Convert a point in content-box display pixels to source-image pixels. */
export function displayToNatural(point: BrushPoint, box: ContainBox): BrushPoint {
  if (box.scale <= 0) {
    return point
  }

  return { x: point.x / box.scale, y: point.y / box.scale }
}

export function traceStroke(ctx: CanvasRenderingContext2D, stroke: BrushStroke): void {
  if (stroke.points.length === 0) {
    return
  }

  ctx.beginPath()
  ctx.moveTo(stroke.points[0]!.x, stroke.points[0]!.y)

  if (stroke.points.length === 1) {
    // A tap — draw a dot by tracing a zero-length line.
    ctx.lineTo(stroke.points[0]!.x + 0.01, stroke.points[0]!.y + 0.01)
  } else {
    for (let i = 1; i < stroke.points.length; i++) {
      ctx.lineTo(stroke.points[i]!.x, stroke.points[i]!.y)
    }
  }

  ctx.stroke()
}

/**
 * Composite the source image + all strokes onto a new canvas at natural
 * resolution and return a PNG data URL. Returns null when the browser canvas
 * pipeline is unavailable (e.g. jsdom).
 */
export async function compositeAnnotatedImage(
  imageDataUrl: string,
  strokes: BrushStroke[],
  displayBox: ContainBox
): Promise<string | null> {
  if (strokes.length === 0 || displayBox.scale <= 0) {
    return null
  }

  const image = await loadImage(imageDataUrl)
  if (!image) {
    return null
  }

  const canvas = document.createElement('canvas')
  canvas.width = image.naturalWidth
  canvas.height = image.naturalHeight

  const ctx = canvas.getContext('2d')
  if (!ctx) {
    return null
  }

  ctx.drawImage(image, 0, 0, image.naturalWidth, image.naturalHeight)

  ctx.strokeStyle = BRUSH_COLOR
  ctx.lineWidth = BRUSH_WIDTH_PX / displayBox.scale
  ctx.lineCap = 'round'
  ctx.lineJoin = 'round'

  for (const stroke of strokes) {
    traceStroke(ctx, {
      points: stroke.points.map(point => displayToNatural(point, displayBox))
    })
  }

  try {
    return canvas.toDataURL('image/png')
  } catch {
    return null
  }
}

function loadImage(src: string): Promise<HTMLImageElement | null> {
  return new Promise(resolve => {
    const image = new Image()
    image.onload = () => resolve(image)
    image.onerror = () => resolve(null)
    image.src = src
  })
}

/** Decode a PNG/JPEG data URL into bytes for the saveImageBuffer IPC. */
export async function dataUrlToBytes(dataUrl: string): Promise<Uint8Array | null> {
  try {
    const response = await fetch(dataUrl)
    const buffer = await response.arrayBuffer()
    return new Uint8Array(buffer)
  } catch {
    return null
  }
}
