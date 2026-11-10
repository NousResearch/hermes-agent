// Rasterise an SVG string to PNG and copy it to the clipboard. Self-contained
// SVGs only (inline styles): Pele output with its palette written onto the
// root qualifies. Falls back to copying the SVG markup as text where image
// clipboard writes aren't permitted.

import { writeClipboardText } from '@/lib/clipboard'

// A percentage width or height is not an intrinsic size (parseFloat("100%")
// would make a 100px PNG); fall back to the viewBox for those.
function isPercentLength(raw: string | null): boolean {
  return Boolean(raw?.trim().endsWith('%'))
}

function viewBoxSize(el: Element): { height: number; width: number } | null {
  const [, , vbW, vbH] = (el.getAttribute('viewBox') || '').split(/[\s,]+/).map(Number)

  return vbW > 0 && vbH > 0 ? { height: vbH, width: vbW } : null
}

function parseSvgLength(raw: string | null): number | null {
  if (!raw || isPercentLength(raw)) {
    return null
  }

  const value = Number.parseFloat(raw)

  return Number.isFinite(value) ? value : null
}

export function svgSize(svg: string): { height: number; width: number } {
  const el = new DOMParser().parseFromString(svg, 'image/svg+xml').documentElement
  const width = parseSvgLength(el.getAttribute('width'))
  const height = parseSvgLength(el.getAttribute('height'))

  if (width && height) {
    return { height, width }
  }

  return viewBoxSize(el) ?? { height: 600, width: 800 }
}

export function svgToPngBlob(svg: string, scale = 2): Promise<Blob> {
  const { height, width } = svgSize(svg)

  return new Promise((resolve, reject) => {
    const image = new Image()

    image.onload = () => {
      const canvas = document.createElement('canvas')
      canvas.width = Math.max(1, Math.round(width * scale))
      canvas.height = Math.max(1, Math.round(height * scale))

      const ctx = canvas.getContext('2d')

      if (!ctx) {
        reject(new Error('no 2d context'))

        return
      }

      ctx.scale(scale, scale)
      ctx.drawImage(image, 0, 0, width, height)
      canvas.toBlob(blob => (blob ? resolve(blob) : reject(new Error('toBlob failed'))), 'image/png')
    }

    image.onerror = () => reject(new Error('svg load failed'))
    image.src = `data:image/svg+xml;charset=utf-8,${encodeURIComponent(svg)}`
  })
}

export async function copySvgAsPng(svg: string): Promise<void> {
  // An insecure origin has no image clipboard: copy the markup while the click still counts.
  if (!navigator.clipboard?.write || typeof ClipboardItem === 'undefined') {
    return writeClipboardText(svg)
  }

  try {
    const blob = await svgToPngBlob(svg)

    await navigator.clipboard.write([new ClipboardItem({ 'image/png': blob })])
  } catch {
    await writeClipboardText(svg)
  }
}
