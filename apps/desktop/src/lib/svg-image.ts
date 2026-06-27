// Rasterise an SVG string to PNG and copy it to the clipboard. Self-contained
// SVGs only (inline styles) — mermaid output qualifies. Falls back to copying
// the SVG markup as text where image clipboard writes aren't permitted.

function svgSize(svg: string): { height: number; width: number } {
  const el = new DOMParser().parseFromString(svg, 'image/svg+xml').documentElement
  const width = parseFloat(el.getAttribute('width') || '')
  const height = parseFloat(el.getAttribute('height') || '')

  if (width && height) {
    return { height, width }
  }

  const [, , vbW, vbH] = (el.getAttribute('viewBox') || '').split(/[\s,]+/).map(Number)

  return vbW && vbH ? { height: vbH, width: vbW } : { height: 600, width: 800 }
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
  try {
    const blob = await svgToPngBlob(svg)

    await navigator.clipboard.write([new ClipboardItem({ 'image/png': blob })])
  } catch {
    await navigator.clipboard.writeText(svg)
  }
}
