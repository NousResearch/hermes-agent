// Rasterise an SVG string to PNG and copy it to the clipboard. Self-contained
// SVGs only (inline styles) — mermaid output qualifies. Falls back to copying
// the SVG markup as text where image clipboard writes aren't permitted.

// Mermaid emits width="100%" plus a viewBox. That percentage is not an
// intrinsic size: the zoom overlay's shrink-to-fit grid can collapse it, and
// parseFloat("100%") makes a 100px PNG. Replace percentage attrs with the
// viewBox pixels; leave explicit pixel attrs alone. No-op when nothing to do.
function isPercentLength(raw: string | null): boolean {
  return Boolean(raw?.trim().endsWith('%'))
}

function viewBoxSize(el: Element): { height: number; width: number } | null {
  const [, , vbW, vbH] = (el.getAttribute('viewBox') || '').split(/[\s,]+/).map(Number)

  return vbW > 0 && vbH > 0 ? { height: vbH, width: vbW } : null
}

// `mermaid.render()` returns the HTML serialisation of the diagram, aimed at
// inline HTML. Its label islands (XHTML inside <foreignObject>) therefore carry
// HTML-only syntax: a `<br/>` in the source comes back as an open `<br>`, and a
// non-breaking space (`&nbsp;`, mermaid's `#nbsp;`, or a literal U+00A0) comes
// back as the `&nbsp;` entity, which XML does not define. Everything below
// parses that string as strict XML (`image/svg+xml`) — and so does Blink when
// the same string reaches an <img> as a data: URI — so either one turns the
// document into a parsererror page and the diagram silently stops rendering.
// Parse it with the HTML parser it was serialised for and re-serialise as XML,
// which fixes every such construct at once rather than one tag at a time.
export function xmlWellFormedSvg(svg: string): string {
  const el = new DOMParser().parseFromString(svg, 'text/html').querySelector('svg')

  if (!el) {
    return svg
  }

  // The HTML parser keeps a label's `xmlns="…/xhtml"` as a plain attribute. The
  // serializer declares that namespace itself, and a spec-conformant one (jsdom)
  // also writes the plain attribute — a duplicate `xmlns`, malformed again.
  for (const node of el.querySelectorAll('[xmlns]')) {
    node.removeAttribute('xmlns')
  }

  return new XMLSerializer().serializeToString(el)
}

export function normalizeSvgSize(svg: string): string {
  const el = new DOMParser().parseFromString(svg, 'image/svg+xml').documentElement

  if (el.tagName !== 'svg') {
    return svg
  }

  const width = el.getAttribute('width')
  const height = el.getAttribute('height')
  const widthPct = isPercentLength(width)
  const heightPct = isPercentLength(height)

  if (!widthPct && !heightPct) {
    return svg
  }

  // Mermaid pins the intrinsic width with an inline max-width in pixels. An
  // inline style beats the container's width caps (max-w-full and friends),
  // so a wide diagram renders at full intrinsic width in the inline preview
  // instead of shrinking to the pane. Relative values stay untouched.
  const maxWidth = el.style.getPropertyValue('max-width')

  if (maxWidth && !isPercentLength(maxWidth) && !maxWidth.trim().endsWith('vw')) {
    el.style.setProperty('max-width', '100%')
  }

  const box = viewBoxSize(el)

  if (!box) {
    return svg
  }

  if (widthPct) {
    el.setAttribute('width', String(box.width))
  }

  // Mermaid usually omits height. Fill it from the viewBox so the overlay has
  // both intrinsic dimensions; don't overwrite an explicit pixel height.
  if (heightPct || (widthPct && !height)) {
    el.setAttribute('height', String(box.height))
  }

  return new XMLSerializer().serializeToString(el)
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
  try {
    const blob = await svgToPngBlob(svg)

    await navigator.clipboard.write([new ClipboardItem({ 'image/png': blob })])
  } catch {
    await navigator.clipboard.writeText(svg)
  }
}
