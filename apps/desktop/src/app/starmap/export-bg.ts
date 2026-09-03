// Colors shared by the export capture paths.
//
// The star-map <canvas> is intentionally TRANSPARENT — the royal-blue backdrop
// the user sees is the page <body>/app chrome behind it, NOT pixels on the
// canvas. A bare `canvas.captureStream()` / `getImageData` therefore returns
// transparency, which muxes as pitch-black in a GIF/video. To export the graph
// the way it actually looks, we must first paint the real page background onto
// the capture surface, then composite the live (transparent) canvas on top.

/**
 * Resolve the page's backdrop as a single css <color> usable as a canvas
 * fillStyle. Reads the <body> background, normalizing modern `color(srgb …)`
 * and `color-mix(…)` tokens (which canvas fillStyle cannot parse) down to an
 * `rgb(r,g,b)` / `#rrggbb` string. Falls back to a readable background-color.
 */
export function pageBackgroundColor(): string {
  const probe = () => {
    try {
      const body = getComputedStyle(document.body)
      const v = body.backgroundColor

      if (v && v !== 'transparent' && v !== 'rgba(0, 0, 0, 0)') {return v}
    } catch {
      /* ignore */
    }

    // Some themes paint bg on the :root / a wrapper instead of body.
    for (const sel of ['html', '[data-theme]', '#root', '.app']) {
      try {
        const el = document.querySelector(sel)

        if (!el) {continue}
        const v = getComputedStyle(el).backgroundColor

        if (v && v !== 'transparent' && v !== 'rgba(0, 0, 0, 0)') {return v}
      } catch {
        /* ignore */
      }
    }

    return '#000000'
  }

  const raw = probe()

  // Normalize color(srgb r g b [ / a]) → rgb(r,g,b)
  const srgb = /color\(\s*srgb\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)[^)]*\)/i.exec(raw)

  if (srgb) {
    const r = Math.round(parseFloat(srgb[1]) * 255)
    const g = Math.round(parseFloat(srgb[2]) * 255)
    const b = Math.round(parseFloat(srgb[3]) * 255)

    return `rgb(${r}, ${g}, ${b})`
  }

  // color-mix(in srgb, A p%, B) — take A's weight (the dominant stop).
  const mix = /color-mix\(\s*in\s+srgb,\s*([^,]+?)\s+([\d.]+)%[^,]*,\s*([^)]+)\)/i.exec(raw)

  if (mix) {
    const t = Math.min(1, Math.max(0, parseFloat(mix[2]) / 100))
    const a = parseColor(mix[1].trim())
    const b = parseColor(mix[3].trim())

    if (a && b) {
      return `rgb(${Math.round(a[0] + (b[0] - a[0]) * (1 - t))}, ${Math.round(
        a[1] + (b[1] - a[1]) * (1 - t)
      )}, ${Math.round(a[2] + (b[2] - a[2]) * (1 - t))})`
    }
  }

  return raw
}

function parseColor(v: string): [number, number, number] | null {
  const hex = /^#([0-9a-f]{3}|[0-9a-f]{6})$/i.exec(v.trim())

  if (hex) {
    let h = hex[1]

    if (h.length === 3) {h = h.split('').map((c) => c + c).join('')}

    return [parseInt(h.slice(0, 2), 16), parseInt(h.slice(2, 4), 16), parseInt(h.slice(4, 6), 16)]
  }

  const srgb = /color\(\s*srgb\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)/i.exec(v)

  if (srgb) {return [Math.round(parseFloat(srgb[1]) * 255), Math.round(parseFloat(srgb[2]) * 255), Math.round(parseFloat(srgb[3]) * 255)]}
  const rgb = /rgba?\(\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)/i.exec(v)

  if (rgb) {return [Math.round(parseFloat(rgb[1])), Math.round(parseFloat(rgb[2])), Math.round(parseFloat(rgb[3]))]}

  return null
}
