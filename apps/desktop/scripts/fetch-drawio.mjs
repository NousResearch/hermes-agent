import { createHash } from 'node:crypto'
import { mkdir, open, readFile, realpath, rename, unlink, writeFile } from 'node:fs/promises'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'

const here = dirname(fileURLToPath(import.meta.url))
const outDir = join(here, '..', 'public', 'drawio')
const viewerPath = join(outDir, 'viewer.min.js')
const tmpViewerPath = join(outDir, 'viewer.min.js.tmp')
const renderPath = join(outDir, 'render.html')

const VIEWER_URL = 'https://viewer.diagrams.net/js/viewer.min.js'
export const EXPECTED_SHA384_BASE64 = 'QzMvVE38vWFeffkNfMOlbAjgYomlGSYS6cNTK/k42HyuUw7Qyy132NIlGvPzI1h/'
export const REQUIRED_SYMBOLS = ['GraphViewer', 'mxUtils']

export function sha384Base64(data) {
  return createHash('sha384').update(data).digest('base64')
}

export function missingSymbols(viewerJs) {
  return REQUIRED_SYMBOLS.filter((sym) => !viewerJs.includes(sym))
}

export async function checkExisting(opts = {}) {
  const targetViewerPath = opts.viewerPath ?? viewerPath

  let viewerJs
  try {
    viewerJs = await readFile(targetViewerPath, 'utf8')
  } catch (err) {
    if (err.code === 'ENOENT') {
      return { ok: false, error: `Missing ${targetViewerPath}` }
    }
    return { ok: false, error: `Failed to read ${targetViewerPath}: ${err.message}` }
  }

  const hash = sha384Base64(viewerJs)
  if (hash !== EXPECTED_SHA384_BASE64) {
    return {
      ok: false,
      error: [
        `Integrity check failed for ${targetViewerPath}`,
        `  Expected SHA-384: ${EXPECTED_SHA384_BASE64}`,
        `  Actual SHA-384:   ${hash}`,
      ].join('\n'),
    }
  }

  const missing = missingSymbols(viewerJs)
  if (missing.length) {
    return {
      ok: false,
      error: `Symbol sanity check failed for ${targetViewerPath}: missing ${missing.join(', ')}`,
    }
  }

  return { ok: true, hash }
}

export async function fetchAndWrite(opts = {}) {
  const targetOutDir = opts.outDir ?? outDir
  const targetViewerPath = opts.viewerPath ?? viewerPath
  const targetTmpViewerPath = opts.tmpViewerPath ?? tmpViewerPath
  const targetRenderPath = opts.renderPath ?? renderPath
  const viewerUrl = opts.viewerUrl ?? VIEWER_URL

  await mkdir(targetOutDir, { recursive: true })

  console.log(`Fetching ${viewerUrl} ...`)
  const res = await fetch(viewerUrl)
  if (!res.ok) {
    return {
      ok: false,
      error: `Failed to fetch ${viewerUrl}: ${res.status} ${res.statusText}`,
    }
  }

  // Stream the response to a temporary file while computing the SHA-384 digest.
  const hash = createHash('sha384')
  const handle = await open(targetTmpViewerPath, 'w')
  try {
    const reader = res.body.getReader()
    while (true) {
      const { done, value } = await reader.read()
      if (done) break
      hash.update(value)
      await handle.write(value)
    }
  } finally {
    await handle.close()
  }

  const actualHash = hash.digest('base64')
  if (actualHash !== EXPECTED_SHA384_BASE64) {
    await unlink(targetTmpViewerPath).catch(() => {})
    return {
      ok: false,
      error: [
        `Integrity check failed for ${viewerUrl}`,
        `  Expected SHA-384: ${EXPECTED_SHA384_BASE64}`,
        `  Actual SHA-384:   ${actualHash}`,
      ].join('\n'),
    }
  }

  // Basic sanity check: the viewer must expose the symbols our render.html relies on.
  const viewerJs = await readFile(targetTmpViewerPath, 'utf8')
  const missing = missingSymbols(viewerJs)
  if (missing.length) {
    await unlink(targetTmpViewerPath).catch(() => {})
    return {
      ok: false,
      error: `Symbol sanity check failed: missing ${missing.join(', ')}`,
    }
  }

  // All checks passed: commit the validated files.
  await rename(targetTmpViewerPath, targetViewerPath)
  const html = buildHtml(viewerJs)
  await writeFile(targetRenderPath, html)

  return { ok: true, hash: actualHash }
}

// The Receiver: This page loads the viewer and listens for XML via postMessage.
// GraphViewer expects the xml option as a raw XML string, not a parsed DOM node.
// Two modes controlled by ev.data.mode:
//   "inline"   — static diagram, no toolbar/nav/pan/zoom (for in-chat display)
//   "expanded" — toolbar, minimap, drag-pan, wheel-zoom (for the full-view dialog)
export function buildHtml(viewerJs) {
  return [
    '<!doctype html><html><head><meta charset="utf-8" />',
    '<style>html,body{margin:0;height:100%;background:#fff;overflow:hidden}</style>',
    '<script>',
    "  window.SHAPES_PATH='';window.STENCIL_PATH='';window.STYLE_PATH='';",
    "  window.PROXY_URL='';window.DRAW_MATH_URL='';window.GRAPH_IMAGE='';",
    '</script>',
    '<script>' + viewerJs + '</script>',
    '<script>',
    "  window.addEventListener('message', function(ev){",
    "    if (ev.data && ev.data.action === 'load') {",
    "      var xml = ev.data.xml;",
    "      var mode = ev.data.mode || 'inline';",
    "      var div = document.createElement('div');",
    "      div.style.width='100%';div.style.height='100%';",
    '      document.body.appendChild(div);',
    '      try {',
    '        var xmlDoc = mxUtils.parseXml(xml);',
    "        var isExpanded = mode === 'expanded';",
    '        var viewer = new GraphViewer(div, xmlDoc.documentElement, {',
    "          shadow: false,",
    "          toolbar: isExpanded ? 'zoom zoomin zoomout fit' : '',",
    '          highlight: true,',
    "          nav: isExpanded,",
    '          resize: true, fit: true, center: true',
    '        });',
    '        if (isExpanded && viewer.graph) {',
    '          // Custom pointer-based pan (avoids mxGraph viewport constraints).',
    '          var panStart = null, panTranslate = null;',
    "          div.addEventListener('pointerdown', function(pev) {",
    '            if (pev.button !== 0) return;',
    '            pev.currentTarget.setPointerCapture(pev.pointerId);',
    '            panStart = { x: pev.clientX, y: pev.clientY };',
    '            panTranslate = { x: viewer.graph.view.translate.x, y: viewer.graph.view.translate.y };',
    '          });',
    "          div.addEventListener('pointermove', function(pev) {",
    '            if (!panStart) return;',
    '            var s = viewer.graph.view.scale;',
    '            viewer.graph.view.setTranslate(',
    '              panTranslate.x + (pev.clientX - panStart.x) / s,',
    '              panTranslate.y + (pev.clientY - panStart.y) / s',
    '            );',
    '          });',
    "          div.addEventListener('pointerup', function() { panStart = null; });",
    "          div.addEventListener('pointercancel', function() { panStart = null; });",
    '          // Smooth proportional wheel zoom toward the cursor position.',
    '          var MIN_ZOOM = 0.1;',
    '          var MAX_ZOOM = 5;',
    "          div.addEventListener('wheel', function(wev) {",
    '            wev.preventDefault();',
    '            if (viewer && viewer.graph) {',
    '              var factor = 1 - wev.deltaY * 0.001;',
    '              var newScale = Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, viewer.graph.view.scale * factor));',
    "              var pt = mxUtils.convertPoint(div, wev.clientX, wev.clientY);",
    '              viewer.graph.zoomTo(newScale, pt.x, pt.y);',
    '            }',
    '          }, { passive: false });',
    '        }',
    '      } catch(e) {',
    "        document.body.innerHTML = '<p style=\"font:13px sans-serif;padding:12px;color:#b00\">Render Error: ' + e + '</p>';",
    '      }',
    '    }',
    '  });',
    '</script>',
    '</head><body></body></html>',
  ].join('\n')
}

async function main() {
  if (process.argv.slice(2).includes('--check')) {
    const result = await checkExisting()
    if (!result.ok) {
      console.error(result.error)
      process.exit(1)
    }
    console.log(`Integrity check passed for ${viewerPath}`)
    return
  }

  const result = await fetchAndWrite()
  if (!result.ok) {
    console.error(result.error)
    process.exit(1)
  }
  console.log(`Wrote ${renderPath} (Inlined Viewer).`)
  console.log(`Wrote ${viewerPath} (SHA-384: ${result.hash}).`)
}

let isMain = false
if (process.argv[1]) {
  try {
    isMain =
      await realpath(fileURLToPath(import.meta.url)) === await realpath(process.argv[1])
  } catch {
    isMain = false
  }
}

if (isMain) {
  main().catch((err) => {
    console.error(err)
    process.exit(1)
  })
}
