import { mkdir, writeFile } from 'node:fs/promises'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'

const here = dirname(fileURLToPath(import.meta.url))
const outDir = join(here, '..', 'public', 'drawio')
const VIEWER_URL = 'https://viewer.diagrams.net/js/viewer.min.js'

await mkdir(outDir, { recursive: true })

const res = await fetch(VIEWER_URL)
const viewerJs = res.ok ? await res.text() : ''

// The Receiver: This page loads the viewer and listens for XML via postMessage.
// GraphViewer expects the xml option as a raw XML string, not a parsed DOM node.
// Two modes controlled by ev.data.mode:
//   "inline"   — static diagram, no toolbar/nav/pan/zoom (for in-chat display)
//   "expanded" — toolbar, minimap, drag-pan, wheel-zoom (for the full-view dialog)
const html = [
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
  '        var isExpanded = mode === \'expanded\';',
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
  '          div.addEventListener(\'pointerdown\', function(pev) {',
  '            if (pev.button !== 0) return;',
  '            pev.currentTarget.setPointerCapture(pev.pointerId);',
  '            panStart = { x: pev.clientX, y: pev.clientY };',
  '            panTranslate = { x: viewer.graph.view.translate.x, y: viewer.graph.view.translate.y };',
  '          });',
  '          div.addEventListener(\'pointermove\', function(pev) {',
  '            if (!panStart) return;',
  '            var s = viewer.graph.view.scale;',
  '            viewer.graph.view.setTranslate(',
  '              panTranslate.x + (pev.clientX - panStart.x) / s,',
  '              panTranslate.y + (pev.clientY - panStart.y) / s',
  '            );',
  '          });',
  '          div.addEventListener(\'pointerup\', function() { panStart = null; });',
  '          div.addEventListener(\'pointercancel\', function() { panStart = null; });',
  '          // Smooth proportional wheel zoom toward the cursor position.',
  '          var MIN_ZOOM = 0.1;',
  '          var MAX_ZOOM = 5;',
  '          div.addEventListener(\'wheel\', function(wev) {',
  '            wev.preventDefault();',
  '            if (viewer && viewer.graph) {',
  '              var factor = 1 - wev.deltaY * 0.001;',
  '              var newScale = Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, viewer.graph.view.scale * factor));',
  '              var pt = mxUtils.convertPoint(div, wev.clientX, wev.clientY);',
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

await writeFile(join(outDir, 'render.html'), html)
if (viewerJs) await writeFile(join(outDir, 'viewer.min.js'), viewerJs)
console.log('Wrote public/drawio/render.html (Inlined Viewer).')
