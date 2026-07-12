'use client'

import { useEffect, useRef, useState } from 'react'

import { Zoomable } from '@/components/ui/zoomable'
import { writeDesktopFileText } from '@/lib/desktop-fs'
import { cn } from '@/lib/utils'

import type { RichFenceProps } from './types'

// Draw.io (.drawio / mxGraph XML) has no npm render lib and its SVG export
// needs a real DOM (foreignObject for text). The official draw.io viewer
// (MIT, vendored at assets/drawio/viewer.min.js) is a browser app, so we
// render it inside a SANDBOXED webview — exactly the pattern the right-rail
// preview pane uses (contextIsolation=yes, nodeIntegration=no, sandbox=yes).
//
// Offline note: the vendored viewer hardcodes window.*_PATH globals pointing
// at viewer.diagrams.net (shapes/stencils/proxy). We null them before the
// script runs so the page can't phone home. Diagrams whose shapes are EMBEDDED
// in the XML (the common case for skill/MCP-generated diagrams) render fully
// offline; diagrams referencing external stencils will show a degraded note.
function wrapDrawioHtml(xml: string, viewerUrl: string): string {
  // Embed the XML as a JS string literal; JSON-encode so any quotes/control
  // chars can't break out of the script context.
  const encoded = JSON.stringify(xml)
  return `<!doctype html><html><head><meta charset="utf-8" />
<style>html,body{margin:0;height:100%;background:transparent}</style>
<script>
  // Disable remote asset paths for offline + privacy before the viewer loads.
  window.SHAPES_PATH='';window.STENCIL_PATH='';window.STYLE_PATH='';
  window.PROXY_URL='';window.DRAW_MATH_URL='';window.GRAPH_IMAGE='';
  window.drawioXml=${encoded};
</script>
<script src="${viewerUrl}"></script>
<script>
  function boot(){
    if(typeof GraphViewer==='undefined'){setTimeout(boot,50);return;}
    var div=document.createElement('div');
    div.style.width='100%';div.style.height='100%';
    document.body.appendChild(div);
    try{
      new GraphViewer(div,{xml:window.drawioXml,shadow:false,toolbar:'',
        highlite:true,nav:false,resize:true,fit:true,center:true});
    }catch(e){
      document.body.innerHTML='<p style="font:13px sans-serif;padding:12px">'+
        'Could not render this Draw.io diagram (it may need external stencils unavailable offline).</p>';
    }
  }
  if(document.readyState!=='loading'){boot();}else{document.addEventListener('DOMContentLoaded',boot);}
</script>
</head><body></body></html>`
}

// Small FNV-1a hash → stable cache filename without pulling a dep.
function fnv1a(text: string): number {
  let h = 0x811c9dc5
  for (let i = 0; i < text.length; i += 1) {
    h ^= text.charCodeAt(i)
    h = Math.imul(h, 0x01000193)
  }
  return h >>> 0
}

function SourcePreview({ code, muted }: { code: string; muted?: boolean }) {
  return (
    <pre
      className={cn(
        'overflow-auto p-3 font-mono text-[0.7rem] leading-relaxed whitespace-pre-wrap wrap-anywhere',
        muted ? 'text-muted-foreground/70' : 'text-foreground/90'
      )}
    >
      {code}
    </pre>
  )
}

export default function DrawioRenderer({ code, streaming }: RichFenceProps) {
  const hostRef = useRef<HTMLDivElement | null>(null)
  const [failed, setFailed] = useState(false)
  const [src, setSrc] = useState('')

  useEffect(() => {
    if (streaming) {
      return
    }

    let cancelled = false
    setFailed(false)

    const viewerUrl = `${import.meta.env.BASE_URL}assets/drawio/viewer.min.js`.replace(
      /([^:])\/\//,
      '$1/'
    )

    void (async () => {
      try {
        // Stable per-content filename (FNV-1a) so repeated renders hit cache.
        const hash = fnv1a(code)
        const file = `drawio-cache/drawio-${hash.toString(36)}.html`
        const html = wrapDrawioHtml(code, viewerUrl)
        const result = await writeDesktopFileText(file, html)
        if (!cancelled) {
          setSrc(result.path)
        }
      } catch {
        if (!cancelled) {
          setFailed(true)
          setSrc('')
        }
      }
    })()

    return () => {
      cancelled = true
    }
  }, [code, streaming])

  if (streaming) {
    return <SourcePreview code={code} muted />
  }

  if (failed || !src) {
    return <SourcePreview code={code} />
  }

  return (
    <Zoomable label="Open diagram" className="my-2">
      <div
        ref={hostRef}
        className="h-[33dvh] w-full overflow-hidden rounded-lg border border-border bg-muted/20"
      >
        {/* webview created imperatively so we control webPreferences (sandbox). */}
        <DrawioWebview src={src} onFail={() => setFailed(true)} />
      </div>
    </Zoomable>
  )
}

// Imperative <webview> (mirrors preview-pane.tsx): sandboxed, no node,
// isolated context. The viewer's remote asset paths are already nulled in the
// wrapper HTML, so this surface can only reach the local viewer script.
function DrawioWebview({ src, onFail }: { src: string; onFail: () => void }) {
  const ref = useRef<HTMLDivElement | null>(null)

  useEffect(() => {
    const host = ref.current
    if (!host || typeof document === 'undefined') {
      return
    }

    const webview = document.createElement('webview') as unknown as {
      setAttribute: (k: string, v: string) => void
      addEventListener: (t: string, cb: () => void) => void
      remove: () => void
      className: string
    }
    webview.className = 'h-full w-full'
    webview.setAttribute('partition', 'persist:hermes-drawio')
    webview.setAttribute('src', src)
    webview.setAttribute(
      'webpreferences',
      'contextIsolation=yes,nodeIntegration=no,sandbox=yes'
    )
    webview.addEventListener('did-fail-load', onFail)

    host.replaceChildren(webview as unknown as Node)

    return () => {
      host.replaceChildren()
    }
  }, [src, onFail])

  return <div ref={ref} className="h-full w-full" />
}
