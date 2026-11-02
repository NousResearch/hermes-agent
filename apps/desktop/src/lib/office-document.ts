import DOMPurify from 'dompurify'
import mammoth from 'mammoth'

const PREVIEW_CSP = "default-src 'none'; base-uri 'none'; form-action 'none'; img-src data:; style-src 'unsafe-inline'"

const PAGE_CSS = `
html,body{margin:0;background:#e7e6e6;color:#222}
body{padding:28px 16px 48px}
.page{box-sizing:border-box;max-width:816px;min-height:1056px;margin:0 auto;padding:72px 72px 96px;background:#fff;box-shadow:0 1px 3px rgba(0,0,0,.22);font:16px/1.45 Georgia,"Times New Roman",serif}
h1,h2,h3{font-family:Calibri,"Segoe UI",sans-serif;line-height:1.2;font-weight:700}
h1{font-size:28px;margin:0 0 14px}
h1.doc-title{font-size:36px;font-weight:400}
h2{font-size:20px;margin:18px 0 8px}
h3{font-size:16px;margin:14px 0 6px}
p{margin:0 0 10px}
p.doc-subtitle{color:#595959;font-size:14px}
ul,ol{margin:0 0 10px;padding-left:1.4em}
blockquote{margin:0 0 12px;padding-left:12px;border-left:3px solid #c8c8c8;color:#333}
table{border-collapse:collapse;margin:0 0 12px}
td,th{border:1px solid #bfbfbf;padding:4px 8px;vertical-align:top}
img{max-width:100%;height:auto}
`

const STYLE_MAP = [
  "p[style-name='Title'] => h1.doc-title:fresh",
  "p[style-name='Subtitle'] => p.doc-subtitle:fresh",
  "p[style-name='Heading 1'] => h1:fresh",
  "p[style-name='Heading 2'] => h2:fresh",
  "p[style-name='Heading 3'] => h3:fresh",
  "p[style-name='Quote'] => blockquote:fresh",
  "p[style-name='Intense Quote'] => blockquote:fresh"
]

function arrayBufferOf(bytes: Uint8Array): ArrayBuffer {
  const copy = new Uint8Array(bytes)

  return copy.buffer
}

/** Semantic HTML, not a paginated Word layout. External links and active
 * content are stripped; the iframe that hosts this document is also sandboxed. */
export async function docxPreviewDocument(bytes: Uint8Array): Promise<string> {
  let html = ''

  try {
    const arrayBuffer = arrayBufferOf(bytes)
    // Node mammoth reads `buffer`; the browser build reads `arrayBuffer`.
    const input: { arrayBuffer: ArrayBuffer; buffer?: Buffer } = { arrayBuffer }

    if (typeof Buffer !== 'undefined') {
      input.buffer = Buffer.from(arrayBuffer)
    }

    const result = await mammoth.convertToHtml({
      ...input,
      styleMap: STYLE_MAP,
      convertImage: mammoth.images.imgElement(image =>
        image.read('base64').then(data => ({ src: `data:${image.contentType};base64,${data}` }))
      )
    } as never)

    html = result.value
  } catch {
    throw new Error('Could not read this Word document')
  }

  return officePageDocument(html)
}

export function officePageDocument(html: string): string {
  const clean = DOMPurify.sanitize(html, {
    FORBID_TAGS: ['script', 'template', 'iframe', 'frame', 'object', 'embed', 'link', 'meta', 'base'],
    FORBID_ATTR: ['href', 'srcset', 'action', 'formaction', 'target', 'ping']
  })

  if (!clean.trim()) {
    throw new Error('This document has no previewable text')
  }

  return `<!doctype html><html><head><meta http-equiv="Content-Security-Policy" content="${PREVIEW_CSP}"><style>${PAGE_CSS}</style></head><body><article class="page">${clean}</article></body></html>`
}
