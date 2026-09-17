// Measures the desktop image-attach pipeline stage by stage on a real image,
// against the real renderer helpers. No Electron, no LLM — just the transforms
// an attached image goes through between the paperclip and prompt.submit.
//
//   node scripts/perf/image-attach-bench.mjs [--kb 900] [--rounds 7]

import { readFileSync, writeFileSync, mkdtempSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { performance } from 'node:perf_hooks'

const args = process.argv.slice(2)
const flag = (name, fallback) => {
  const i = args.indexOf(`--${name}`)

  return i >= 0 ? Number(args[i + 1]) : fallback
}

const ROUNDS = flag('rounds', 7)
const SIZES_KB = args.includes('--kb') ? [flag('kb', 900)] : [120, 900, 3200]

const dir = mkdtempSync(join(tmpdir(), 'hermes-img-bench-'))

/** A PNG-shaped byte blob of a given size. The pipeline treats it as opaque
 *  bytes everywhere we measure, so the pixels don't matter — the length does. */
function makeImage(kb) {
  const bytes = Buffer.alloc(kb * 1024)
  Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]).copy(bytes)

  for (let i = 8; i < bytes.length; i += 1) {
    bytes[i] = (i * 2654435761) & 0xff
  }

  const path = join(dir, `img-${kb}kb.png`)
  writeFileSync(path, bytes)

  return path
}

const stat = samples => {
  const s = [...samples].sort((a, b) => a - b)

  return {
    mean: s.reduce((a, b) => a + b, 0) / s.length,
    p50: s[Math.floor(s.length * 0.5)],
    p95: s[Math.min(s.length - 1, Math.floor(s.length * 0.95))],
    max: s[s.length - 1]
  }
}

const time = fn => {
  const t0 = performance.now()
  const out = fn()

  return { ms: performance.now() - t0, out }
}

// --- the stages, transcribed from the shipped code paths -------------------

// electron/hardening.ts :: readFileDataUrlForIpc — main-process side of
// window.hermesDesktop.readFileDataUrl.
const readFileDataUrl = path => {
  const data = readFileSync(path)

  return `data:image/png;base64,${data.toString('base64')}`
}

// use-prompt-actions/utils.ts :: base64FromDataUrl
const base64FromDataUrl = dataUrl => {
  const comma = dataUrl.indexOf(',')

  return comma >= 0 ? dataUrl.slice(comma + 1) : ''
}

// The JSON-RPC frame the renderer sends for image.attach_bytes.
const encodeRpcFrame = (base64, filename) =>
  JSON.stringify({
    jsonrpc: '2.0',
    id: 1,
    method: 'image.attach_bytes',
    params: { session_id: 'bench', content_base64: base64, filename }
  })

// lib/embedded-images.ts :: extractEmbeddedImages — runs on the optimistic
// bubble text on EVERY DirectiveContent render, and the composer's base64
// preview is what it scans.
const DATA_IMAGE_PREFIX = 'data:image/'
const BASE64_MARKER = ';base64,'
const MIN_EMBEDDED_IMAGE_BASE64_LENGTH = 64

const isImageMimeCode = c =>
  (c >= 48 && c <= 57) || (c >= 65 && c <= 90) || (c >= 97 && c <= 122) || c === 43 || c === 45 || c === 46 || c === 95

const isBase64Code = c =>
  (c >= 48 && c <= 57) || (c >= 65 && c <= 90) || (c >= 97 && c <= 122) || c === 43 || c === 47 || c === 61

function readDataImageUrl(text, start) {
  if (!text.startsWith(DATA_IMAGE_PREFIX, start)) {
    return null
  }

  let cursor = start + DATA_IMAGE_PREFIX.length

  while (cursor < text.length && isImageMimeCode(text.charCodeAt(cursor))) {
    cursor += 1
  }

  if (cursor === start + DATA_IMAGE_PREFIX.length || !text.startsWith(BASE64_MARKER, cursor)) {
    return null
  }

  cursor += BASE64_MARKER.length
  const base64Start = cursor

  while (cursor < text.length && isBase64Code(text.charCodeAt(cursor))) {
    cursor += 1
  }

  if (cursor - base64Start < MIN_EMBEDDED_IMAGE_BASE64_LENGTH) {
    return null
  }

  return { end: cursor, url: text.slice(start, cursor) }
}

function extractEmbeddedImages(text) {
  if (!text || !text.includes(DATA_IMAGE_PREFIX)) {
    return { cleanedText: text, images: [] }
  }

  const images = []
  const pieces = []
  let appendCursor = 0
  let searchCursor = 0

  while (searchCursor < text.length) {
    const dataStart = text.indexOf(DATA_IMAGE_PREFIX, searchCursor)

    if (dataStart === -1) {
      break
    }

    const dataUrl = readDataImageUrl(text, dataStart)

    if (!dataUrl) {
      searchCursor = dataStart + DATA_IMAGE_PREFIX.length

      continue
    }

    pieces.push(text.slice(appendCursor, dataStart))
    images.push(dataUrl.url)
    appendCursor = dataUrl.end
    searchCursor = dataUrl.end
  }

  if (!images.length) {
    return { cleanedText: text, images: [] }
  }

  pieces.push(text.slice(appendCursor))

  return {
    cleanedText: pieces
      .join('')
      .replace(/[ \t]+\n/g, '\n')
      .replace(/\n{3,}/g, '\n\n')
      .trim(),
    images
  }
}

// lib/render-weight.ts :: payloadCharacters — walks every string in a message's
// content, including the data URL riding in attachmentRefs.
const RENDER_WEIGHT_CHARS = 512
const MAX_MEASURED_MESSAGE_CHARS = 300 * RENDER_WEIGHT_CHARS
const NON_RENDERED_CONTENT_FIELDS = new Set(['id', 'role', 'toolCallId', 'toolName', 'type'])

function payloadCharacters(roots, budget) {
  const seen = new WeakSet()
  const pending = [...roots]
  let characters = 0

  while (pending.length > 0 && characters < budget) {
    const value = pending.pop()

    if (typeof value === 'string') {
      characters += Math.min(value.length, budget - characters)

      continue
    }

    if (!value || typeof value !== 'object' || seen.has(value)) {
      continue
    }

    seen.add(value)

    if (Array.isArray(value)) {
      for (const nested of value) {
        pending.push(nested)
      }

      continue
    }

    for (const [key, nested] of Object.entries(value)) {
      if (!NON_RENDERED_CONTENT_FIELDS.has(key)) {
        pending.push(nested)
      }
    }
  }

  return characters
}

// store/composer.ts :: cloneDraft — every composer draft stash copies each
// attachment object; previewUrl (the data URL) rides along by reference, but
// the surrounding string ops on the draft still run.
const cloneDraft = draft => ({
  attachments: draft.attachments.map(a => ({ ...a })),
  text: draft.text
})

// --- run -------------------------------------------------------------------

const results = []

for (const kb of SIZES_KB) {
  const path = makeImage(kb)
  const rows = {}
  const record = (stage, ms) => {
    ;(rows[stage] ??= []).push(ms)
  }

  let dataUrl = ''
  let base64 = ''
  let frame = ''
  let bubbleText = ''

  for (let round = 0; round < ROUNDS; round += 1) {
    // 1. preview read (attachImagePath → attachmentPreviewDataUrl)
    const preview = time(() => readFileDataUrl(path))
    record('preview_read_dataurl', preview.ms)
    dataUrl = preview.out

    // 2. submit-time SECOND read of the same file (readImageForRemoteAttach)
    const attachRead = time(() => readFileDataUrl(path))
    record('attach_read_dataurl', attachRead.ms)

    // 3. strip the data: prefix
    const strip = time(() => base64FromDataUrl(attachRead.out))
    record('base64_from_dataurl', strip.ms)
    base64 = strip.out

    // 4. JSON-RPC frame for image.attach_bytes
    const encode = time(() => encodeRpcFrame(base64, 'img.png'))
    record('rpc_frame_encode', encode.ms)
    frame = encode.out

    // 5. the optimistic bubble carries the data URL as its attachmentRef
    bubbleText = dataUrl
    const extract = time(() => extractEmbeddedImages(bubbleText))
    record('extract_embedded_images', extract.ms)

    // 6. render-weight walk over the message holding that ref
    const content = [{ type: 'text', text: 'what is this' }, { attachmentRefs: [dataUrl] }]
    const weigh = time(() => payloadCharacters(content, MAX_MEASURED_MESSAGE_CHARS))
    record('render_weight_walk', weigh.ms)

    // 7. draft stash clone with the attachment held
    const draft = { attachments: [{ id: 'a', kind: 'image', label: 'i', previewUrl: dataUrl, path }], text: 'hi' }
    const clone = time(() => cloneDraft(draft))
    record('draft_clone', clone.ms)
  }

  results.push({
    kb,
    fileBytes: readFileSync(path).length,
    dataUrlChars: dataUrl.length,
    rpcFrameChars: frame.length,
    rows
  })
}

for (const r of results) {
  console.log(`\n=== ${r.kb} KB image (${r.fileBytes} bytes on disk) ===`)
  console.log(
    `data URL: ${r.dataUrlChars.toLocaleString()} chars   RPC frame: ${r.rpcFrameChars.toLocaleString()} chars ` +
      `(${(r.rpcFrameChars / r.fileBytes).toFixed(2)}x the file)`
  )
  console.log('')
  console.log('stage                        mean      p50      p95      max')

  let total = 0

  for (const [stage, samples] of Object.entries(r.rows)) {
    const s = stat(samples)
    total += s.mean
    console.log(
      `${stage.padEnd(26)} ${s.mean.toFixed(2).padStart(7)}  ${s.p50.toFixed(2).padStart(7)}  ` +
        `${s.p95.toFixed(2).padStart(7)}  ${s.max.toFixed(2).padStart(7)}   ms`
    )
  }

  console.log(`${'TOTAL (mean)'.padEnd(26)} ${total.toFixed(2).padStart(7)} ms`)
}
