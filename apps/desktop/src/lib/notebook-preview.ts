/**
 * Parse a Jupyter notebook (nbformat v4, and the v3-on-disk shape with a
 * single worksheets[0].cells list) into a renderable structure. Does not
 * execute anything — only saved outputs. The preview pane is the consumer.
 */

const IMAGE_MIMES = ['image/png', 'image/jpeg', 'image/gif', 'image/webp'] as const
const ANSI_RE = new RegExp(`${String.fromCharCode(27)}\\[[0-9;]*[A-Za-z]`, 'g')

export function isNotebookPath(path: string): boolean {
  const clean = path.split(/[?#]/, 1)[0] || path

  return /\.ipynb$/i.test(clean)
}

export type NotebookCellKind = 'code' | 'markdown' | 'raw'

export type ParsedNotebookOutput =
  | { html: string; type: 'html' }
  | { dataUrl: string; mime: string; type: 'image' }
  | { svg: string; type: 'svg' }
  | { text: string; type: 'markdown' }
  | { text: string; type: 'text' }
  | { ename: string; evalue: string; traceback: string; type: 'error' }
  | { name: 'stderr' | 'stdout'; text: string; type: 'stream' }
  | { type: 'widget' }

export interface ParsedNotebookCell {
  executionCount: null | number
  kind: NotebookCellKind
  outputs: ParsedNotebookOutput[]
  source: string
}

export interface ParsedNotebook {
  cells: ParsedNotebookCell[]
  language: string
}

export function parseNotebook(text: string): null | ParsedNotebook {
  let parsed: unknown

  try {
    parsed = JSON.parse(text)
  } catch {
    return null
  }

  if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
    return null
  }

  const root = parsed as Record<string, unknown>
  const cells = notebookCells(root)

  if (!cells) {
    return null
  }

  const language = notebookLanguage(root)
  const rendered: ParsedNotebookCell[] = []

  for (const cell of cells) {
    if (!cell || typeof cell !== 'object' || Array.isArray(cell)) {
      continue
    }

    const record = cell as Record<string, unknown>
    const kind = cellKind(record.cell_type)

    if (!kind) {
      continue
    }

    rendered.push({
      executionCount: executionCount(record.execution_count),
      kind,
      outputs: kind === 'code' ? parseOutputs(record.outputs) : [],
      source: joinSource(record.source)
    })
  }

  return { cells: rendered, language }
}

function notebookCells(root: Record<string, unknown>): null | unknown[] {
  if (Array.isArray(root.cells)) {
    return root.cells
  }

  // nbformat v3: cells live under worksheets[0].cells
  const worksheets = root.worksheets

  if (!Array.isArray(worksheets) || worksheets.length === 0) {
    return null
  }

  const first = worksheets[0]

  if (!first || typeof first !== 'object' || Array.isArray(first)) {
    return null
  }

  const cells = (first as Record<string, unknown>).cells

  return Array.isArray(cells) ? cells : null
}

function notebookLanguage(root: Record<string, unknown>): string {
  const metadata = asRecord(root.metadata)
  const kernelspec = asRecord(metadata?.kernelspec)
  const languageInfo = asRecord(metadata?.language_info)
  const fromKernel = stringOrEmpty(kernelspec?.language)
  const fromInfo = stringOrEmpty(languageInfo?.name)

  return (fromKernel || fromInfo || 'python').toLowerCase()
}

function cellKind(value: unknown): NotebookCellKind | null {
  if (value === 'markdown' || value === 'code' || value === 'raw') {
    return value
  }

  // nbformat v3
  if (value === 'heading') {
    return 'markdown'
  }

  return null
}

function executionCount(value: unknown): null | number {
  return typeof value === 'number' && Number.isFinite(value) ? value : null
}

function parseOutputs(value: unknown): ParsedNotebookOutput[] {
  if (!Array.isArray(value)) {
    return []
  }

  const outputs: ParsedNotebookOutput[] = []

  for (const item of value) {
    const parsed = parseOutput(item)

    if (parsed) {
      outputs.push(parsed)
    }
  }

  return outputs
}

function parseOutput(value: unknown): null | ParsedNotebookOutput {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    return null
  }

  const output = value as Record<string, unknown>
  const type = stringOrEmpty(output.output_type)

  if (type === 'stream') {
    const text = cleanStreamText(joinSource(output.text))

    if (!text) {
      return null
    }

    return { name: stringOrEmpty(output.name) === 'stderr' ? 'stderr' : 'stdout', text, type: 'stream' }
  }

  if (type === 'error' || type === 'pyerr') {
    const traceback = Array.isArray(output.traceback)
      ? cleanStreamText(output.traceback.filter(line => typeof line === 'string').join('\n'))
      : ''

    return {
      ename: stringOrEmpty(output.ename),
      evalue: stringOrEmpty(output.evalue),
      traceback,
      type: 'error'
    }
  }

  if (type === 'execute_result' || type === 'display_data' || type === 'pyout') {
    return parseMimeBundle(mimeBundle(output))
  }

  return null
}

function mimeBundle(output: Record<string, unknown>): Record<string, unknown> {
  const data = asRecord(output.data)

  if (data) {
    return data
  }

  // nbformat v3 stores mime payloads flat on the output.
  const bundle: Record<string, unknown> = {}

  if (output.text !== undefined) {
    bundle['text/plain'] = output.text
  }

  if (output.html !== undefined) {
    bundle['text/html'] = output.html
  }

  if (output.png !== undefined) {
    bundle['image/png'] = output.png
  }

  if (output.jpeg !== undefined) {
    bundle['image/jpeg'] = output.jpeg
  }

  if (output.svg !== undefined) {
    bundle['image/svg+xml'] = output.svg
  }

  return bundle
}

function parseMimeBundle(data: Record<string, unknown>): null | ParsedNotebookOutput {
  if ('application/vnd.jupyter.widget-view+json' in data) {
    return { type: 'widget' }
  }

  for (const mime of IMAGE_MIMES) {
    if (mime in data) {
      const dataUrl = imageDataUrl(mime, data[mime])

      if (dataUrl) {
        return { dataUrl, mime, type: 'image' }
      }
    }
  }

  if ('image/svg+xml' in data) {
    const svg = joinSource(data['image/svg+xml']).trim()

    if (svg) {
      return { svg, type: 'svg' }
    }
  }

  if ('text/html' in data) {
    const html = joinSource(data['text/html'])

    if (html.trim()) {
      return { html, type: 'html' }
    }
  }

  if ('text/markdown' in data) {
    const text = joinSource(data['text/markdown'])

    if (text.trim()) {
      return { text, type: 'markdown' }
    }
  }

  if ('application/json' in data) {
    const json = data['application/json']
    const text = typeof json === 'string' ? json : JSON.stringify(json, null, 2)

    if (text.trim()) {
      return { text, type: 'text' }
    }
  }

  if ('text/plain' in data) {
    const text = cleanStreamText(joinSource(data['text/plain']))

    if (text.trim()) {
      return { text, type: 'text' }
    }
  }

  return null
}

function imageDataUrl(mime: string, value: unknown): null | string {
  const payload = joinSource(value).replace(/\s+/g, '')

  if (!payload) {
    return null
  }

  return `data:${mime};base64,${payload}`
}

export function joinSource(value: unknown): string {
  if (typeof value === 'string') {
    return value
  }

  if (Array.isArray(value)) {
    return value.filter(item => typeof item === 'string').join('')
  }

  return ''
}

export function cleanStreamText(text: string): string {
  const cleaned = text.replace(ANSI_RE, '').replace(/\r\n/g, '\n')

  return cleaned
    .split('\n')
    .map(line => {
      const frames = line.split('\r').filter(Boolean)

      return frames.length ? frames[frames.length - 1] : ''
    })
    .join('\n')
}

function asRecord(value: unknown): null | Record<string, unknown> {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    return null
  }

  return value as Record<string, unknown>
}

function stringOrEmpty(value: unknown): string {
  return typeof value === 'string' ? value : ''
}
