import type { CanvasBlock, CanvasDatum, CanvasDefinition } from './types'
import { CANVAS_SCHEMA_VERSION } from './canvas-policy'

type JsonObject = Record<string, unknown>

function object(value: unknown): JsonObject | null {
  return value && typeof value === 'object' && !Array.isArray(value) ? (value as JsonObject) : null
}

function string(value: unknown, fallback = ''): string {
  return typeof value === 'string' ? value : fallback
}

function number(value: unknown): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : 0
}

function scalar(value: unknown): boolean {
  return value === null || ['string', 'number', 'boolean'].includes(typeof value)
}

function display(value: unknown): string {
  if (value === null || value === undefined) return ''
  if (typeof value === 'string') return value
  if (typeof value === 'number' || typeof value === 'boolean') return String(value)

  try {
    return JSON.stringify(value)
  } catch {
    return String(value)
  }
}

function label(value: string): string {
  return value
    .replace(/[_-]+/g, ' ')
    .replace(/([a-z0-9])([A-Z])/g, '$1 $2')
    .replace(/^\w/, character => character.toUpperCase())
}

function tableRows(value: unknown): { columns: string[]; rows: string[][] } | null {
  if (!Array.isArray(value) || value.length === 0) return null

  if (value.every(item => Array.isArray(item))) {
    const width = Math.max(...value.map(row => row.length))
    return {
      columns: Array.from({ length: width }, (_, index) =>
        index === 0 ? 'Label' : index === 1 ? 'Value' : `Column ${index + 1}`
      ),
      rows: value.map(row => Array.from({ length: width }, (_, index) => display(row[index])))
    }
  }

  if (value.every(item => scalar(item))) {
    return { columns: ['Value'], rows: value.map(item => [display(item)]) }
  }

  const objects = value.map(object).filter((item): item is JsonObject => Boolean(item))
  if (objects.length === 0) return null
  const columns = Array.from(new Set(objects.flatMap(row => Object.keys(row).filter(key => scalar(row[key])))))
  if (columns.length === 0) return null

  return {
    columns: columns.map(label),
    rows: objects.map(row => columns.map(column => display(row[column])))
  }
}

function chartRows(value: unknown): CanvasDatum[] {
  if (Array.isArray(value)) {
    return value.flatMap(item => {
      if (Array.isArray(item) && item.length >= 2 && Number.isFinite(Number(item[1]))) {
        return [{ label: display(item[0]), value: Number(item[1]) }]
      }
      const row = object(item)
      if (!row) return []
      const labelEntry = Object.entries(row).find(([, entry]) => typeof entry === 'string')
      const valueEntry = Object.entries(row).find(([, entry]) => typeof entry === 'number' && Number.isFinite(entry))
      return labelEntry && valueEntry ? [{ label: display(labelEntry[1]), value: Number(valueEntry[1]) }] : []
    })
  }

  const record = object(value)
  if (!record) return []
  return Object.entries(record).flatMap(([key, entry]) =>
    typeof entry === 'number' && Number.isFinite(entry) ? [{ label: label(key), value: entry }] : []
  )
}

function documentBlocks(payload: JsonObject): CanvasBlock[] {
  const documentObject = object(payload.document)
  const document = Array.isArray(payload.document)
    ? payload.document
    : Array.isArray(documentObject?.sections)
      ? documentObject.sections
      : []
  const blocks: CanvasBlock[] = []
  document.forEach((section, sectionIndex) => {
    const group = object(section)
    const title = string(group?.title, `Section ${sectionIndex + 1}`)
    const children = Array.isArray(group?.children)
      ? group.children
      : Array.isArray(group?.elements)
        ? group.elements
        : []
    const kpis = children.flatMap(child => {
      const item = object(child)
      const kind = string(item?.kind, string(item?.type))
      return kind === 'kpi'
        ? [
            {
              label: string(item?.label),
              value: `${display(item?.value)}${string(item?.unit)}`,
              change: string(item?.change, string(item?.deltaPct))
            }
          ]
        : []
    })
    if (kpis.length) blocks.push({ type: 'kpis', items: kpis })
    children.forEach((child, index) => {
      const item = object(child)
      const kind = string(item?.kind, string(item?.type))
      if (kind === 'text' || kind === 'callout')
        blocks.push({
          type: 'insight',
          id: `${sectionIndex}-${index}`,
          title: string(item?.title, title),
          body: string(item?.text)
        })
      if (kind === 'table') {
        const table = tableRows(item?.rows)
        const columns = Array.isArray(item?.columns)
          ? item.columns.map(value => display(object(value)?.label ?? value))
          : table?.columns || []
        if (table)
          blocks.push({
            type: 'table',
            id: `${sectionIndex}-${index}`,
            title: string(item?.title, title),
            columns,
            rows: table.rows
          })
      }
      if (kind === 'list' && Array.isArray(item?.items))
        blocks.push({
          type: 'list',
          id: `${sectionIndex}-${index}`,
          title: string(item?.title, title),
          items: item.items.map(display),
          ordered: Boolean(item?.ordered)
        })
      if (kind === 'image') {
        const src = string(item?.src, string(item?.dataUri, string(item?.url)))
        if (src)
          blocks.push({
            type: 'image',
            id: `${sectionIndex}-${index}`,
            src,
            alt: string(item?.alt, title),
            caption: string(item?.caption)
          })
      }
      if (kind === 'divider') blocks.push({ type: 'divider', id: `${sectionIndex}-${index}` })
      if (kind === 'chart') {
        const labels = Array.isArray(item?.labels) ? item.labels.map(value => String(value)) : []
        const series = Array.isArray(item?.series) ? object(item.series[0]) : null
        const values = Array.isArray(series?.data) ? series.data : []
        const chartData = labels.map((label, valueIndex) => ({ label, value: number(values[valueIndex]) }))
        const chartType = string(item?.chartType, string(item?.variant))
        const blockType =
          chartType === 'pie' || chartType === 'donut'
            ? 'pie-chart'
            : chartType === 'line'
              ? 'line-chart'
              : chartType === 'area'
                ? 'area-chart'
                : 'bar-chart'
        if (chartData.length)
          blocks.push({
            type: blockType,
            id: `${sectionIndex}-${index}`,
            title: string(item?.title, string(series?.name, title)),
            data: chartData
          })
      }
    })
  })
  return blocks
}

function genericDataBlocks(payload: JsonObject): CanvasBlock[] {
  const reportData = object(payload.data)
  if (!reportData) return []
  const blocks: CanvasBlock[] = []

  Object.entries(reportData).forEach(([key, value], index) => {
    const title = label(key)
    const record = object(value)

    if (scalar(value)) {
      blocks.push({ type: 'insight', id: `data-${index}`, title, body: display(value) })
      return
    }

    if (record && Object.values(record).every(scalar)) {
      const entries = Object.entries(record)
      const numeric = chartRows(record)

      if (numeric.length === entries.length && entries.length > 1 && entries.length <= 24 && key !== 'totals') {
        blocks.push({ type: 'bar-chart', id: `data-${index}`, title, data: numeric })
      } else {
        blocks.push({
          type: 'kpis',
          items: entries.map(([entryKey, entry]) => ({ label: label(entryKey), value: display(entry), change: '' }))
        })
      }
      return
    }

    const table = tableRows(value)
    if (!table) return
    const chart = chartRows(value)
    if (chart.length === table.rows.length && chart.length > 1 && chart.length <= 24 && table.columns.length <= 2) {
      blocks.push({ type: 'bar-chart', id: `data-chart-${index}`, title, data: chart })
    } else {
      blocks.push({ type: 'table', id: `data-table-${index}`, title, ...table })
    }
  })

  return blocks
}

function generatedBlocks(payload: JsonObject): CanvasBlock[] {
  const declared = documentBlocks(payload)
  if (declared.length) return declared

  return genericDataBlocks(payload)
}

export function parseCanvasManifest(text: string, fallbackProfile: string): CanvasDefinition {
  const payload = object(JSON.parse(text))
  if (
    !payload ||
    (string(payload.schema) !== 'hermes.canvas/v1' && string(payload.schemaVersion) !== CANVAS_SCHEMA_VERSION)
  ) {
    throw new Error('El manifiesto no cumple el formato hermes.canvas/v1')
  }
  const source = object(payload.source) || {}
  const blocks = Array.isArray(payload.blocks) ? (payload.blocks as CanvasBlock[]) : generatedBlocks(payload)
  if (blocks.length === 0) throw new Error('El manifiesto no contiene bloques visuales')

  return {
    schema: 'hermes.canvas/v1',
    schemaVersion: string(payload.schemaVersion, CANVAS_SCHEMA_VERSION),
    id: string(payload.id),
    title: string(payload.title, 'Canvas sin título'),
    profile: string(payload.profile, fallbackProfile),
    summary: string(payload.summary, string(payload.title, '')),
    createdAt: string(payload.createdAt, string(payload.updatedAt)),
    updatedAt: string(payload.updatedAt),
    version: number(payload.version) || 1,
    source: { prompt: string(source.prompt), instructions: string(source.instructions) },
    blocks
  }
}
