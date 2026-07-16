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

function data(value: unknown, labelKey: string, valueKey: string): CanvasDatum[] {
  return Array.isArray(value)
    ? value.flatMap(item => {
        const row = object(item)
        if (!row) return []
        const label = string(row[labelKey])
        return label ? [{ label, value: number(row[valueKey]) }] : []
      })
    : []
}

function documentBlocks(payload: JsonObject): CanvasBlock[] {
  const document = Array.isArray(payload.document) ? payload.document : []
  const blocks: CanvasBlock[] = []
  document.forEach((section, sectionIndex) => {
    const group = object(section)
    const title = string(group?.title, `Section ${sectionIndex + 1}`)
    const children = Array.isArray(group?.children) ? group.children : []
    const kpis = children.flatMap(child => {
      const item = object(child)
      return string(item?.kind) === 'kpi'
        ? [{ label: string(item?.label), value: `${number(item?.value)}${string(item?.unit)}`, change: string(item?.deltaPct) }]
        : []
    })
    if (kpis.length) blocks.push({ type: 'kpis', items: kpis })
    children.forEach((child, index) => {
      const item = object(child)
      const kind = string(item?.kind)
      if (kind === 'text' || kind === 'callout') blocks.push({ type: 'insight', id: `${sectionIndex}-${index}`, title: string(item?.title, title), body: string(item?.text) })
      if (kind === 'table') blocks.push({ type: 'table', id: `${sectionIndex}-${index}`, title, columns: Array.isArray(item?.columns) ? item.columns.map(value => String(value)) : [], rows: Array.isArray(item?.rows) ? item.rows.map(row => Array.isArray(row) ? row.map(value => String(value)) : []) : [] })
      if (kind === 'chart') {
        const labels = Array.isArray(item?.labels) ? item.labels.map(value => String(value)) : []
        const series = Array.isArray(item?.series) ? object(item.series[0]) : null
        const values = Array.isArray(series?.data) ? series.data : []
        const chartData = labels.map((label, valueIndex) => ({ label, value: number(values[valueIndex]) }))
        if (chartData.length) blocks.push({ type: string(item?.type) === 'pie' || string(item?.type) === 'donut' ? 'pie-chart' : 'bar-chart', id: `${sectionIndex}-${index}`, title: string(series?.name, title), data: chartData })
      }
    })
  })
  return blocks
}

function generatedBlocks(payload: JsonObject): CanvasBlock[] {
  const declared = documentBlocks(payload)
  if (declared.length) return declared
  const reportData = object(payload.data) || {}
  const kpis = object(reportData.kpis) || {}
  const trend = data(reportData.trend, 'date', 'sessions').map(item => ({
    ...item,
    label: item.label.slice(5)
  }))
  const traffic = data(reportData.trafficSources, 'name', 'sessions')
  const topPages = Array.isArray(reportData.topPages)
    ? reportData.topPages.flatMap(item => {
        const row = object(item)
        const path = string(row?.path)
        return path
          ? [[String(number(row?.rank) || 0), path, String(number(row?.pageViews)), String(number(row?.sessions))]]
          : []
      })
    : []
  const ranking = Array.isArray(reportData.advisorRanking)
    ? reportData.advisorRanking.flatMap(item => {
        const row = object(item)
        const name = string(row?.name)
        return name ? [[name, String(number(row?.sessions))]] : []
      })
    : []
  const blocks: CanvasBlock[] = []

  if (Object.keys(kpis).length > 0) {
    blocks.push({
      type: 'kpis',
      items: [
        { label: 'Sesiones', value: String(number(kpis.sessions)), change: '' },
        { label: 'Usuarios', value: String(number(kpis.users)), change: '' },
        { label: 'Sesiones con interacción', value: String(number(kpis.engagedSessions)), change: '' }
      ]
    })
  }
  if (trend.length > 0) blocks.push({ type: 'bar-chart', id: 'trend', title: 'Sesiones por día', data: trend })
  if (traffic.length > 0) blocks.push({ type: 'pie-chart', id: 'traffic', title: 'Fuentes de tráfico', data: traffic })
  if (ranking.length > 0)
    blocks.push({
      type: 'table',
      id: 'ranking',
      title: 'Asesores con más sesiones',
      columns: ['Asesor', 'Sesiones'],
      rows: ranking
    })
  if (topPages.length > 0)
    blocks.push({
      type: 'table',
      id: 'top-pages',
      title: 'Top pages',
      columns: ['#', 'Page', 'Views', 'Sessions'],
      rows: topPages
    })
  if (string(reportData.scopeNote))
    blocks.push({ type: 'insight', id: 'scope', title: 'Contexto', body: string(reportData.scopeNote) })

  return blocks
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
