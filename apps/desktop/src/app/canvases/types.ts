export interface CanvasDefinition {
  blocks: CanvasBlock[]
  createdAt: string
  id: string
  profile: string
  schema?: 'hermes.canvas/v1'
  schemaVersion?: string
  source: CanvasSource
  summary: string
  title: string
  updatedAt: string
  version: number
}

export interface CanvasSource {
  instructions: string
  prompt: string
}

export type CanvasBlock = BarChartBlock | InsightBlock | KpiBlock | PieChartBlock | TableBlock

export interface CanvasDatum {
  color?: string
  label: string
  value: number
}

export interface BarChartBlock {
  data: CanvasDatum[]
  id: string
  title: string
  type: 'bar-chart'
}
export interface InsightBlock {
  body: string
  id: string
  title: string
  type: 'insight'
}
export interface KpiBlock {
  items: Array<{ change: string; label: string; value: string }>
  type: 'kpis'
}
export interface PieChartBlock {
  data: CanvasDatum[]
  id: string
  title: string
  type: 'pie-chart'
}
export interface TableBlock {
  columns: string[]
  id: string
  rows: string[][]
  title: string
  type: 'table'
}
