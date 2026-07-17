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

export type CanvasBlock =
  | AreaChartBlock
  | BarChartBlock
  | DividerBlock
  | ImageBlock
  | InsightBlock
  | KpiBlock
  | LineChartBlock
  | ListBlock
  | PieChartBlock
  | TableBlock

export interface CanvasDatum {
  color?: string
  label: string
  value: number
}

export interface CanvasChartDatum {
  label: string
  [key: string]: number | string | undefined
}

export interface CanvasChartSeries {
  color?: string
  key: string
  label: string
}

export interface BarChartBlock {
  data: CanvasChartDatum[]
  id: string
  series: CanvasChartSeries[]
  stacked?: boolean
  title: string
  type: 'bar-chart'
}
export interface LineChartBlock {
  data: CanvasChartDatum[]
  id: string
  series: CanvasChartSeries[]
  title: string
  type: 'line-chart'
}
export interface AreaChartBlock {
  data: CanvasChartDatum[]
  id: string
  series: CanvasChartSeries[]
  title: string
  type: 'area-chart'
}
export interface InsightBlock {
  body: string
  id: string
  title: string
  type: 'insight'
}
export interface DividerBlock {
  id: string
  type: 'divider'
}
export interface ImageBlock {
  alt: string
  caption?: string
  id: string
  src: string
  type: 'image'
}
export interface KpiBlock {
  items: Array<{ change: string; label: string; value: string }>
  type: 'kpis'
}
export interface ListBlock {
  id: string
  items: string[]
  ordered?: boolean
  title: string
  type: 'list'
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
