'use client'

import { useMemo, useState, useRef, useCallback } from 'react'
import { cn } from '@/lib/utils'
import { ZoomIn, HelpCircle, X, ChevronRight, MessageCircle } from '@/lib/icons'
import type { RichFenceProps } from './types'

interface Series {
  name: string
  values: number[]
  color?: string
  strokeDash?: string
}

interface ScatterPoint {
  x: number
  y: number
  label?: string
  size?: number
  color?: string
}

interface DataItem {
  label: string
  value: number
  color?: string
}

interface ChartSpec {
  type?: 'line' | 'bar' | 'area' | 'scatter' | 'bubble' | 'pie' | 'donut'
  title?: string
  subtitle?: string
  labels?: string[]
  series?: Series[]
  points?: ScatterPoint[]
  slices?: DataItem[]
  data?: any[]
  unit?: string
  unitPosition?: 'prefix' | 'suffix'
  horizontal?: boolean
  stacked?: boolean
  showGrid?: boolean
  showValues?: boolean
}

// Hermes Semantic Palette Tokens
const THEME_PALETTE = [
  '#38bdf8', // Sky 400
  '#34d399', // Emerald 400
  '#fbbf24', // Amber 400
  '#f43f5e', // Rose 500
  '#a78bfa', // Violet 400
  '#22d3ee', // Cyan 400
  '#fb923c', // Orange 400
  '#e879f9', // Fuchsia 400
  '#818cf8', // Indigo 400
  '#4ade80'  // Green 400
]

export function getNiceMax(val: number): number {
  if (val <= 0) return 10
  const power = Math.pow(10, Math.floor(Math.log10(val)))
  const fraction = val / power
  let niceFraction = 10
  if (fraction <= 1.2) niceFraction = 1.2
  else if (fraction <= 2) niceFraction = 2
  else if (fraction <= 2.5) niceFraction = 2.5
  else if (fraction <= 5) niceFraction = 5
  else niceFraction = 10
  return niceFraction * power
}

export function parseChartSpec(code: string): ChartSpec | null {
  try {
    const raw = code.trim()
    if (raw.startsWith('{')) {
      return JSON.parse(raw) as ChartSpec
    }
    // Key-value / list shorthand
    const lines = raw.split('\n').map(l => l.trim()).filter(Boolean)
    const spec: ChartSpec = { type: 'bar', labels: [], series: [{ name: 'Values', values: [] }] }
    for (const line of lines) {
      if (line.startsWith('type:')) spec.type = line.split(':')[1]?.trim() as any
      else if (line.startsWith('title:')) spec.title = line.split(':')[1]?.trim()
      else if (line.startsWith('subtitle:')) spec.subtitle = line.split(':')[1]?.trim()
      else if (line.startsWith('unit:')) spec.unit = line.split(':')[1]?.trim()
      else if (line.includes(':')) {
        const [k, v] = line.split(':')
        const num = parseFloat(v.trim().replace(/[^0-9.-]/g, ''))
        if (!isNaN(num)) {
          spec.labels?.push(k.trim())
          spec.series?.[0].values.push(num)
        }
      }
    }
    return spec.labels?.length || spec.data?.length || spec.points?.length ? spec : null
  } catch {
    return null
  }
}

export default function ChartRenderer({ code, streaming }: RichFenceProps) {
  const [hoverIndex, setHoverIndex] = useState<number | null>(null)
  const [hoverSecondary, setHoverSecondary] = useState<number | null>(null)
  const [showValuesToggle, setShowValuesToggle] = useState<boolean | null>(null)
  const [chartTypeOverride, setChartTypeOverride] = useState<string | null>(null)
  const [isMaximized, setIsMaximized] = useState(false)

  // Explain Panel Overlay State
  const [showExplainPanel, setShowExplainPanel] = useState(false)
  const [selectedPoint, setSelectedPoint] = useState<{
    label: string
    value: string
    context: string
    x?: number | string
    y?: number | string
  } | null>(null)

  // Interactive Resizing States
  const [customWidth, setCustomWidth] = useState<number | null>(null)
  const [isResizing, setIsResizing] = useState(false)
  const cardRef = useRef<HTMLDivElement | null>(null)
  const svgRef = useRef<SVGSVGElement | null>(null)

  const spec = useMemo(() => parseChartSpec(code), [code])

  // Resize drag listener (bottom-right / edge resize)
  const handlePointerDown = useCallback((e: React.PointerEvent) => {
    e.preventDefault()
    e.stopPropagation()
    setIsResizing(true)

    const startX = e.clientX
    const startW = cardRef.current?.getBoundingClientRect().width || 490

    const onPointerMove = (moveEvt: PointerEvent) => {
      const deltaX = moveEvt.clientX - startX
      const nextW = Math.min(840, Math.max(340, startW + deltaX))
      setCustomWidth(nextW)
    }

    const onPointerUp = () => {
      setIsResizing(false)
      window.removeEventListener('pointermove', onPointerMove)
      window.removeEventListener('pointerup', onPointerUp)
    }

    window.addEventListener('pointermove', onPointerMove)
    window.addEventListener('pointerup', onPointerUp)
  }, [])

  // Universal Prompt Injection into Desktop Composer & Clipboard
  const injectPromptToComposer = useCallback((promptText: string) => {
    const richInput = document.querySelector('[data-slot="composer-rich-input"]') as HTMLElement | null
    const textarea = document.querySelector('textarea') as HTMLTextAreaElement | null

    if (richInput) {
      richInput.textContent = promptText
      richInput.dispatchEvent(new Event('input', { bubbles: true }))
      richInput.focus()
    } else if (textarea) {
      textarea.value = promptText
      textarea.dispatchEvent(new Event('input', { bubbles: true }))
      textarea.focus()
    }

    navigator.clipboard.writeText(promptText)
  }, [])

  if (streaming || !spec) {
    return (
      <pre className="overflow-auto p-2.5 font-mono text-[0.7rem] text-(--ui-text-tertiary) leading-relaxed whitespace-pre-wrap">
        {code}
      </pre>
    )
  }

  const {
    type: defaultType = 'bar',
    title = 'Data Visualization',
    subtitle,
    labels = [],
    series = [],
    points = [],
    slices = [],
    data = [],
    unit = '',
    unitPosition = 'prefix',
    horizontal = false,
    stacked = false,
    showGrid = true,
    showValues: defaultShowValues = false
  } = spec

  const type = (chartTypeOverride || defaultType) as 'line' | 'bar' | 'area' | 'scatter' | 'bubble' | 'pie' | 'donut'
  const showValues = showValuesToggle !== null ? showValuesToggle : defaultShowValues

  const formatVal = (val: number) => {
    const formatted = Math.abs(val) >= 1000 ? `${(val / 1000).toFixed(1)}k` : Number.isInteger(val) ? val.toString() : val.toFixed(1)
    return unitPosition === 'prefix' ? `${unit}${formatted}` : `${formatted}${unit}`
  }

  const width = 480
  const isPolar = type === 'pie' || type === 'donut'
  const height = isPolar ? 175 : 165

  // Safe padding with generous right safety margin to eliminate clipping on tooltips
  const padLeft = horizontal ? 68 : 42
  const padRight = (type === 'scatter' || type === 'bubble') ? 34 : 20
  const padTop = 20
  const padBottom = 24
  const chartW = width - padLeft - padRight
  const chartH = height - padTop - padBottom

  // Normalizations for Items
  const normalizedItems: DataItem[] = slices.length
    ? slices
    : data.length && typeof data[0] === 'object' && 'value' in data[0]
    ? data
    : labels.length && series.length && series[0].values.length
    ? labels.map((l, i) => ({
        label: l,
        value: series[0].values[i] || 0,
        color: series[0].color || THEME_PALETTE[i % THEME_PALETTE.length]
      }))
    : []

  const totalPieVal = normalizedItems.reduce((sum, s) => sum + Math.max(0, s.value), 0) || 1

  // Scale calculations
  const allValues: number[] = []
  if (series.length) {
    if (stacked && type === 'bar') {
      for (let i = 0; i < (labels.length || series[0].values.length); i++) {
        const sum = series.reduce((acc, s) => acc + (s.values[i] || 0), 0)
        allValues.push(sum)
      }
    } else {
      allValues.push(...series.flatMap(s => s.values))
    }
  } else if (points.length) {
    allValues.push(...points.map(p => p.y))
  }

  const rawMax = Math.max(...allValues, 1)
  const maxVal = getNiceMax(rawMax)
  const minVal = 0
  const range = maxVal - minVal || 1

  // Key stats for instant explanation
  const avgVal = allValues.length ? allValues.reduce((a, b) => a + b, 0) / allValues.length : 0
  const minObserved = allValues.length ? Math.min(...allValues) : 0
  const maxObserved = allValues.length ? Math.max(...allValues) : 0

  const allXValues = points.map(p => p.x)
  const rawMaxX = points.length ? Math.max(...allXValues, 1) : 100
  const maxXVal = getNiceMax(rawMaxX)
  const minXVal = 0
  const xRange = maxXVal - minXVal || 1

  const getY = (val: number) => padTop + chartH - ((val - minVal) / range) * chartH
  const getX = (idx: number) => padLeft + (idx / Math.max(1, labels.length - 1)) * chartW
  const getBarX = (idx: number) => padLeft + (idx / Math.max(1, labels.length)) * chartW
  const barSlotW = chartW / Math.max(1, labels.length)

  const getHorizontalY = (idx: number) => padTop + (idx / Math.max(1, labels.length)) * chartH
  const getHorizontalBarW = (val: number) => Math.max(2, (val / range) * chartW)
  const horizSlotH = chartH / Math.max(1, labels.length)

  const getScatterX = (xVal: number) => padLeft + ((xVal - minXVal) / xRange) * chartW

  // Pie / Donut Math
  let cumulativeAngle = -Math.PI / 2
  const pieSlices = normalizedItems.map((s, idx) => {
    const fraction = Math.max(0, s.value) / totalPieVal
    const angle = fraction * 2 * Math.PI
    const startAngle = cumulativeAngle
    const endAngle = cumulativeAngle + angle
    cumulativeAngle += angle

    const isHovered = hoverIndex === idx
    const r = isHovered ? 58 : 54
    const innerR = type === 'donut' ? (isHovered ? 34 : 31) : 0
    const cx = 95
    const cy = height / 2

    const x1 = cx + r * Math.cos(startAngle)
    const y1 = cy + r * Math.sin(startAngle)
    const x2 = cx + r * Math.cos(endAngle)
    const y2 = cy + r * Math.sin(endAngle)

    const ix1 = cx + innerR * Math.cos(endAngle)
    const iy1 = cy + innerR * Math.sin(endAngle)
    const ix2 = cx + innerR * Math.cos(startAngle)
    const iy2 = cy + innerR * Math.sin(startAngle)

    const largeArc = angle > Math.PI ? 1 : 0

    let path = ''
    if (fraction >= 0.999) {
      path = type === 'donut'
        ? `M ${cx - r} ${cy} A ${r} ${r} 0 1 0 ${cx + r} ${cy} A ${r} ${r} 0 1 0 ${cx - r} ${cy} M ${cx - innerR} ${cy} A ${innerR} ${innerR} 0 1 1 ${cx + innerR} ${cy} A ${innerR} ${innerR} 0 1 1 ${cx - innerR} ${cy} Z`
        : `M ${cx - r} ${cy} A ${r} ${r} 0 1 0 ${cx + r} ${cy} A ${r} ${r} 0 1 0 ${cx - r} ${cy} Z`
    } else if (type === 'donut') {
      path = `M ${x1} ${y1} A ${r} ${r} 0 ${largeArc} 1 ${x2} ${y2} L ${ix1} ${iy1} A ${innerR} ${innerR} 0 ${largeArc} 0 ${ix2} ${iy2} Z`
    } else {
      path = `M ${cx} ${cy} L ${x1} ${y1} A ${r} ${r} 0 ${largeArc} 1 ${x2} ${y2} Z`
    }

    const color = s.color || THEME_PALETTE[idx % THEME_PALETTE.length]
    return { ...s, path, color, fraction, isHovered }
  })

  // Switchable Chart Types for Linear Data
  const canSwitchType = series.length > 0 && !isPolar
  const availableTypes: Array<'bar' | 'line' | 'area'> = ['bar', 'line', 'area']

  return (
    <div
      ref={cardRef}
      style={isMaximized ? { width: '100%', maxWidth: '100%' } : customWidth ? { width: `${customWidth}px`, maxWidth: '100%' } : undefined}
      className={cn(
        'group/chart relative my-2 rounded-lg border border-(--ui-stroke-tertiary) bg-(--ui-widget-surface-background,var(--ui-surface-primary)) p-3 transition-all shadow-sm',
        !customWidth && !isMaximized && 'max-w-[490px]',
        isResizing && 'select-none ring-1 ring-(--ui-accent)'
      )}
    >
      {/* Header with Title, Subtitle, and Action Toolbar */}
      <div className="mb-2 flex items-center justify-between gap-2 px-0.5">
        <div className="min-w-0 flex-1">
          {title && (
            <div className="text-[0.78125rem] font-semibold text-(--ui-text-primary) tracking-tight truncate">
              {title}
            </div>
          )}
          {subtitle && (
            <div className="text-[0.65625rem] text-(--ui-text-secondary) truncate">
              {subtitle}
            </div>
          )}
        </div>

        {/* Feature-Rich Quick Actions */}
        <div className="flex items-center gap-1 opacity-60 group-hover/chart:opacity-100 transition-opacity">
          {/* Quick Chart Type Switcher */}
          {canSwitchType && (
            <div className="flex items-center rounded-md border border-(--ui-stroke-quaternary) bg-(--ui-surface-secondary) p-0.5 text-[0.625rem]">
              {availableTypes.map(t => (
                <button
                  key={t}
                  type="button"
                  onClick={() => setChartTypeOverride(t)}
                  className={cn(
                    'px-1.5 py-0.5 rounded capitalize transition-colors',
                    type === t
                      ? 'bg-(--ui-accent) text-white font-medium shadow-xs'
                      : 'text-(--ui-text-secondary) hover:text-(--ui-text-primary)'
                  )}
                >
                  {t}
                </button>
              ))}
            </div>
          )}

          {/* Explain Insights & Trend Panel Toggle */}
          <button
            type="button"
            onClick={() => {
              setShowExplainPanel(prev => !prev)
              setSelectedPoint(null)
            }}
            title="Open instant chart insights & explain panel"
            className={cn(
              "flex size-6 items-center justify-center rounded border transition-colors",
              showExplainPanel
                ? "border-(--ui-accent) bg-(--ui-accent)/15 text-(--ui-accent)"
                : "border-(--ui-stroke-quaternary) bg-(--ui-surface-secondary) text-(--ui-text-tertiary) hover:text-(--ui-accent) hover:border-(--ui-accent)/50"
            )}
          >
            <HelpCircle className="size-3.5" />
          </button>

          {/* Toggle Value Labels */}
          {!isPolar && (
            <button
              type="button"
              onClick={() => setShowValuesToggle(prev => (prev === null ? !defaultShowValues : !prev))}
              title="Toggle value labels"
              className={cn(
                'flex size-6 items-center justify-center rounded border transition-colors text-[0.625rem] font-mono font-semibold',
                showValues
                  ? 'border-(--ui-accent) bg-(--ui-accent)/10 text-(--ui-accent)'
                  : 'border-(--ui-stroke-quaternary) bg-(--ui-surface-secondary) text-(--ui-text-tertiary) hover:text-(--ui-text-primary)'
              )}
            >
              123
            </button>
          )}

          {/* Toggle Full-Width / Maximize */}
          <button
            type="button"
            onClick={() => {
              setIsMaximized(prev => !prev)
              setCustomWidth(null)
            }}
            title={isMaximized ? "Restore default width" : "Expand full width"}
            className={cn(
              "flex size-6 items-center justify-center rounded border transition-colors",
              isMaximized
                ? "border-(--ui-accent) bg-(--ui-accent)/10 text-(--ui-accent)"
                : "border-(--ui-stroke-quaternary) bg-(--ui-surface-secondary) text-(--ui-text-tertiary) hover:text-(--ui-text-primary)"
            )}
          >
            <ZoomIn className="size-3" />
          </button>
        </div>
      </div>

      {/* SVG Canvas with Overflow Handling */}
      <div className="relative w-full overflow-visible">
        <svg
          ref={svgRef}
          viewBox={`0 0 ${width} ${height}`}
          className="h-auto w-full select-none font-sans overflow-visible"
        >
          {/* Vertical Linear Grid: 4 Clean Ticks */}
          {!isPolar && !horizontal && (
            <>
              {[0, 0.333, 0.666, 1].map((ratio, i) => {
                const y = padTop + chartH * (1 - ratio)
                const v = minVal + range * ratio
                return (
                  <g key={i}>
                    {showGrid && (
                      <line
                        x1={padLeft}
                        y1={y}
                        x2={width - padRight}
                        y2={y}
                        stroke="var(--ui-stroke-quaternary, var(--ui-stroke-tertiary))"
                        strokeDasharray={ratio === 0 ? 'none' : '2.5 2.5'}
                        strokeWidth={ratio === 0 ? '0.85' : '0.55'}
                      />
                    )}
                    <text
                      x={padLeft - 6}
                      y={y + 3}
                      textAnchor="end"
                      fill="var(--ui-text-tertiary)"
                      fontSize="8.5"
                      fontFamily="inherit"
                    >
                      {formatVal(v)}
                    </text>
                  </g>
                )
              })}
            </>
          )}

          {/* Horizontal Grid */}
          {horizontal && (
            <>
              {[0, 0.5, 1].map((ratio, i) => {
                const x = padLeft + chartW * ratio
                const v = minVal + range * ratio
                return (
                  <g key={i}>
                    {showGrid && (
                      <line
                        x1={x}
                        y1={padTop}
                        x2={x}
                        y2={padTop + chartH}
                        stroke="var(--ui-stroke-quaternary, var(--ui-stroke-tertiary))"
                        strokeDasharray={ratio === 0 ? 'none' : '2.5 2.5'}
                        strokeWidth={ratio === 0 ? '0.85' : '0.55'}
                      />
                    )}
                    <text
                      x={x}
                      y={height - 6}
                      textAnchor="middle"
                      fill="var(--ui-text-tertiary)"
                      fontSize="8.5"
                      fontFamily="inherit"
                    >
                      {formatVal(v)}
                    </text>
                  </g>
                )
              })}
            </>
          )}

          {/* Standard Bar Chart (Vertical) */}
          {type === 'bar' && !horizontal &&
            labels.map((lbl, idx) => {
              const count = series.length
              const groupW = barSlotW * 0.74
              const singleW = stacked ? groupW : groupW / count
              let stackBottom = 0

              return (
                <g key={idx}>
                  {series.map((s, sIdx) => {
                    const val = s.values[idx] || 0
                    const barHeight = Math.max(2, (val / range) * chartH)
                    const y = stacked
                      ? getY(stackBottom + val)
                      : getY(val)
                    const x = stacked
                      ? getBarX(idx) + (barSlotW - groupW) / 2
                      : getBarX(idx) + (barSlotW - groupW) / 2 + sIdx * singleW
                    
                    if (stacked) stackBottom += val

                    const color = s.color || THEME_PALETTE[sIdx % THEME_PALETTE.length]
                    const isHovered = hoverIndex === idx && (hoverSecondary === null || hoverSecondary === sIdx)

                    return (
                      <g key={sIdx}>
                        <rect
                          x={x}
                          y={y}
                          width={Math.max(2, singleW - 1.5)}
                          height={stacked ? barHeight : Math.max(2, chartH - (y - padTop))}
                          fill={color}
                          opacity={isHovered ? 1 : 0.85}
                          rx={stacked ? '1' : '2'}
                          className="transition-all duration-100 cursor-pointer"
                          onMouseEnter={() => {
                            setHoverIndex(idx)
                            setHoverSecondary(sIdx)
                          }}
                          onMouseLeave={() => {
                            setHoverIndex(null)
                            setHoverSecondary(null)
                          }}
                          onClick={() => {
                            setSelectedPoint({
                              label: `${lbl} (${s.name})`,
                              value: formatVal(val),
                              context: `${title} — ${s.name} at ${lbl}`,
                              x: lbl,
                              y: val
                            })
                            setShowExplainPanel(false)
                          }}
                        />
                        {(showValues || isHovered) && (
                          <text
                            x={x + (singleW - 1.5) / 2}
                            y={y - 3}
                            textAnchor="middle"
                            fill="var(--ui-text-primary)"
                            fontSize="8"
                            fontWeight="600"
                          >
                            {formatVal(val)}
                          </text>
                        )}
                      </g>
                    )
                  })}
                </g>
              )
            })}

          {/* Horizontal Bar Chart */}
          {type === 'bar' && horizontal &&
            labels.map((lbl, idx) => {
              const count = series.length
              const groupH = horizSlotH * 0.7
              const singleH = stacked ? groupH : groupH / count
              let stackLeft = 0

              return (
                <g key={idx}>
                  <text
                    x={padLeft - 6}
                    y={getHorizontalY(idx) + horizSlotH / 2 + 3}
                    textAnchor="end"
                    fill={hoverIndex === idx ? 'var(--ui-text-primary)' : 'var(--ui-text-secondary)'}
                    fontSize="8.5"
                    fontWeight={hoverIndex === idx ? '600' : 'normal'}
                  >
                    {lbl}
                  </text>
                  {series.map((s, sIdx) => {
                    const val = s.values[idx] || 0
                    const barW = getHorizontalBarW(val)
                    const x = padLeft + (stacked ? stackLeft : 0)
                    const y = stacked
                      ? getHorizontalY(idx) + (horizSlotH - groupH) / 2
                      : getHorizontalY(idx) + (horizSlotH - groupH) / 2 + sIdx * singleH
                    
                    if (stacked) stackLeft += barW

                    const color = s.color || THEME_PALETTE[sIdx % THEME_PALETTE.length]
                    const isHovered = hoverIndex === idx && (hoverSecondary === null || hoverSecondary === sIdx)

                    return (
                      <g key={sIdx}>
                        <rect
                          x={x}
                          y={y}
                          width={barW}
                          height={Math.max(2, singleH - 1.5)}
                          fill={color}
                          opacity={isHovered ? 1 : 0.85}
                          rx="1.5"
                          className="transition-all duration-100 cursor-pointer"
                          onMouseEnter={() => {
                            setHoverIndex(idx)
                            setHoverSecondary(sIdx)
                          }}
                          onMouseLeave={() => {
                            setHoverIndex(null)
                            setHoverSecondary(null)
                          }}
                          onClick={() => {
                            setSelectedPoint({
                              label: `${lbl} (${s.name})`,
                              value: formatVal(val),
                              context: `${title} — ${s.name} at ${lbl}`,
                              x: lbl,
                              y: val
                            })
                            setShowExplainPanel(false)
                          }}
                        />
                        {(showValues || isHovered) && (
                          <text
                            x={x + barW + 4}
                            y={y + singleH / 2 + 2.5}
                            fill="var(--ui-text-primary)"
                            fontSize="8.5"
                            fontWeight="600"
                          >
                            {formatVal(val)}
                          </text>
                        )}
                      </g>
                    )
                  })}
                </g>
              )
            })}

          {/* Line & Area Chart */}
          {(type === 'line' || type === 'area') &&
            series.map((s, sIdx) => {
              const color = s.color || THEME_PALETTE[sIdx % THEME_PALETTE.length]
              const pointsStr = s.values.map((v, i) => `${getX(i)},${getY(v)}`).join(' ')
              const areaPoints = [
                `${getX(0)},${padTop + chartH}`,
                pointsStr,
                `${getX(s.values.length - 1)},${padTop + chartH}`
              ].join(' ')

              return (
                <g key={sIdx}>
                  {type === 'area' && (
                    <polygon
                      points={areaPoints}
                      fill={color}
                      fillOpacity="0.14"
                    />
                  )}
                  <polyline
                    fill="none"
                    stroke={color}
                    strokeWidth="2"
                    strokeDasharray={s.strokeDash || 'none'}
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    points={pointsStr}
                  />
                  {s.values.map((v, i) => {
                    const cx = getX(i)
                    const cy = getY(v)
                    const isHovered = hoverIndex === i
                    const lbl = labels[i] || `Index ${i}`

                    // Edge-aware tooltip anchoring
                    const tooltipW = 56
                    const clampedBoxX = Math.min(width - tooltipW - 4, Math.max(4, cx - tooltipW / 2))

                    return (
                      <g key={i}>
                        <circle
                          cx={cx}
                          cy={cy}
                          r={isHovered ? 4.5 : 2.75}
                          fill={color}
                          stroke="var(--ui-widget-surface-background, var(--ui-surface-primary))"
                          strokeWidth="1.5"
                          className="transition-all duration-100 cursor-pointer"
                          onMouseEnter={() => setHoverIndex(i)}
                          onMouseLeave={() => setHoverIndex(null)}
                          onClick={() => {
                            setSelectedPoint({
                              label: `${lbl} (${s.name})`,
                              value: formatVal(v),
                              context: `${title} — ${s.name} at ${lbl}`,
                              x: lbl,
                              y: v
                            })
                            setShowExplainPanel(false)
                          }}
                        />
                        {showValues && !isHovered && (
                          <text
                            x={cx}
                            y={cy - 6}
                            textAnchor="middle"
                            fill="var(--ui-text-primary)"
                            fontSize="7.5"
                            fontWeight="500"
                          >
                            {formatVal(v)}
                          </text>
                        )}
                        {isHovered && (
                          <g>
                            <rect
                              x={clampedBoxX}
                              y={cy - 18}
                              width={tooltipW}
                              height="13"
                              rx="2"
                              fill="#0f172a"
                              stroke="rgba(255,255,255,0.15)"
                              strokeWidth="0.5"
                            />
                            <text
                              x={clampedBoxX + tooltipW / 2}
                              y={cy - 8.5}
                              textAnchor="middle"
                              fill="#ffffff"
                              fontSize="8"
                              fontWeight="600"
                            >
                              {formatVal(v)}
                            </text>
                          </g>
                        )}
                      </g>
                    )
                  })}
                </g>
              )
            })}

          {/* Scatter / Bubble Plot with Edge-Aware Collision Proof Tooltips */}
          {(type === 'scatter' || type === 'bubble') && (
            <g>
              {points.map((pt, idx) => {
                const cx = getScatterX(pt.x)
                const cy = getY(pt.y)
                const isHovered = hoverIndex === idx
                const color = pt.color || THEME_PALETTE[idx % THEME_PALETTE.length]
                const radius = pt.size ? Math.min(12, Math.max(4, pt.size * 0.55)) : (type === 'bubble' ? 6.5 : (isHovered ? 5.5 : 3.5))
                const tooltipText = pt.label || `${pt.x}, ${formatVal(pt.y)}`
                const tooltipW = Math.max(68, tooltipText.length * 6.2 + 14)

                // Edge collision protection: clamp tooltip box inside canvas bounds
                const minBoxX = 6
                const maxBoxX = width - tooltipW - 6
                const idealBoxX = cx - tooltipW / 2
                const clampedBoxX = Math.min(maxBoxX, Math.max(minBoxX, idealBoxX))

                return (
                  <g key={idx}>
                    <circle
                      cx={cx}
                      cy={cy}
                      r={radius}
                      fill={color}
                      fillOpacity="0.88"
                      stroke="var(--ui-widget-surface-background, var(--ui-surface-primary))"
                      strokeWidth="1.5"
                      className="transition-all duration-100 cursor-pointer"
                      onMouseEnter={() => setHoverIndex(idx)}
                      onMouseLeave={() => setHoverIndex(null)}
                      onClick={() => {
                        setSelectedPoint({
                          label: pt.label || `Point (${pt.x}, ${formatVal(pt.y)})`,
                          value: formatVal(pt.y),
                          context: `${title} — X: ${pt.x}, Y: ${formatVal(pt.y)}${pt.size ? `, Size: ${pt.size}` : ''}`,
                          x: pt.x,
                          y: pt.y
                        })
                        setShowExplainPanel(false)
                      }}
                    />
                    {showValues && !isHovered && (
                      <text
                        x={cx}
                        y={cy - radius - 3}
                        textAnchor="middle"
                        fill="var(--ui-text-primary)"
                        fontSize="7.5"
                        fontWeight="600"
                      >
                        {formatVal(pt.y)}
                      </text>
                    )}
                    {isHovered && (
                      <g>
                        <rect
                          x={clampedBoxX}
                          y={cy - radius - 16}
                          width={tooltipW}
                          height="14"
                          rx="3"
                          fill="#0f172a"
                          stroke="rgba(255,255,255,0.25)"
                          strokeWidth="0.75"
                        />
                        <text
                          x={clampedBoxX + tooltipW / 2}
                          y={cy - radius - 6}
                          textAnchor="middle"
                          fill="#ffffff"
                          fontSize="8"
                          fontWeight="600"
                        >
                          {tooltipText}
                        </text>
                      </g>
                    )}
                  </g>
                )
              })}
              {[0, 0.5, 1].map((ratio, i) => {
                const xVal = minXVal + xRange * ratio
                const xPos = padLeft + chartW * ratio
                return (
                  <text
                    key={i}
                    x={xPos}
                    y={height - 6}
                    textAnchor="middle"
                    fill="var(--ui-text-tertiary)"
                    fontSize="8.5"
                  >
                    {xVal >= 1000 ? `${(xVal / 1000).toFixed(1)}k` : xVal.toFixed(0)}
                  </text>
                )
              })}
            </g>
          )}

          {/* Pie / Donut Chart */}
          {(type === 'pie' || type === 'donut') && (
            <g>
              {pieSlices.map((s, idx) => (
                <path
                  key={idx}
                  d={s.path}
                  fill={s.color}
                  fillOpacity={s.isHovered ? 1 : 0.85}
                  stroke="var(--ui-widget-surface-background, var(--ui-surface-primary))"
                  strokeWidth="1.5"
                  className="transition-all duration-150 cursor-pointer"
                  onMouseEnter={() => setHoverIndex(idx)}
                  onMouseLeave={() => setHoverIndex(null)}
                  onClick={() => {
                    setSelectedPoint({
                      label: s.label,
                      value: `${formatVal(s.value)} (${((s.value / totalPieVal) * 100).toFixed(0)}%)`,
                      context: `${title} — Slice ${s.label}`,
                      x: s.label,
                      y: s.value
                    })
                    setShowExplainPanel(false)
                  }}
                />
              ))}
              {type === 'donut' && (
                <g>
                  <text
                    x={95}
                    y={height / 2 + 4}
                    textAnchor="middle"
                    fill="var(--ui-text-primary)"
                    fontSize="11"
                    fontWeight="600"
                  >
                    {hoverIndex !== null && normalizedItems[hoverIndex]
                      ? formatVal(normalizedItems[hoverIndex].value)
                      : formatVal(totalPieVal)}
                  </text>
                  <text
                    x={95}
                    y={height / 2 + 14}
                    textAnchor="middle"
                    fill="var(--ui-text-tertiary)"
                    fontSize="8"
                  >
                    {hoverIndex !== null && normalizedItems[hoverIndex]
                      ? `${((normalizedItems[hoverIndex].value / totalPieVal) * 100).toFixed(0)}%`
                      : 'Total'}
                  </text>
                </g>
              )}
              {/* Legend beside Pie */}
              <g transform="translate(180, 20)">
                {normalizedItems.map((s, idx) => (
                  <g
                    key={idx}
                    transform={`translate(0, ${idx * 17})`}
                    className="cursor-pointer"
                    onMouseEnter={() => setHoverIndex(idx)}
                    onMouseLeave={() => setHoverIndex(null)}
                    onClick={() => {
                      setSelectedPoint({
                        label: s.label,
                        value: `${formatVal(s.value)} (${((s.value / totalPieVal) * 100).toFixed(0)}%)`,
                        context: `${title} — Slice ${s.label}`,
                        x: s.label,
                        y: s.value
                      })
                      setShowExplainPanel(false)
                    }}
                  >
                    <rect
                      x="0"
                      y="1.5"
                      width="7.5"
                      height="7.5"
                      rx="1.5"
                      fill={s.color || THEME_PALETTE[idx % THEME_PALETTE.length]}
                    />
                    <text
                      x="13"
                      y="8.5"
                      fill={hoverIndex === idx ? 'var(--ui-text-primary)' : 'var(--ui-text-secondary)'}
                      fontSize="9"
                      fontWeight={hoverIndex === idx ? '600' : 'normal'}
                    >
                      {s.label}
                    </text>
                    <text
                      x="260"
                      y="8.5"
                      textAnchor="end"
                      fill="var(--ui-text-tertiary)"
                      fontSize="8.5"
                    >
                      {formatVal(s.value)} ({((s.value / totalPieVal) * 100).toFixed(0)}%)
                    </text>
                  </g>
                ))}
              </g>
            </g>
          )}

          {/* X Axis Labels for Linear Charts */}
          {!isPolar && !horizontal && type !== 'scatter' && type !== 'bubble' &&
            labels.map((lbl, idx) => {
              const x = type === 'bar' ? getBarX(idx) + barSlotW / 2 : getX(idx)
              return (
                <text
                  key={idx}
                  x={x}
                  y={height - 6}
                  textAnchor="middle"
                  fill={hoverIndex === idx ? 'var(--ui-text-primary)' : 'var(--ui-text-tertiary)'}
                  fontSize="8.5"
                  fontWeight={hoverIndex === idx ? '600' : 'normal'}
                >
                  {lbl}
                </text>
              )
            })}
        </svg>
      </div>

      {/* Series Legend for Linear Charts */}
      {series.length > 1 && !isPolar && (
        <div className="mt-2 flex flex-wrap items-center justify-center gap-3 text-[0.65625rem] text-(--ui-text-secondary)">
          {series.map((s, idx) => (
            <div key={idx} className="flex items-center gap-1">
              <div
                className="h-1.5 w-1.5 rounded-full"
                style={{ backgroundColor: s.color || THEME_PALETTE[idx % THEME_PALETTE.length] }}
              />
              <span>{s.name}</span>
            </div>
          ))}
        </div>
      )}

      {/* Instant Chart Insights & Statistical Overview Drawer */}
      {showExplainPanel && (
        <div className="mt-2.5 rounded-md border border-(--ui-stroke-tertiary) bg-(--ui-surface-secondary) p-3 transition-all animate-in fade-in slide-in-from-top-1 duration-150">
          <div className="flex items-center justify-between gap-2 border-b border-(--ui-stroke-quaternary) pb-1.5">
            <div className="flex items-center gap-1.5 text-[0.75rem] font-semibold text-(--ui-text-primary)">
              <span className="size-2 rounded-full bg-(--ui-accent) animate-pulse" />
              <span>Chart Intelligence & Insights</span>
            </div>
            <button
              type="button"
              onClick={() => setShowExplainPanel(false)}
              className="text-(--ui-text-tertiary) hover:text-(--ui-text-primary)"
            >
              <X className="size-3.5" />
            </button>
          </div>

          {/* Quick Metrics Bar */}
          <div className="mt-2 grid grid-cols-3 gap-2 text-center">
            <div className="rounded bg-(--ui-widget-surface-background,var(--ui-surface-primary)) p-1.5 border border-(--ui-stroke-quaternary)">
              <div className="text-[0.625rem] text-(--ui-text-tertiary)">Average</div>
              <div className="text-[0.75rem] font-mono font-semibold text-(--ui-text-primary)">
                {formatVal(avgVal)}
              </div>
            </div>
            <div className="rounded bg-(--ui-widget-surface-background,var(--ui-surface-primary)) p-1.5 border border-(--ui-stroke-quaternary)">
              <div className="text-[0.625rem] text-(--ui-text-tertiary)">Min Observed</div>
              <div className="text-[0.75rem] font-mono font-semibold text-emerald-400">
                {formatVal(minObserved)}
              </div>
            </div>
            <div className="rounded bg-(--ui-widget-surface-background,var(--ui-surface-primary)) p-1.5 border border-(--ui-stroke-quaternary)">
              <div className="text-[0.625rem] text-(--ui-text-tertiary)">Peak / Max</div>
              <div className="text-[0.75rem] font-mono font-semibold text-rose-400">
                {formatVal(maxObserved)}
              </div>
            </div>
          </div>

          <div className="mt-2.5 flex flex-wrap items-center gap-1.5">
            <button
              type="button"
              onClick={() => {
                const prompt = `Can you provide a comprehensive executive summary and key takeaway insights from this "${title}" chart? (${subtitle || 'Data breakdown'})`
                injectPromptToComposer(prompt)
                setShowExplainPanel(false)
              }}
              className="inline-flex items-center gap-1 rounded bg-(--ui-accent) px-2.5 py-1 text-[0.65625rem] font-medium text-white shadow-xs hover:opacity-90 transition-opacity"
            >
              <MessageCircle className="size-3" />
              <span>Ask Hermes to explain trends</span>
            </button>
            <button
              type="button"
              onClick={() => {
                const prompt = `Analyze the outliers, anomalies, and key inflection points in this "${title}" chart dataset.`
                injectPromptToComposer(prompt)
                setShowExplainPanel(false)
              }}
              className="inline-flex items-center gap-1 rounded border border-(--ui-stroke-quaternary) bg-(--ui-widget-surface-background,var(--ui-surface-primary)) px-2 py-1 text-[0.65625rem] font-medium text-(--ui-text-primary) hover:border-(--ui-accent) transition-colors"
            >
              <span>Spot anomalies</span>
            </button>
          </div>
        </div>
      )}

      {/* Interactive Point Deep Dive & Explain Modal / Bottom-Sheet */}
      {selectedPoint && (
        <div className="mt-2.5 rounded-md border border-(--ui-stroke-tertiary) bg-(--ui-surface-secondary) p-2.5 transition-all animate-in fade-in slide-in-from-top-1 duration-150">
          <div className="flex items-start justify-between gap-2">
            <div>
              <div className="flex items-center gap-1.5">
                <span className="h-2 w-2 rounded-full bg-(--ui-accent)" />
                <span className="text-[0.75rem] font-semibold text-(--ui-text-primary)">
                  {selectedPoint.label}
                </span>
                <span className="rounded bg-(--ui-accent)/10 px-1 py-0.2 text-[0.6875rem] font-mono font-medium text-(--ui-accent)">
                  {selectedPoint.value}
                </span>
              </div>
              <p className="mt-1 text-[0.6875rem] text-(--ui-text-secondary) leading-relaxed">
                Analyze this specific data point, reason about outliers, or explain unit economics.
              </p>
            </div>
            <button
              type="button"
              onClick={() => setSelectedPoint(null)}
              className="text-(--ui-text-tertiary) hover:text-(--ui-text-primary)"
            >
              <X className="size-3.5" />
            </button>
          </div>

          <div className="mt-2 flex flex-wrap items-center gap-1.5">
            <button
              type="button"
              onClick={() => {
                const prompt = `Why is ${selectedPoint.label} at ${selectedPoint.value} in the "${title}" chart? Can you explain the drivers and economics behind this?`
                injectPromptToComposer(prompt)
                setSelectedPoint(null)
              }}
              className="inline-flex items-center gap-1 rounded bg-(--ui-accent) px-2 py-1 text-[0.65625rem] font-medium text-white shadow-xs hover:opacity-90 transition-opacity"
            >
              <span>Explain this point</span>
              <ChevronRight className="size-3" />
            </button>
            <button
              type="button"
              onClick={() => {
                const prompt = `Compare ${selectedPoint.label} (${selectedPoint.value}) against the rest of the dataset in "${title}" and show where the biggest optimization opportunities are.`
                injectPromptToComposer(prompt)
                setSelectedPoint(null)
              }}
              className="inline-flex items-center gap-1 rounded border border-(--ui-stroke-quaternary) bg-(--ui-widget-surface-background,var(--ui-surface-primary)) px-2 py-1 text-[0.65625rem] font-medium text-(--ui-text-primary) hover:border-(--ui-accent) transition-colors"
            >
              <span>Compare with average</span>
            </button>
          </div>
        </div>
      )}

      {/* Corner Resize Handle */}
      <div
        onPointerDown={handlePointerDown}
        title="Drag to resize chart"
        className="absolute bottom-1 right-1 flex size-3.5 cursor-nwse-resize items-center justify-center rounded opacity-25 hover:opacity-100 group-hover/chart:opacity-60 transition-opacity"
      >
        <svg
          width="8"
          height="8"
          viewBox="0 0 8 8"
          className="text-(--ui-text-tertiary) hover:text-(--ui-text-primary)"
        >
          <line x1="6.5" y1="1.5" x2="1.5" y2="6.5" stroke="currentColor" strokeWidth="1" strokeLinecap="round" />
          <line x1="6.5" y1="4" x2="4" y2="6.5" stroke="currentColor" strokeWidth="1" strokeLinecap="round" />
        </svg>
      </div>
    </div>
  )
}
