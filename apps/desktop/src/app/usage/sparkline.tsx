import { memo, useCallback, useState } from 'react'

import { cn } from '@/lib/utils'

interface BarSparklineProps {
  data: number[]
  color: string
  height?: number
  barGap?: number
  barWidth?: number
  tooltipFormatter?: (n: number) => string
  className?: string
}

export const BarSparkline = memo(function BarSparkline({
  data,
  color,
  height = 40,
  barGap = 1,
  barWidth = 6,
  tooltipFormatter = (n: number) => String(n),
  className
}: BarSparklineProps) {
  const [hoveredIdx, setHoveredIdx] = useState<number | null>(null)
  const max = Math.max(1, ...data.map(d => Math.abs(d)))
  const availableWidth = Math.max(0, (data.length - 1) * barGap + data.length * barWidth)

  const cell = useCallback(
    (idx: number) => {
      const v = data[idx] ?? 0
      const barHeight = max > 0 ? (Math.abs(v) / max) * height : 0
      const left = idx * (barWidth + barGap)
      return { left, barHeight, v }
    },
    [data, max, height, barWidth, barGap]
  )

  return (
    <div
      className={cn('relative w-full cursor-crosshair', className)}
      style={{ height: `${height + 16}px` }}
      onMouseLeave={() => setHoveredIdx(null)}
    >
      <svg
        width="100%"
        height={height}
        viewBox={`0 0 ${availableWidth} ${height}`}
        preserveAspectRatio="none"
        className="overflow-visible"
      >
        {data.map((v, i) => {
          const { left, barHeight } = cell(i)
          const isHover = hoveredIdx === i
          return (
            <g
              key={i}
              onMouseEnter={() => setHoveredIdx(i)}
              onMouseLeave={() => setHoveredIdx(null)}
              className="cursor-pointer"
            >
              <rect
                x={left}
                y={height - barHeight}
                width={barWidth}
                height={barHeight}
                fill={color}
                opacity={hoveredIdx !== null && !isHover ? 0.25 : 0.75}
              />
            </g>
          )
        })}
        {hoveredIdx !== null && (() => {
          const { left, barHeight, v } = cell(hoveredIdx)
          return (
            <g>
              <text
                x={left + barWidth / 2}
                y={height - barHeight - 3}
                textAnchor="middle"
                className="text-[9px] fill-(--ui-text-primary) font-mono pointer-events-none"
              >
                {tooltipFormatter(v)}
              </text>
            </g>
          )
        })()}
      </svg>
    </div>
  )
})
