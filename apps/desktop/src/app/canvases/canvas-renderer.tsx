import { Bar, BarChart, Cell, Pie, PieChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'

import type { CanvasBlock, CanvasDatum } from './types'

const FALLBACK_COLORS = ['#ff385f', '#ff6b84', '#ff96a8', '#ffc2cc', '#5c667a']

function blockTitle(title: string) {
  return <h2 className="text-sm font-semibold text-(--ui-text-primary)">{title}</h2>
}

function chartColor(item: CanvasDatum, index: number) {
  return item.color || FALLBACK_COLORS[index % FALLBACK_COLORS.length]
}

function KpisBlock({ block }: { block: Extract<CanvasBlock, { type: 'kpis' }> }) {
  return (
    <div className="grid gap-3 sm:grid-cols-3">
      {block.items.map(item => (
        <article
          className="rounded-xl border border-(--ui-stroke-tertiary) bg-(--ui-chat-bubble-background) p-4"
          key={item.label}
        >
          <div className="text-xs text-(--ui-text-tertiary)">{item.label}</div>
          <div className="mt-2 text-2xl font-semibold tracking-tight text-(--ui-text-primary)">{item.value}</div>
          {item.change ? <div className="mt-2 text-xs font-medium text-[#e72b4f]">{item.change}</div> : null}
        </article>
      ))}
    </div>
  )
}

function BarChartBlock({ block }: { block: Extract<CanvasBlock, { type: 'bar-chart' }> }) {
  return (
    <article className="rounded-xl border border-(--ui-stroke-tertiary) bg-(--ui-chat-bubble-background) p-4">
      {blockTitle(block.title)}
      <div className="mt-4 h-64">
        <ResponsiveContainer height="100%" width="100%">
          <BarChart data={block.data} margin={{ top: 4, right: 8, left: -22, bottom: 0 }}>
            <XAxis axisLine={false} dataKey="label" fontSize={12} tickLine={false} />
            <YAxis axisLine={false} fontSize={12} tickLine={false} />
            <Tooltip cursor={{ fill: 'rgba(255, 56, 95, 0.08)' }} />
            <Bar dataKey="value" radius={[6, 6, 2, 2]}>
              {block.data.map((item, index) => (
                <Cell fill={chartColor(item, index)} key={item.label} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </article>
  )
}

function PieChartBlock({ block }: { block: Extract<CanvasBlock, { type: 'pie-chart' }> }) {
  return (
    <article className="rounded-xl border border-(--ui-stroke-tertiary) bg-(--ui-chat-bubble-background) p-4">
      {blockTitle(block.title)}
      <div className="mt-2 grid items-center gap-2 sm:grid-cols-[1fr_auto]">
        <div className="h-56 min-w-0">
          <ResponsiveContainer height="100%" width="100%">
            <PieChart>
              <Tooltip />
              <Pie data={block.data} dataKey="value" innerRadius="58%" outerRadius="82%" paddingAngle={3}>
                {block.data.map((item, index) => (
                  <Cell fill={chartColor(item, index)} key={item.label} />
                ))}
              </Pie>
            </PieChart>
          </ResponsiveContainer>
        </div>
        <ul className="space-y-2 text-xs text-(--ui-text-secondary)">
          {block.data.map((item, index) => (
            <li className="flex items-center justify-between gap-5" key={item.label}>
              <span className="flex items-center gap-2">
                <i className="h-2 w-2 rounded-full" style={{ backgroundColor: chartColor(item, index) }} />
                {item.label}
              </span>
              <strong className="text-(--ui-text-primary)">{item.value}</strong>
            </li>
          ))}
        </ul>
      </div>
    </article>
  )
}

function InsightBlock({ block }: { block: Extract<CanvasBlock, { type: 'insight' }> }) {
  return (
    <article className="rounded-xl border border-[#ffc2cc] bg-[#fff4f6] p-4 dark:bg-[#411823]">
      {blockTitle(block.title)}
      <p className="mt-2 text-sm leading-6 text-(--ui-text-secondary)">{block.body}</p>
    </article>
  )
}

function TableBlock({ block }: { block: Extract<CanvasBlock, { type: 'table' }> }) {
  return (
    <article className="overflow-hidden rounded-xl border border-(--ui-stroke-tertiary) bg-(--ui-chat-bubble-background)">
      <div className="px-4 py-4">{blockTitle(block.title)}</div>
      <div className="overflow-x-auto">
        <table className="w-full min-w-[680px] border-collapse text-left text-xs">
          <thead className="border-y border-(--ui-stroke-tertiary) bg-(--ui-chat-surface-background) text-(--ui-text-tertiary)">
            <tr>
              {block.columns.map(column => (
                <th className="px-4 py-3 font-medium" key={column}>
                  {column}
                </th>
              ))}
            </tr>
          </thead>
          <tbody className="text-(--ui-text-secondary)">
            {block.rows.map(row => (
              <tr className="border-b border-(--ui-stroke-tertiary) last:border-0" key={row[0]}>
                {row.map((cell, index) => (
                  <td
                    className={index === 0 ? 'px-4 py-3 font-medium text-(--ui-text-primary)' : 'px-4 py-3'}
                    key={`${row[0]}-${index}`}
                  >
                    {cell}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </article>
  )
}

export function CanvasRenderer({ blocks }: { blocks: CanvasBlock[] }) {
  return (
    <div className="space-y-4">
      {blocks.map(block => {
        if (block.type === 'kpis') return <KpisBlock block={block} key={block.type} />
        if (block.type === 'bar-chart') return <BarChartBlock block={block} key={block.id} />
        if (block.type === 'pie-chart') return <PieChartBlock block={block} key={block.id} />
        if (block.type === 'insight') return <InsightBlock block={block} key={block.id} />
        return <TableBlock block={block} key={block.id} />
      })}
    </div>
  )
}
