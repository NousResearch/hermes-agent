import {
  flexRender,
  getCoreRowModel,
  getFilteredRowModel,
  getPaginationRowModel,
  getSortedRowModel,
  type SortingState,
  useReactTable
} from '@tanstack/react-table'
import { useMemo, useState } from 'react'
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  Cell,
  Line,
  LineChart,
  Legend,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis
} from 'recharts'

import type { CanvasBlock, CanvasChartSeries, CanvasDatum } from './types'

const FALLBACK_COLORS = ['#ff385f', '#ff6b84', '#ff96a8', '#ffc2cc', '#5c667a']

function blockTitle(title: string) {
  return <h2 className="text-sm font-semibold text-(--ui-text-primary)">{title}</h2>
}

function chartColor(item: CanvasDatum, index: number) {
  return item.color || FALLBACK_COLORS[index % FALLBACK_COLORS.length]
}

function seriesColor(series: CanvasChartSeries, index: number) {
  return series.color || FALLBACK_COLORS[index % FALLBACK_COLORS.length]
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
            {block.series.length > 1 ? <Legend /> : null}
            {block.series.map((series, index) => (
              <Bar
                dataKey={series.key}
                fill={seriesColor(series, index)}
                key={series.key}
                name={series.label}
                radius={block.stacked ? undefined : [6, 6, 2, 2]}
                stackId={block.stacked ? 'canvas-stack' : undefined}
              />
            ))}
          </BarChart>
        </ResponsiveContainer>
      </div>
    </article>
  )
}

function LineChartBlock({ block }: { block: Extract<CanvasBlock, { type: 'line-chart' }> }) {
  return (
    <article className="rounded-xl border border-(--ui-stroke-tertiary) bg-(--ui-chat-bubble-background) p-4">
      {blockTitle(block.title)}
      <div className="mt-4 h-64">
        <ResponsiveContainer height="100%" width="100%">
          <LineChart data={block.data} margin={{ top: 4, right: 8, left: -22, bottom: 0 }}>
            <XAxis axisLine={false} dataKey="label" fontSize={12} tickLine={false} />
            <YAxis axisLine={false} fontSize={12} tickLine={false} />
            <Tooltip />
            {block.series.length > 1 ? <Legend /> : null}
            {block.series.map((series, index) => (
              <Line
                dataKey={series.key}
                dot={false}
                key={series.key}
                name={series.label}
                stroke={seriesColor(series, index)}
                strokeWidth={2.5}
                type="monotone"
              />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </div>
    </article>
  )
}

function AreaChartBlock({ block }: { block: Extract<CanvasBlock, { type: 'area-chart' }> }) {
  return (
    <article className="rounded-xl border border-(--ui-stroke-tertiary) bg-(--ui-chat-bubble-background) p-4">
      {blockTitle(block.title)}
      <div className="mt-4 h-64">
        <ResponsiveContainer height="100%" width="100%">
          <AreaChart data={block.data} margin={{ top: 4, right: 8, left: -22, bottom: 0 }}>
            <defs>
              {block.series.map((series, index) => (
                <linearGradient id={`canvas-area-${block.id}-${series.key}`} key={series.key} x1="0" x2="0" y1="0" y2="1">
                  <stop offset="5%" stopColor={seriesColor(series, index)} stopOpacity={0.4} />
                  <stop offset="95%" stopColor={seriesColor(series, index)} stopOpacity={0.03} />
                </linearGradient>
              ))}
            </defs>
            <XAxis axisLine={false} dataKey="label" fontSize={12} tickLine={false} />
            <YAxis axisLine={false} fontSize={12} tickLine={false} />
            <Tooltip />
            {block.series.length > 1 ? <Legend /> : null}
            {block.series.map((series, index) => (
              <Area
                dataKey={series.key}
                fill={`url(#canvas-area-${block.id}-${series.key})`}
                key={series.key}
                name={series.label}
                stroke={seriesColor(series, index)}
                strokeWidth={2.5}
                type="monotone"
              />
            ))}
          </AreaChart>
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

function ListBlock({ block }: { block: Extract<CanvasBlock, { type: 'list' }> }) {
  const Component = block.ordered ? 'ol' : 'ul'

  return (
    <article className="rounded-xl border border-(--ui-stroke-tertiary) bg-(--ui-chat-bubble-background) p-4">
      {blockTitle(block.title)}
      <Component className={block.ordered ? 'mt-3 list-decimal space-y-2 pl-5' : 'mt-3 list-disc space-y-2 pl-5'}>
        {block.items.map((item, index) => (
          <li className="text-sm leading-6 text-(--ui-text-secondary)" key={`${index}-${item}`}>
            {item}
          </li>
        ))}
      </Component>
    </article>
  )
}

function ImageBlock({ block }: { block: Extract<CanvasBlock, { type: 'image' }> }) {
  return (
    <figure className="overflow-hidden rounded-xl border border-(--ui-stroke-tertiary) bg-(--ui-chat-bubble-background)">
      <img alt={block.alt} className="max-h-[36rem] w-full object-contain" src={block.src} />
      {block.caption ? (
        <figcaption className="border-t border-(--ui-stroke-tertiary) px-4 py-3 text-xs text-(--ui-text-tertiary)">
          {block.caption}
        </figcaption>
      ) : null}
    </figure>
  )
}

function TableBlock({ block }: { block: Extract<CanvasBlock, { type: 'table' }> }) {
  const [query, setQuery] = useState('')
  const [sorting, setSorting] = useState<SortingState>([])
  const columns = useMemo(
    () =>
      block.columns.map((column, index) => ({
        id: `column-${index}`,
        accessorFn: (row: string[]) => row[index] || '',
        header: column
      })),
    [block.columns]
  )
  const table = useReactTable({
    columns,
    data: block.rows,
    getCoreRowModel: getCoreRowModel(),
    getFilteredRowModel: getFilteredRowModel(),
    getPaginationRowModel: getPaginationRowModel(),
    getSortedRowModel: getSortedRowModel(),
    globalFilterFn: 'includesString',
    initialState: { pagination: { pageIndex: 0, pageSize: 25 } },
    onGlobalFilterChange: setQuery,
    onSortingChange: setSorting,
    state: { globalFilter: query, sorting }
  })
  const filteredRows = table.getFilteredRowModel().rows.length

  return (
    <article className="overflow-hidden rounded-xl border border-(--ui-stroke-tertiary) bg-(--ui-chat-bubble-background)">
      <div className="flex flex-wrap items-center justify-between gap-3 px-4 py-4">
        <div>
          {blockTitle(block.title)}
          <div className="mt-1 text-[0.6875rem] text-(--ui-text-tertiary)">
            {filteredRows.toLocaleString()} of {block.rows.length.toLocaleString()} rows
          </div>
        </div>
        <label className="relative min-w-48 flex-1 sm:max-w-72">
          <span className="sr-only">Search {block.title}</span>
          <input
            className="h-8 w-full rounded-md border border-(--ui-stroke-tertiary) bg-(--ui-chat-surface-background) px-3 text-xs text-(--ui-text-primary) outline-none placeholder:text-(--ui-text-quaternary) focus:border-(--ui-focus-ring)"
            onChange={event => setQuery(event.target.value)}
            placeholder="Search rows…"
            type="search"
            value={query}
          />
        </label>
      </div>
      <div className="overflow-x-auto">
        <table className="w-full min-w-[680px] border-collapse text-left text-xs">
          <thead className="border-y border-(--ui-stroke-tertiary) bg-(--ui-chat-surface-background) text-(--ui-text-tertiary)">
            {table.getHeaderGroups().map(headerGroup => (
              <tr key={headerGroup.id}>
                {headerGroup.headers.map(header => (
                  <th className="whitespace-nowrap px-4 py-3 font-medium" key={header.id}>
                    <button
                      className="flex items-center gap-1 hover:text-(--ui-text-primary) disabled:cursor-default"
                      disabled={!header.column.getCanSort()}
                      onClick={header.column.getToggleSortingHandler()}
                      type="button"
                    >
                      {flexRender(header.column.columnDef.header, header.getContext())}
                      <span aria-hidden className="w-3 text-[0.625rem]">
                        {header.column.getIsSorted() === 'asc'
                          ? '▲'
                          : header.column.getIsSorted() === 'desc'
                            ? '▼'
                            : ''}
                      </span>
                    </button>
                  </th>
                ))}
              </tr>
            ))}
          </thead>
          <tbody className="text-(--ui-text-secondary)">
            {table.getRowModel().rows.map(row => (
              <tr className="border-b border-(--ui-stroke-tertiary) last:border-0" key={row.id}>
                {row.getVisibleCells().map((cell, index) => (
                  <td
                    className={
                      index === 0 ? 'max-w-80 px-4 py-3 font-medium text-(--ui-text-primary)' : 'max-w-80 px-4 py-3'
                    }
                    key={cell.id}
                  >
                    <span className="line-clamp-3 break-words">{String(cell.getValue() ?? '')}</span>
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="flex flex-wrap items-center justify-between gap-3 border-t border-(--ui-stroke-tertiary) px-4 py-3 text-xs text-(--ui-text-tertiary)">
        <label className="flex items-center gap-2">
          Rows
          <select
            className="h-7 rounded border border-(--ui-stroke-tertiary) bg-(--ui-chat-surface-background) px-2 text-(--ui-text-secondary)"
            onChange={event => table.setPageSize(Number(event.target.value))}
            value={table.getState().pagination.pageSize}
          >
            {[25, 50, 100].map(size => (
              <option key={size} value={size}>
                {size}
              </option>
            ))}
          </select>
        </label>
        <div className="flex items-center gap-2">
          <span>
            Page {table.getState().pagination.pageIndex + 1} of {Math.max(table.getPageCount(), 1)}
          </span>
          <button
            className="rounded px-2 py-1 hover:bg-(--ui-control-hover-background) disabled:opacity-40"
            disabled={!table.getCanPreviousPage()}
            onClick={() => table.previousPage()}
            type="button"
          >
            Previous
          </button>
          <button
            className="rounded px-2 py-1 hover:bg-(--ui-control-hover-background) disabled:opacity-40"
            disabled={!table.getCanNextPage()}
            onClick={() => table.nextPage()}
            type="button"
          >
            Next
          </button>
        </div>
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
        if (block.type === 'line-chart') return <LineChartBlock block={block} key={block.id} />
        if (block.type === 'area-chart') return <AreaChartBlock block={block} key={block.id} />
        if (block.type === 'pie-chart') return <PieChartBlock block={block} key={block.id} />
        if (block.type === 'insight') return <InsightBlock block={block} key={block.id} />
        if (block.type === 'list') return <ListBlock block={block} key={block.id} />
        if (block.type === 'image') return <ImageBlock block={block} key={block.id} />
        if (block.type === 'divider') return <hr className="border-(--ui-stroke-tertiary)" key={block.id} />
        return <TableBlock block={block} key={block.id} />
      })}
    </div>
  )
}
