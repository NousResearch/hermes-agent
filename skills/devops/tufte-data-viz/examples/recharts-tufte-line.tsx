/**
 * Tufte-styled line chart using Recharts.
 *
 * Demonstrates: no gridlines, no legend, direct labels, range-frame Y-axis,
 * off-white background, serif fonts, annotation of notable data point.
 */

import React from 'react';
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceLine,
  ReferenceDot,
} from 'recharts';

const TUFTE = {
  bg: '#fffff8',
  text: '#111',
  textSecondary: '#666',
  textTertiary: '#999',
  axis: '#ccc',
  seriesDefault: '#666',
  highlight: '#e41a1c',
  font: '"ET Book", "Palatino Linotype", Palatino, "Book Antiqua", Georgia, serif',
  fontSans: 'system-ui, -apple-system, sans-serif',
};

const data = [
  { month: 'Jan', revenue: 4200, target: 4000 },
  { month: 'Feb', revenue: 4800, target: 4200 },
  { month: 'Mar', revenue: 5100, target: 4400 },
  { month: 'Apr', revenue: 4900, target: 4600 },
  { month: 'May', revenue: 5600, target: 4800 },
  { month: 'Jun', revenue: 6200, target: 5000 },
];

const TufteTooltip = ({ active, payload, label }: any) => {
  if (!active || !payload?.length) return null;
  return (
    <div
      style={{
        fontFamily: TUFTE.font,
        fontSize: 13,
        color: TUFTE.text,
        background: 'none',
        border: 'none',
        padding: 0,
      }}
    >
      <div style={{ color: TUFTE.textSecondary, fontSize: 11 }}>{label}</div>
      {payload.map((p: any) => (
        <div key={p.dataKey}>
          {p.name}: <strong>${p.value.toLocaleString()}</strong>
        </div>
      ))}
    </div>
  );
};

export function RevenueChart() {
  const peakIdx = data.reduce(
    (maxI, d, i) => (d.revenue > data[maxI].revenue ? i : maxI),
    0,
  );

  return (
    <div style={{ background: TUFTE.bg, padding: '24px 16px 16px' }}>
      <h3
        style={{
          fontFamily: TUFTE.font,
          fontWeight: 400,
          fontSize: 18,
          color: TUFTE.text,
          margin: '0 0 4px 48px',
        }}
      >
        Monthly Revenue vs. Target
      </h3>
      <p
        style={{
          fontFamily: TUFTE.font,
          fontSize: 13,
          color: TUFTE.textSecondary,
          margin: '0 0 16px 48px',
        }}
      >
        Revenue exceeded target every month, accelerating in Q2
      </p>

      <ResponsiveContainer width="100%" aspect={1.5}>
        <LineChart
          data={data}
          margin={{ top: 8, right: 100, bottom: 8, left: 8 }}
        >
          <CartesianGrid stroke="none" />

          <XAxis
            dataKey="month"
            tickLine={false}
            axisLine={{ stroke: TUFTE.axis, strokeWidth: 0.5 }}
            tick={{
              fontSize: 12,
              fill: TUFTE.textTertiary,
              fontFamily: TUFTE.fontSans,
            }}
          />

          <YAxis
            tickLine={false}
            axisLine={false}
            tick={{
              fontSize: 12,
              fill: TUFTE.textTertiary,
              fontFamily: TUFTE.fontSans,
            }}
            domain={['dataMin - 200', 'dataMax + 400']}
          />

          <Tooltip
            content={<TufteTooltip />}
            cursor={{ stroke: TUFTE.axis, strokeWidth: 0.5 }}
          />

          {/* Target line: gray dashed */}
          <Line
            type="monotone"
            dataKey="target"
            stroke={TUFTE.seriesDefault}
            strokeWidth={1}
            strokeDasharray="4 3"
            dot={false}
            name="Target"
          />

          {/* Revenue line: highlighted */}
          <Line
            type="monotone"
            dataKey="revenue"
            stroke={TUFTE.highlight}
            strokeWidth={2}
            dot={false}
            name="Revenue"
          />

          {/* Direct labels at right edge (no Legend component) */}
          <ReferenceLine
            y={data[data.length - 1].revenue}
            stroke="none"
            label={{
              value: `Revenue: $${data[data.length - 1].revenue.toLocaleString()}`,
              position: 'right',
              fill: TUFTE.highlight,
              fontSize: 13,
              fontFamily: TUFTE.font,
            }}
          />
          <ReferenceLine
            y={data[data.length - 1].target}
            stroke="none"
            label={{
              value: `Target: $${data[data.length - 1].target.toLocaleString()}`,
              position: 'right',
              fill: TUFTE.textSecondary,
              fontSize: 13,
              fontFamily: TUFTE.font,
            }}
          />

          {/* Annotate the peak */}
          <ReferenceDot
            x={data[peakIdx].month}
            y={data[peakIdx].revenue}
            r={3}
            fill={TUFTE.highlight}
            stroke="none"
            label={{
              value: `Peak: $${data[peakIdx].revenue.toLocaleString()}`,
              position: 'top',
              fill: TUFTE.text,
              fontSize: 12,
              fontFamily: TUFTE.font,
              offset: 12,
            }}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
