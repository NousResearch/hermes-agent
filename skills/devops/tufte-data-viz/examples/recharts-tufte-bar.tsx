/**
 * Tufte-styled horizontal bar chart using Recharts.
 *
 * Demonstrates: horizontal layout, sorted by value, direct value labels,
 * hidden X-axis (values are on bars), no legend, no gridlines.
 */

import React from 'react';
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  LabelList,
} from 'recharts';

const TUFTE = {
  bg: '#fffff8',
  text: '#111',
  textSecondary: '#666',
  textTertiary: '#999',
  seriesDefault: '#7a7a7a',
  highlight: '#e41a1c',
  font: '"ET Book", "Palatino Linotype", Palatino, "Book Antiqua", Georgia, serif',
};

const data = [
  { name: 'Product A', value: 42000 },
  { name: 'Product B', value: 38000 },
  { name: 'Product C', value: 27000 },
  { name: 'Product D', value: 19000 },
  { name: 'Product E', value: 12000 },
].sort((a, b) => b.value - a.value);

export function ProductChart() {
  return (
    <div style={{ background: TUFTE.bg, padding: '24px 16px 16px' }}>
      <h3
        style={{
          fontFamily: TUFTE.font,
          fontWeight: 400,
          fontSize: 18,
          color: TUFTE.text,
          margin: '0 0 16px 0',
        }}
      >
        Revenue by Product
      </h3>

      <ResponsiveContainer width="100%" height={data.length * 48 + 32}>
        <BarChart
          data={data}
          layout="vertical"
          margin={{ top: 0, right: 80, bottom: 0, left: 0 }}
        >
          {/* Hidden — values are directly on bars */}
          <XAxis type="number" hide />

          <YAxis
            type="category"
            dataKey="name"
            width={90}
            tickLine={false}
            axisLine={false}
            tick={{
              fontSize: 13,
              fill: TUFTE.text,
              fontFamily: TUFTE.font,
            }}
          />

          <Bar
            dataKey="value"
            fill={TUFTE.seriesDefault}
            barSize={20}
            radius={[0, 2, 2, 0]}
          >
            <LabelList
              dataKey="value"
              position="right"
              fill={TUFTE.textSecondary}
              fontSize={13}
              fontFamily={TUFTE.font}
              formatter={(v: number) => `$${(v / 1000).toFixed(0)}k`}
            />
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
