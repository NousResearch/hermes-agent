# Recharts: Tufte Configuration

Load this file when generating any Recharts (React) chart code.

## Tufte constants

```tsx
const TUFTE = {
  bg: '#fffff8',
  text: '#111',
  textSecondary: '#666',
  textTertiary: '#999',
  axis: '#ccc',
  grid: '#eee',
  seriesDefault: '#666',
  highlight: '#e41a1c',
  categorical: ['#4e79a7', '#f28e2b', '#e15759', '#76b7b2'],
  font: '"ET Book", "Palatino Linotype", Palatino, "Book Antiqua", Georgia, serif',
  fontSans: 'system-ui, -apple-system, sans-serif',
};
```

## Component-by-component configuration

### CartesianGrid

Always remove or make invisible:

```tsx
<CartesianGrid stroke="none" />
```

If gridlines are truly needed for reading precision:

```tsx
<CartesianGrid
  vertical={false}
  stroke={TUFTE.grid}
  strokeOpacity={0.1}
  strokeWidth={0.5}
/>
```

### XAxis

```tsx
<XAxis
  dataKey="name"
  tickLine={false}
  axisLine={{ stroke: TUFTE.axis, strokeWidth: 0.5 }}
  tick={{ fontSize: 12, fill: TUFTE.textTertiary, fontFamily: TUFTE.fontSans }}
  dy={8}
/>
```

### YAxis

```tsx
<YAxis
  tickLine={false}
  axisLine={false}
  tick={{ fontSize: 12, fill: TUFTE.textTertiary, fontFamily: TUFTE.fontSans }}
  width={45}
  dx={-4}
/>
```

Note: `axisLine={false}` on YAxis. The left axis line is optional — tick labels alone provide sufficient reference. Include it only if the chart feels ungrounded without it.

### Tooltip — custom minimal component

```tsx
const TufteTooltip = ({ active, payload, label }: any) => {
  if (!active || !payload?.length) return null;
  return (
    <div style={{
      fontFamily: TUFTE.font,
      fontSize: 13,
      color: TUFTE.text,
      background: 'none',
      border: 'none',
      padding: 0,
    }}>
      <div style={{ color: TUFTE.textSecondary, fontSize: 11 }}>{label}</div>
      {payload.map((p: any) => (
        <div key={p.dataKey}>
          {p.name}: <strong>{p.value.toLocaleString()}</strong>
        </div>
      ))}
    </div>
  );
};

// Usage:
<Tooltip content={<TufteTooltip />} cursor={{ stroke: TUFTE.axis, strokeWidth: 0.5 }} />
```

### Removing Legend — direct label pattern

Never use `<Legend />`. Instead, label each line at its last data point:

```tsx
const DirectLabel = ({ data, dataKey, stroke, name }: {
  data: any[]; dataKey: string; stroke: string; name: string;
}) => {
  const last = data[data.length - 1];
  if (!last) return null;
  return (
    <text
      x="97%"
      y={0} // Will be positioned by ReferenceDot
      fill={stroke}
      fontSize={13}
      fontFamily={TUFTE.font}
      textAnchor="start"
    >
      {name}
    </text>
  );
};
```

A simpler approach using `<ReferenceLine>`:

```tsx
<ReferenceLine
  y={data[data.length - 1]?.revenue}
  stroke="none"
  label={{
    value: 'Revenue',
    position: 'right',
    fill: TUFTE.text,
    fontSize: 13,
    fontFamily: TUFTE.font,
  }}
/>
```

### Line

```tsx
<Line
  type="monotone"
  dataKey="value"
  stroke={TUFTE.seriesDefault}
  strokeWidth={1.5}
  dot={false}
  activeDot={{ r: 3, fill: TUFTE.text, stroke: 'none' }}
/>
```

For highlighted series:

```tsx
<Line
  type="monotone"
  dataKey="highlight"
  stroke={TUFTE.highlight}
  strokeWidth={2}
  dot={false}
/>
```

### Bar

```tsx
<Bar dataKey="value" fill={TUFTE.seriesDefault} radius={[2, 2, 0, 0]}>
  <LabelList
    dataKey="value"
    position="top"
    fill={TUFTE.textSecondary}
    fontSize={12}
    fontFamily={TUFTE.font}
    formatter={(v: number) => v.toLocaleString()}
  />
</Bar>
```

### Area (use sparingly)

If area fill is needed, keep opacity extremely low:

```tsx
<Area
  type="monotone"
  dataKey="value"
  stroke={TUFTE.seriesDefault}
  strokeWidth={1.5}
  fill={TUFTE.seriesDefault}
  fillOpacity={0.05}
/>
```

## Range-frame custom tick

To constrain axis ticks to the actual data range:

```tsx
const RangeFrameTick = ({
  x, y, payload, data, dataKey, orientation,
}: any) => {
  const values = data.map((d: any) => d[dataKey]);
  const min = Math.min(...values);
  const max = Math.max(...values);
  if (payload.value < min || payload.value > max) return null;
  return (
    <text
      x={x}
      y={y}
      dy={orientation === 'bottom' ? 16 : 0}
      dx={orientation === 'left' ? -8 : 0}
      textAnchor={orientation === 'left' ? 'end' : 'middle'}
      fill={TUFTE.textTertiary}
      fontSize={12}
      fontFamily={TUFTE.fontSans}
    >
      {payload.value}
    </text>
  );
};
```

## Complete example: line chart

```tsx
import {
  ResponsiveContainer, LineChart, Line, XAxis, YAxis,
  CartesianGrid, Tooltip, ReferenceLine,
} from 'recharts';

const data = [
  { month: 'Jan', revenue: 4200, target: 4000 },
  { month: 'Feb', revenue: 4800, target: 4200 },
  { month: 'Mar', revenue: 5100, target: 4400 },
  { month: 'Apr', revenue: 4900, target: 4600 },
  { month: 'May', revenue: 5600, target: 4800 },
  { month: 'Jun', revenue: 6200, target: 5000 },
];

export function RevenueChart() {
  return (
    <div style={{ background: TUFTE.bg, padding: '24px 16px 16px' }}>
      <h3 style={{
        fontFamily: TUFTE.font, fontWeight: 400, fontSize: 18,
        color: TUFTE.text, margin: '0 0 4px 48px',
      }}>
        Monthly Revenue vs. Target
      </h3>
      <p style={{
        fontFamily: TUFTE.font, fontSize: 13, color: TUFTE.textSecondary,
        margin: '0 0 16px 48px',
      }}>
        Revenue exceeded target every month, accelerating in Q2
      </p>
      <ResponsiveContainer width="100%" height={400} aspect={1.5}>
        <LineChart data={data} margin={{ top: 8, right: 80, bottom: 8, left: 8 }}>
          <CartesianGrid stroke="none" />
          <XAxis
            dataKey="month"
            tickLine={false}
            axisLine={{ stroke: TUFTE.axis, strokeWidth: 0.5 }}
            tick={{ fontSize: 12, fill: TUFTE.textTertiary, fontFamily: TUFTE.fontSans }}
          />
          <YAxis
            tickLine={false}
            axisLine={false}
            tick={{ fontSize: 12, fill: TUFTE.textTertiary, fontFamily: TUFTE.fontSans }}
            domain={['dataMin - 200', 'dataMax + 200']}
          />
          <Tooltip content={<TufteTooltip />} />
          <Line
            type="monotone" dataKey="target" stroke={TUFTE.seriesDefault}
            strokeWidth={1} strokeDasharray="4 3" dot={false}
          />
          <Line
            type="monotone" dataKey="revenue" stroke={TUFTE.highlight}
            strokeWidth={2} dot={false}
          />
          {/* Direct labels at right edge */}
          <ReferenceLine y={data[data.length - 1].revenue} stroke="none" label={{
            value: `Revenue: ${data[data.length - 1].revenue.toLocaleString()}`,
            position: 'right', fill: TUFTE.highlight, fontSize: 13, fontFamily: TUFTE.font,
          }} />
          <ReferenceLine y={data[data.length - 1].target} stroke="none" label={{
            value: `Target: ${data[data.length - 1].target.toLocaleString()}`,
            position: 'right', fill: TUFTE.textSecondary, fontSize: 13, fontFamily: TUFTE.font,
          }} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
```

## Complete example: horizontal bar chart

```tsx
import { ResponsiveContainer, BarChart, Bar, XAxis, YAxis, LabelList } from 'recharts';

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
      <h3 style={{
        fontFamily: TUFTE.font, fontWeight: 400, fontSize: 18,
        color: TUFTE.text, margin: '0 0 16px 0',
      }}>
        Revenue by Product
      </h3>
      <ResponsiveContainer width="100%" height={data.length * 48 + 32}>
        <BarChart data={data} layout="vertical" margin={{ top: 0, right: 80, bottom: 0, left: 0 }}>
          <XAxis type="number" hide />
          <YAxis
            type="category" dataKey="name" width={90}
            tickLine={false} axisLine={false}
            tick={{ fontSize: 13, fill: TUFTE.text, fontFamily: TUFTE.font }}
          />
          <Bar dataKey="value" fill={TUFTE.seriesDefault} barSize={20} radius={[0, 2, 2, 0]}>
            <LabelList
              dataKey="value" position="right"
              fill={TUFTE.textSecondary} fontSize={13} fontFamily={TUFTE.font}
              formatter={(v: number) => `$${(v / 1000).toFixed(0)}k`}
            />
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
```

## Small multiples pattern

```tsx
const metrics = ['revenue', 'users', 'conversion'];

<div style={{
  display: 'grid',
  gridTemplateColumns: `repeat(${metrics.length}, 1fr)`,
  gap: 16,
  background: TUFTE.bg,
  padding: 16,
}}>
  {metrics.map((metric) => (
    <div key={metric}>
      <h4 style={{ fontFamily: TUFTE.font, fontWeight: 400, fontSize: 14, color: TUFTE.text }}>
        {metric}
      </h4>
      <ResponsiveContainer width="100%" height={200}>
        <LineChart data={data}>
          <CartesianGrid stroke="none" />
          <XAxis dataKey="month" tickLine={false} axisLine={false} tick={{ fontSize: 10 }} />
          <YAxis tickLine={false} axisLine={false} tick={{ fontSize: 10 }} width={35} />
          <Line dataKey={metric} stroke={TUFTE.seriesDefault} strokeWidth={1.5} dot={false} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  ))}
</div>
```
