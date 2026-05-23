# ECharts: Tufte Configuration

Load this file when generating any ECharts chart code.

## Tufte theme registration

Register once at app initialization:

```typescript
import * as echarts from 'echarts';

const tufteTheme = {
  color: ['#666666', '#4e79a7', '#f28e2b', '#e15759', '#76b7b2'],
  backgroundColor: '#fffff8',
  textStyle: {
    fontFamily: '"ET Book", "Palatino Linotype", Palatino, Georgia, serif',
    color: '#111',
  },
  title: {
    textStyle: {
      fontSize: 18,
      fontWeight: 'normal',
      color: '#111',
      fontFamily: '"ET Book", "Palatino Linotype", Palatino, Georgia, serif',
    },
    subtextStyle: {
      fontSize: 13,
      color: '#666',
      fontFamily: '"ET Book", "Palatino Linotype", Palatino, Georgia, serif',
    },
  },
  line: {
    itemStyle: { borderWidth: 0 },
    lineStyle: { width: 1.5 },
    symbolSize: 0,
    symbol: 'circle',
    smooth: false,
  },
  bar: {
    itemStyle: {
      barBorderWidth: 0,
      barBorderColor: '#ccc',
    },
  },
  categoryAxis: {
    axisLine: { show: true, lineStyle: { color: '#ccc', width: 0.5 } },
    axisTick: { show: false },
    axisLabel: {
      color: '#999',
      fontSize: 11,
      fontFamily: 'system-ui, -apple-system, sans-serif',
    },
    splitLine: { show: false },
  },
  valueAxis: {
    axisLine: { show: false },
    axisTick: { show: false },
    axisLabel: {
      color: '#999',
      fontSize: 11,
      fontFamily: 'system-ui, -apple-system, sans-serif',
    },
    splitLine: { show: false },
  },
  legend: { show: false },
  tooltip: {
    backgroundColor: 'rgba(255,255,248,0.95)',
    borderWidth: 0,
    textStyle: {
      fontSize: 12,
      color: '#333',
      fontFamily: 'system-ui, sans-serif',
    },
    extraCssText: 'box-shadow: none;',
  },
  grid: {
    show: false,
    left: 60,
    right: 80,
    top: 60,
    bottom: 40,
    containLabel: false,
  },
};

echarts.registerTheme('tufte', tufteTheme);
```

Usage: `echarts.init(dom, 'tufte')`.

## Base option object

For inline use without theme registration:

```typescript
const tufteBaseOption: echarts.EChartsOption = {
  backgroundColor: '#fffff8',
  textStyle: {
    fontFamily: '"ET Book", "Palatino Linotype", Palatino, Georgia, serif',
    color: '#111',
  },
  grid: { show: false, left: 60, right: 80, top: 60, bottom: 40 },
  legend: { show: false },
  tooltip: {
    trigger: 'axis',
    backgroundColor: 'rgba(255,255,248,0.95)',
    borderWidth: 0,
    textStyle: { fontSize: 12, color: '#333' },
    extraCssText: 'box-shadow: none;',
  },
  xAxis: {
    type: 'category',
    axisLine: { lineStyle: { color: '#ccc', width: 0.5 } },
    axisTick: { show: false },
    axisLabel: { color: '#999', fontSize: 11, fontFamily: 'system-ui, sans-serif' },
    splitLine: { show: false },
  },
  yAxis: {
    type: 'value',
    axisLine: { show: false },
    axisTick: { show: false },
    axisLabel: { color: '#999', fontSize: 11, fontFamily: 'system-ui, sans-serif' },
    splitLine: { show: false },
  },
};
```

## Direct labeling

Label series endpoints instead of using legend:

```typescript
series: [
  {
    name: 'Revenue',
    type: 'line',
    data: revenueData,
    lineStyle: { color: '#e41a1c', width: 2 },
    itemStyle: { color: '#e41a1c' },
    showSymbol: false,
    endLabel: {
      show: true,
      formatter: '{a}: {c}',
      fontSize: 13,
      fontFamily: '"ET Book", Palatino, Georgia, serif',
      color: '#e41a1c',
    },
  },
  {
    name: 'Target',
    type: 'line',
    data: targetData,
    lineStyle: { color: '#999', width: 1, type: 'dashed' },
    itemStyle: { color: '#999' },
    showSymbol: false,
    endLabel: {
      show: true,
      formatter: '{a}: {c}',
      fontSize: 13,
      fontFamily: '"ET Book", Palatino, Georgia, serif',
      color: '#999',
    },
  },
]
```

## Annotations via markPoint / markLine

```typescript
series: [{
  // ... base config
  markPoint: {
    data: [
      { type: 'max', name: 'Peak' },
    ],
    symbol: 'circle',
    symbolSize: 6,
    label: {
      show: true,
      position: 'top',
      formatter: 'Peak: {c}',
      fontSize: 12,
      fontStyle: 'italic',
      color: '#333',
      fontFamily: '"ET Book", Palatino, Georgia, serif',
    },
    itemStyle: { color: '#e41a1c' },
  },
  markLine: {
    data: [
      { type: 'average', name: 'Avg' },
    ],
    lineStyle: { color: '#ccc', width: 0.5, type: 'dashed' },
    label: {
      position: 'end',
      formatter: 'Avg: {c}',
      fontSize: 11,
      color: '#999',
    },
  },
}]
```

## Range-frame via axis min/max

```typescript
// Compute from data
const values = data.flat();
const dataMin = Math.min(...values);
const dataMax = Math.max(...values);
const padding = (dataMax - dataMin) * 0.05;

yAxis: {
  min: dataMin - padding,
  max: dataMax + padding,
}
```

## Complete example: line chart

```typescript
const option: echarts.EChartsOption = {
  ...tufteBaseOption,
  title: {
    text: 'Monthly Revenue vs. Target',
    subtext: 'Revenue exceeded target every month, accelerating in Q2',
    left: 'left',
  },
  xAxis: {
    ...tufteBaseOption.xAxis,
    data: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'],
  },
  yAxis: {
    ...tufteBaseOption.yAxis,
    min: 3800,
    max: 6500,
  },
  series: [
    {
      name: 'Target',
      type: 'line',
      data: [4000, 4200, 4400, 4600, 4800, 5000],
      lineStyle: { color: '#999', width: 1, type: 'dashed' },
      showSymbol: false,
      endLabel: {
        show: true,
        formatter: 'Target: {c}',
        color: '#999',
        fontSize: 13,
        fontFamily: '"ET Book", Palatino, Georgia, serif',
      },
    },
    {
      name: 'Revenue',
      type: 'line',
      data: [4200, 4800, 5100, 4900, 5600, 6200],
      lineStyle: { color: '#e41a1c', width: 2 },
      showSymbol: false,
      endLabel: {
        show: true,
        formatter: 'Revenue: {c}',
        color: '#e41a1c',
        fontSize: 13,
        fontFamily: '"ET Book", Palatino, Georgia, serif',
      },
      markPoint: {
        data: [{ type: 'max', name: 'Peak' }],
        symbol: 'circle',
        symbolSize: 6,
        label: {
          show: true,
          position: 'top',
          formatter: 'Peak: {c}',
          fontSize: 12,
          fontStyle: 'italic',
          color: '#333',
        },
        itemStyle: { color: '#e41a1c' },
      },
    },
  ],
};
```

## Complete example: horizontal bar chart

```typescript
const barOption: echarts.EChartsOption = {
  ...tufteBaseOption,
  title: { text: 'Revenue by Product', left: 'left' },
  grid: { show: false, left: 100, right: 80, top: 50, bottom: 20 },
  xAxis: { type: 'value', show: false },
  yAxis: {
    type: 'category',
    data: ['Product E', 'Product D', 'Product C', 'Product B', 'Product A'],
    axisLine: { show: false },
    axisTick: { show: false },
    axisLabel: { color: '#111', fontSize: 13, fontFamily: '"ET Book", Palatino, Georgia, serif' },
  },
  series: [{
    type: 'bar',
    data: [12000, 19000, 27000, 38000, 42000],
    itemStyle: { color: '#666' },
    barWidth: 20,
    label: {
      show: true,
      position: 'right',
      formatter: (params: any) => `$${(params.value / 1000).toFixed(0)}k`,
      fontSize: 13,
      color: '#666',
      fontFamily: '"ET Book", Palatino, Georgia, serif',
    },
  }],
};
```
