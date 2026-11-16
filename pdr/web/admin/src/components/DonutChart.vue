<script setup lang="ts">
import { computed } from 'vue'
import type { Slice } from '../api'

const props = defineProps<{ slices: Slice[]; caption: string }>()

const total = computed(() => props.slices.reduce((n, s) => n + s.qty, 0))
const ring = computed(() => {
  if (!total.value) return '#eeedf7'
  let acc = 0
  const stops: string[] = []
  for (const s of props.slices) {
    const a = (acc / total.value) * 100
    const b = ((acc + s.qty) / total.value) * 100
    stops.push(`${s.color} ${(a + 0.6).toFixed(2)}% ${(b - 0.6).toFixed(2)}%`, `#fff ${(b - 0.6).toFixed(2)}% ${(b + 0.6).toFixed(2)}%`)
    acc += s.qty
  }
  return `conic-gradient(${stops.join(',')})`
})
const pct = (q: number) => (total.value ? Math.round((q / total.value) * 100) : 0)
</script>

<template>
  <div class="donut-wrap">
    <div class="donut" :style="{ background: ring }" role="img" :aria-label="`${caption} ${total}`">
      <div class="hole"><b>{{ total }}</b><small>{{ caption }}</small></div>
    </div>
    <ul class="legend">
      <li v-for="s in slices" :key="s.name">
        <i :style="{ background: s.color }" />
        <span class="grow">{{ s.name }}</span>
        <b>{{ s.qty }}</b><small>（{{ pct(s.qty) }}%）</small>
      </li>
    </ul>
  </div>
</template>

<style scoped>
.donut-wrap { display: flex; align-items: center; gap: 28px; flex-wrap: wrap; }
.donut { width: 150px; height: 150px; border-radius: 50%; display: flex; align-items: center; justify-content: center; flex: none; }
.hole {
  width: 104px; height: 104px; border-radius: 50%; background: #fff; display: flex; flex-direction: column; align-items: center; justify-content: center;
  box-shadow: inset 0 2px 6px rgba(108, 99, 190, 0.12);
}
.hole b { font-size: 26px; line-height: 1.1; }
.hole small { font-size: 12px; color: var(--muted); }
.legend { list-style: none; margin: 0; padding: 0; flex: 1; min-width: 200px; display: flex; flex-direction: column; gap: 4px; }
.legend li { display: flex; align-items: center; gap: 10px; padding: 9px 0; border-bottom: 1px solid #f0eff8; font-size: 13.5px; }
.legend li:last-child { border-bottom: 0; }
.legend i { width: 10px; height: 10px; border-radius: 3px; flex: none; }
.legend small { color: var(--muted); font-size: 12px; }
</style>
