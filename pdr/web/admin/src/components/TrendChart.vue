<script setup lang="ts">
import { computed, ref } from 'vue'
import type { TrendPoint } from '../api'

const props = defineProps<{ points: TrendPoint[]; unit?: string; max?: number }>()

const W = 600
const H = 190
const hover = ref(3)
const top = computed(() => {
  const m = Math.max(...props.points.map((p) => p.value), 1)
  // 取 8 的倍数作上限，四等分刻度都是整数
  return props.max ?? Math.ceil(m / 8) * 8
})
const ticks = computed(() => [4, 3, 2, 1, 0].map((i) => Math.round((top.value / 4) * i)))
const xy = computed(() => props.points.map((p, i) => [
  props.points.length > 1 ? (i / (props.points.length - 1)) * W : W / 2,
  H - (p.value / top.value) * H
]))
const line = computed(() => xy.value.map((p) => p.join(',')).join(' '))
const area = computed(() => `0,${H} ${line.value} ${W},${H}`)
const tip = computed(() => {
  const i = Math.min(hover.value, props.points.length - 1)
  const [x, y] = xy.value[i] || [0, 0]
  return { left: (x / W) * 100, top: (y / H) * 100, p: props.points[i] }
})
</script>

<template>
  <div class="trend">
    <div class="axis" aria-hidden="true">
      <span v-for="t in ticks" :key="t">{{ t }}</span>
    </div>
    <div class="plot">
      <div class="grid" aria-hidden="true"><i v-for="t in ticks" :key="t" /></div>
      <svg :viewBox="`0 0 ${W} ${H}`" preserveAspectRatio="none" aria-hidden="true">
        <defs>
          <linearGradient id="trend-fill" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0" stop-color="#7B6CF6" stop-opacity="0.22" />
            <stop offset="1" stop-color="#7B6CF6" stop-opacity="0" />
          </linearGradient>
        </defs>
        <polygon :points="area" fill="url(#trend-fill)" />
        <polyline :points="line" fill="none" stroke="#5443D6" stroke-width="2.5" vector-effect="non-scaling-stroke" stroke-linejoin="round" />
      </svg>
      <button
        v-for="(p, i) in points"
        :key="p.day"
        type="button"
        class="hit"
        :style="{ left: `calc(${(i / Math.max(points.length - 1, 1)) * 100}% - 14px)`, top: `calc(${(1 - p.value / top) * 100}% - 14px)` }"
        :aria-label="`${p.day} ${p.value}${unit ?? ''}`"
        @mouseenter="hover = i"
        @focus="hover = i"
      >
        <span class="dot" :class="{ on: i === hover }" />
      </button>
      <div v-if="tip.p" class="tip" :style="{ left: `${tip.left}%`, top: `${tip.top}%` }">
        <b>{{ tip.p.value }}</b><small>{{ tip.p.day }} {{ unit }}</small>
      </div>
    </div>
    <div class="days" aria-hidden="true">
      <span v-for="p in points" :key="p.day">{{ p.day }}</span>
    </div>
    <p class="sr-only">{{ points.map((p) => `${p.day} ${p.value}`).join('，') }}</p>
  </div>
</template>

<style scoped>
.trend { display: grid; grid-template-columns: 32px 1fr; grid-template-rows: 190px auto; column-gap: 8px; row-gap: 8px; }
.axis { display: flex; flex-direction: column; justify-content: space-between; font-size: 11.5px; color: var(--faint); text-align: right; transform: translateY(-7px); height: calc(100% + 14px); }
.plot { position: relative; }
.grid { position: absolute; inset: 0; display: flex; flex-direction: column; justify-content: space-between; }
.grid i { display: block; height: 1px; background: #efedf8; }
svg { position: absolute; inset: 0; width: 100%; height: 100%; overflow: visible; }
.hit { position: absolute; width: 28px; height: 28px; border: 0; padding: 0; background: transparent; display: flex; align-items: center; justify-content: center; }
.dot { width: 9px; height: 9px; border-radius: 50%; background: #fff; border: 2px solid var(--accent); }
.dot.on { width: 12px; height: 12px; box-shadow: 0 0 0 4px rgba(84, 67, 214, 0.18); }
.tip {
  position: absolute; transform: translate(-50%, calc(-100% - 14px)); background: #23263f; color: #fff; border-radius: 10px;
  padding: 5px 10px; white-space: nowrap; pointer-events: none; display: flex; gap: 6px; align-items: baseline; font-size: 12px;
}
.tip b { font-size: 14px; }
.tip small { color: #c9c6e6; }
.days { grid-column: 2; display: flex; justify-content: space-between; font-size: 12px; color: var(--muted); }
</style>
