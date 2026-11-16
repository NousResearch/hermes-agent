<script setup lang="ts">
import { computed, ref } from 'vue'
import type { CalendarEvent } from '../api'
import AppIcon from './AppIcon.vue'

const props = defineProps<{ today: string; events: CalendarEvent[] }>()

const base = new Date(props.today + 'T00:00:00')
const month = ref(new Date(base.getFullYear(), base.getMonth(), 1))
const selected = ref(base.getDate())

const title = computed(() => `${month.value.getFullYear()} 年 ${month.value.getMonth() + 1} 月`)
const cells = computed(() => {
  const y = month.value.getFullYear()
  const m = month.value.getMonth()
  const lead = (new Date(y, m, 1).getDay() + 6) % 7
  const days = new Date(y, m + 1, 0).getDate()
  const out: { n: number; inMonth: boolean; event?: CalendarEvent; today: boolean }[] = []
  const prevDays = new Date(y, m, 0).getDate()
  for (let i = lead; i > 0; i--) out.push({ n: prevDays - i + 1, inMonth: false, today: false })
  for (let d = 1; d <= days; d++) {
    const key = `${String(m + 1).padStart(2, '0')}-${String(d).padStart(2, '0')}`
    out.push({
      n: d, inMonth: true, event: props.events.find((e) => e.date === key),
      today: y === base.getFullYear() && m === base.getMonth() && d === base.getDate()
    })
  }
  return out
})
const sameMonth = computed(() => month.value.getMonth() === base.getMonth() && month.value.getFullYear() === base.getFullYear())
const picked = computed(() => (sameMonth.value ? cells.value.find((c) => c.inMonth && c.n === selected.value)?.event : undefined))

function shift(delta: number): void {
  month.value = new Date(month.value.getFullYear(), month.value.getMonth() + delta, 1)
}
</script>

<template>
  <div>
    <div class="head">
      <div>
        <h2 class="sec">日历</h2>
        <p class="muted">{{ title }}</p>
      </div>
      <div class="row">
        <button type="button" class="ib" aria-label="上个月" @click="shift(-1)"><AppIcon name="chevronLeft" /></button>
        <button type="button" class="ib" aria-label="下个月" @click="shift(1)"><AppIcon name="chevronRight" /></button>
      </div>
    </div>
    <div class="grid">
      <span v-for="w in ['一', '二', '三', '四', '五', '六', '日']" :key="w" class="wk">{{ w }}</span>
      <button
        v-for="(c, i) in cells"
        :key="i"
        type="button"
        class="day"
        :class="{ out: !c.inMonth, today: c.today, ev: !!c.event, sel: c.inMonth && sameMonth && c.n === selected }"
        :disabled="!c.inMonth"
        :title="c.event?.title"
        @click="selected = c.n"
      >{{ c.n }}</button>
    </div>
    <p class="picked">{{ picked ? `${picked.date}：${picked.title}` : '当天没有安排' }}</p>
  </div>
</template>

<style scoped>
.head { display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 10px; }
.head p { margin: 2px 0 0; font-size: 12.5px; }
.grid { display: grid; grid-template-columns: repeat(7, 1fr); gap: 2px; text-align: center; }
.wk { font-size: 12px; color: var(--faint); padding: 4px 0; }
.day {
  border: 0; background: none; width: 30px; height: 30px; margin: 0 auto; border-radius: 50%; font-size: 12.5px; color: var(--text-2); position: relative;
}
.day:hover:not(:disabled) { background: #f2f0fb; }
.day.out { color: #c3c5d6; cursor: default; opacity: 1; }
.day.ev::after { content: ''; position: absolute; left: 50%; bottom: 3px; width: 4px; height: 4px; margin-left: -2px; border-radius: 50%; background: #f2b33d; }
.day.today { background: linear-gradient(135deg, var(--accent-2), var(--accent)); color: #fff; box-shadow: 0 4px 10px rgba(84, 67, 214, 0.35); }
.day.sel:not(.today) { box-shadow: inset 0 0 0 2px var(--accent); }
.picked { margin: 10px 0 0; font-size: 12.5px; color: var(--muted); }
</style>
