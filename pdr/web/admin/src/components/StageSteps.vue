<script setup lang="ts">
import type { Stage } from '../api'
import { STAGES } from '../lib/format'

const props = defineProps<{ current: Stage; notes?: Partial<Record<Stage, string>>; skipped?: Stage[] }>()
const idx = STAGES.findIndex((s) => s.key === props.current)
</script>

<template>
  <ol class="steps card">
    <li
      v-for="(s, i) in STAGES"
      :key="s.key"
      :class="{ done: i < idx && !skipped?.includes(s.key), cur: i === idx, skip: skipped?.includes(s.key) }"
      :aria-current="i === idx ? 'step' : undefined"
    >
      <span class="n">{{ i + 1 }}</span>
      <span>{{ s.name }}<template v-if="notes?.[s.key]"> · {{ notes[s.key] }}</template><template v-if="i === idx">（当前）</template></span>
    </li>
  </ol>
</template>

<style scoped>
.steps { list-style: none; margin: 0; padding: 12px 16px; display: flex; flex-wrap: wrap; gap: 6px 4px; align-items: center; font-size: 13px; color: var(--muted); }
li { display: flex; align-items: center; gap: 6px; }
li:not(:last-child)::after { content: ''; width: 14px; height: 1px; background: #d6d3ea; margin: 0 4px; }
.n { width: 20px; height: 20px; border-radius: 50%; display: inline-flex; align-items: center; justify-content: center; font-size: 11.5px; background: #eeedf7; color: var(--muted); }
.done { color: var(--ok); }
.done .n { background: var(--ok-soft); color: var(--ok); }
.cur { color: var(--accent); font-weight: 600; }
.cur .n { background: var(--accent); color: #fff; }
.skip { color: var(--faint); text-decoration: line-through; }
</style>
