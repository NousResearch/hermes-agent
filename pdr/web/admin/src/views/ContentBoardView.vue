<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import { api, type ContentBoard } from '../api'
import { MODALITY_TINT } from '../lib/format'
import { errorText, toast } from '../lib/toast'

const board = ref<ContentBoard | null>(null)
const top = computed(() => Math.max(...(board.value?.funnel.map((f) => f.qty) ?? [1])))
const modTop = computed(() => Math.max(...(board.value?.byModality.map((m) => m.qty) ?? [1])))
const slowest = computed(() => Math.max(...(board.value?.avgMinutes.map((m) => m.minutes) ?? [1])))

onMounted(async () => {
  try {
    board.value = await api.contentBoard()
  } catch (e) {
    toast(errorText(e))
  }
})
</script>

<template>
  <div class="stack">
    <div class="page-head">
      <div>
        <h1>内容看板</h1>
        <p>本周各阶段的流转、各类文件的处理量与失败原因；数据范围外的资产不计入。</p>
      </div>
      <span class="pill gray">统计周期：11-10 至 11-16</span>
    </div>

    <div v-if="board" class="grid">
      <section class="card card-pad funnel">
        <h2 class="sec">阶段漏斗（个）</h2>
        <ul>
          <li v-for="(f, i) in board.funnel" :key="f.stage">
            <span class="fname">{{ f.name }}</span>
            <span class="track"><i :style="{ width: (f.qty / top) * 100 + '%', opacity: 1 - i * 0.08 }" /></span>
            <b>{{ f.qty }}</b>
            <small class="muted">{{ i ? Math.round((f.qty / board.funnel[i - 1].qty) * 100) + '%' : '—' }}</small>
          </li>
        </ul>
        <p class="muted hint">百分比为相对上一阶段的通过率；人工核对与分类需要人处理，耗时最长。</p>
      </section>

      <section class="card card-pad">
        <h2 class="sec">各类文件处理量</h2>
        <ul class="bars">
          <li v-for="m in board.byModality" :key="m.modality">
            <span class="pill" :class="MODALITY_TINT[m.modality]">{{ m.modality }}</span>
            <span class="track"><i :style="{ width: (m.qty / modTop) * 100 + '%' }" /></span>
            <b>{{ m.qty }}</b>
            <small :class="m.failed ? 'bad' : 'muted'">失败 {{ m.failed }}</small>
          </li>
        </ul>
      </section>

      <section class="card card-pad">
        <h2 class="sec">失败原因</h2>
        <table class="tbl">
          <thead><tr><th>错误码</th><th>说明</th><th>次数</th><th>建议</th></tr></thead>
          <tbody>
            <tr v-for="f in board.failures" :key="f.code">
              <td class="mono">{{ f.code }}</td>
              <td>{{ f.name }}</td>
              <td>{{ f.qty }}</td>
              <td class="muted">{{ f.hint }}</td>
            </tr>
          </tbody>
        </table>
      </section>

      <section class="card card-pad">
        <h2 class="sec">各阶段平均耗时（分钟）</h2>
        <ul class="bars">
          <li v-for="m in board.avgMinutes" :key="m.stage">
            <span class="fname">{{ m.name }}</span>
            <span class="track amber"><i :style="{ width: (m.minutes / slowest) * 100 + '%' }" /></span>
            <b>{{ m.minutes }}</b>
          </li>
        </ul>
      </section>
    </div>
  </div>
</template>

<style scoped>
.grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 16px; }
.sec { margin-bottom: 14px; }
ul { list-style: none; margin: 0; padding: 0; display: flex; flex-direction: column; gap: 12px; }
li { display: flex; align-items: center; gap: 12px; font-size: 13.5px; }
.fname { width: 64px; flex: none; color: var(--text-2); }
.track { flex: 1; height: 12px; border-radius: 999px; background: #f0eef9; overflow: hidden; }
.track i { display: block; height: 100%; border-radius: inherit; background: linear-gradient(90deg, #9e92fb, #5443d6); }
.track.amber i { background: linear-gradient(90deg, #ffd66b, #f2b33d); }
li b { width: 34px; text-align: right; }
li small { width: 44px; font-size: 12px; }
.bars .pill { width: 72px; justify-content: center; }
.bad { color: var(--bad); }
.hint { margin: 14px 0 0; font-size: 12.5px; }
@media (max-width: 1100px) { .grid { grid-template-columns: 1fr; } }
</style>
