<script setup lang="ts">
import { computed, onMounted, ref, watch } from 'vue'
import { api, type QaSession } from '../api'
import AppIcon from '../components/AppIcon.vue'
import { errorText, toast } from '../lib/toast'

const rows = ref<QaSession[]>([])
const kw = ref('')
const feedback = ref<'' | 'up' | 'down' | 'none'>('')
const open = ref<QaSession | null>(null)

const shown = computed(() => rows.value.filter((r) =>
  !feedback.value || (feedback.value === 'none' ? r.feedback === null : r.feedback === feedback.value)))
const stats = computed(() => ({
  total: rows.value.length,
  noCite: rows.value.filter((r) => !r.citations.length).length,
  down: rows.value.filter((r) => r.feedback === 'down').length,
  avg: rows.value.length ? (rows.value.reduce((n, r) => n + r.durationMs, 0) / rows.value.length / 1000).toFixed(1) : '0'
}))

async function load(): Promise<void> {
  try {
    rows.value = await api.qaSessions(kw.value)
  } catch (e) {
    toast(errorText(e))
  }
}

let timer: ReturnType<typeof setTimeout> | undefined
watch(kw, () => {
  clearTimeout(timer)
  timer = setTimeout(load, 250)
})
onMounted(load)
</script>

<template>
  <div class="stack">
    <div class="page-head">
      <div>
        <h1>问答会话审计</h1>
        <p>用户端的每次问答都会留存问题、回答、引用的资料与反馈，便于发现资料缺口和回答质量问题。</p>
      </div>
    </div>

    <div class="stats">
      <div class="card stat"><small>会话</small><b>{{ stats.total }}</b></div>
      <div class="card stat"><small>没有引用资料的回答</small><b class="warn">{{ stats.noCite }}</b></div>
      <div class="card stat"><small>点踩</small><b class="bad">{{ stats.down }}</b></div>
      <div class="card stat"><small>平均回答耗时</small><b>{{ stats.avg }} 秒</b></div>
    </div>

    <section class="card">
      <div class="card-head">
        <h2 class="sec">会话记录</h2>
        <div class="row">
          <select v-model="feedback" class="fld" aria-label="按反馈筛选">
            <option value="">全部反馈</option>
            <option value="up">点赞</option>
            <option value="down">点踩</option>
            <option value="none">无反馈</option>
          </select>
          <input v-model="kw" class="fld" type="search" placeholder="搜索问题或提问人" aria-label="搜索问题或提问人" />
        </div>
      </div>
      <div class="tbl-wrap pad">
        <table class="tbl">
          <thead><tr><th>时间</th><th>提问人</th><th>问题</th><th>资料范围</th><th>引用</th><th>反馈</th><th>耗时</th><th /></tr></thead>
          <tbody>
            <tr v-for="r in shown" :key="r.id">
              <td class="muted nowrap">{{ r.createdAt }}</td>
              <td class="nowrap">{{ r.user }}<br /><small class="faint">{{ r.region }}</small></td>
              <td class="q">{{ r.question }}</td>
              <td class="nowrap">{{ r.scopeName }}</td>
              <td><span class="pill" :class="r.citations.length ? 'purple' : 'amber'">{{ r.citations.length ? `${r.citations.length} 份` : '无' }}</span></td>
              <td>
                <AppIcon v-if="r.feedback === 'up'" name="thumbUp" :size="15" class="up" />
                <AppIcon v-else-if="r.feedback === 'down'" name="thumbDown" :size="15" class="down" />
                <span v-else class="faint">—</span>
              </td>
              <td class="muted nowrap">{{ (r.durationMs / 1000).toFixed(1) }} 秒</td>
              <td><button type="button" class="btn-link" @click="open = r">查看</button></td>
            </tr>
          </tbody>
        </table>
        <p v-if="!shown.length" class="empty">没有符合条件的会话</p>
      </div>
    </section>

    <div v-if="open" class="mask" @click.self="open = null">
      <aside class="drawer" role="dialog" aria-modal="true" aria-labelledby="sess-title">
        <header class="card-head">
          <h2 id="sess-title" class="sec">会话详情</h2>
          <button type="button" class="ib" aria-label="关闭" @click="open = null"><AppIcon name="x" /></button>
        </header>
        <div class="dbody">
          <p class="muted small">{{ open.user }} · {{ open.region }} · {{ open.createdAt }} · 共 {{ open.turns }} 轮 · 资料范围「{{ open.scopeName }}」</p>
          <div class="bubble q">{{ open.question }}</div>
          <div class="bubble a">{{ open.answer }}</div>
          <h3>引用的资料</h3>
          <ol v-if="open.citations.length" class="cites">
            <li v-for="(c, i) in open.citations" :key="i"><b>{{ c.title }}</b><span class="muted">{{ c.locator }}</span></li>
          </ol>
          <p v-else class="notice warn">这次回答没有找到可引用的资料，可能是知识库缺少相关内容。</p>
        </div>
      </aside>
    </div>
  </div>
</template>

<style scoped>
.stats { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 14px; }
.stat { padding: 16px 18px; display: flex; flex-direction: column; gap: 4px; }
.stat small { color: var(--muted); font-size: 12.5px; }
.stat b { font-size: 24px; }
.stat .warn { color: var(--warn); }
.stat .bad { color: var(--bad); }
.pad { padding: 4px 20px 12px; }
.q { min-width: 240px; }
.nowrap { white-space: nowrap; }
.up { color: var(--ok); }
.down { color: var(--bad); }
.mask { position: fixed; inset: 0; z-index: 40; background: rgba(35, 38, 63, 0.28); display: flex; justify-content: flex-end; }
.drawer { width: min(480px, 100%); height: 100%; background: #fff; box-shadow: -10px 0 30px rgba(35, 38, 63, 0.18); display: flex; flex-direction: column; }
.dbody { padding: 16px 20px; overflow-y: auto; display: flex; flex-direction: column; gap: 12px; }
.small { font-size: 12.5px; margin: 0; }
.bubble { padding: 12px 14px; border-radius: 14px; font-size: 14px; line-height: 1.7; }
.bubble.q { align-self: flex-end; background: var(--accent-soft); max-width: 90%; }
.bubble.a { background: var(--soft); }
h3 { font-size: 14px; margin-top: 4px; }
.cites { margin: 0; padding-left: 20px; display: flex; flex-direction: column; gap: 6px; font-size: 13.5px; }
.cites span { margin-left: 8px; font-size: 12.5px; }
@media (max-width: 960px) { .stats { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
</style>
