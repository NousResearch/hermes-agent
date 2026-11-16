<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import { useRouter } from 'vue-router'
import { api, type Asset, type Dashboard, type Perm } from '../api'
import AppIcon from '../components/AppIcon.vue'
import DonutChart from '../components/DonutChart.vue'
import MiniCalendar from '../components/MiniCalendar.vue'
import TrendChart from '../components/TrendChart.vue'
import UploadDialog from '../components/UploadDialog.vue'
import { MODALITY_TINT, nextAction, STATUS } from '../lib/format'
import type { IconName } from '../lib/icons'
import { errorText, toast } from '../lib/toast'
import { useAuth } from '../stores/auth'

const auth = useAuth()
const router = useRouter()
const data = ref<Dashboard | null>(null)
const tab = ref(0)
const uploading = ref(false)

const hour = new Date().getHours()
const greeting = hour < 11 ? '早上好' : hour < 14 ? '中午好' : hour < 18 ? '下午好' : '晚上好'

// 待办页签只出现登录者能处理的类型
const tabs = computed(() => ([
  { name: '全部', test: () => true, perm: 'content:asset' },
  { name: '待核对', test: (a: Asset) => a.stage === 'review', perm: 'content:review' },
  { name: '待标注', test: (a: Asset) => a.stage === 'label', perm: 'content:label' },
  { name: '待分类', test: (a: Asset) => a.stage === 'classify', perm: 'content:classify' },
  { name: '已完成', test: (a: Asset) => a.status === 'done', perm: 'content:asset' }
] as { name: string; test: (a: Asset) => boolean; perm: Perm }[]).filter((t) => auth.can(t.perm)))
const rows = computed(() => {
  const t = tabs.value[Math.min(tab.value, tabs.value.length - 1)]
  return (data.value?.myTasks ?? []).filter((a) => (t ? t.test(a) : true))
})
const plan = computed(() => data.value?.weekPlan)
const planPct = computed(() => (plan.value ? Math.round((plan.value.stored / plan.value.planned) * 100) : 0))

const KPI_ICON: Record<string, IconName> = { uploaded: 'upload', manual: 'review', stored: 'database', failed: 'alert' }

async function load(): Promise<void> {
  try {
    data.value = await api.dashboard()
  } catch (e) {
    toast(errorText(e))
  }
}

async function act(a: Asset): Promise<void> {
  const next = nextAction(a)
  if (next.retry) {
    if (!auth.can('content:retry')) return toast('没有重新执行的权限')
    await api.retry(a.no)
    toast('已重新执行，进入解析队列')
    return load()
  }
  if (next.path) router.push({ path: next.path, query: { no: a.no } })
  else if (auth.can('content:asset')) router.push({ path: '/content/tasks', query: { keyword: a.no } })
}

onMounted(load)
</script>

<template>
  <div class="dash">
    <div class="page-head">
      <div>
        <h1>{{ greeting }}，{{ auth.user?.name }}</h1>
        <p>这是今天内容处理的情况，只显示你有权限的数据与功能（数据范围：{{ auth.user?.dataScopeText }}）。</p>
      </div>
      <button v-if="auth.can('content:upload')" type="button" class="btn-p" @click="uploading = true">
        <AppIcon name="plus" :size="16" />上传文件
      </button>
    </div>

    <div v-if="data" class="cols">
      <div class="left stack">
        <div class="kpis">
          <div v-for="k in data.kpis" :key="k.key" class="kpi" :class="k.tint">
            <span class="tile"><AppIcon :name="KPI_ICON[k.key] || 'chart'" :size="24" /></span>
            <div>
              <p class="kname">{{ k.name }}</p>
              <p class="kval">{{ k.value }}</p>
              <p class="kdelta">
                <template v-if="k.delta"><b :class="k.delta > 0 ? 'up' : 'down'">{{ k.delta > 0 ? '+' : '' }}{{ k.delta }}%</b> 较上周</template>
                <template v-else>与上周持平</template>
              </p>
            </div>
          </div>
        </div>

        <section class="card">
          <div class="card-head">
            <h2 class="sec">我的任务</h2>
            <RouterLink v-if="auth.can('content:asset')" to="/content/tasks" class="btn">全部任务 <AppIcon name="arrowRight" :size="14" /></RouterLink>
          </div>
          <div class="tabs pad-x" role="tablist">
            <button
              v-for="(t, i) in tabs"
              :key="t.name"
              type="button"
              role="tab"
              :aria-selected="i === Math.min(tab, tabs.length - 1)"
              @click="tab = i"
            >{{ t.name }}<span class="qty">{{ (data.myTasks).filter(t.test).length }}</span></button>
          </div>
          <ul class="tasks">
            <li v-for="a in rows" :key="a.no">
              <span class="check" :class="{ done: a.status === 'done' }"><AppIcon v-if="a.status === 'done'" name="check" :size="12" :stroke="3" /></span>
              <span class="grow">
                <span class="tname ellipsis">{{ a.name }}</span>
                <small class="muted">{{ a.no }} · {{ a.note }}</small>
              </span>
              <span class="pill" :class="MODALITY_TINT[a.modality]">{{ a.modality }}</span>
              <span class="date"><AppIcon name="calendar" :size="13" />{{ a.updatedAt.slice(0, 5) }}</span>
              <span class="pill" :class="STATUS[a.status][1]">{{ STATUS[a.status][0] }}</span>
              <button type="button" class="btn-link act" @click="act(a)">{{ nextAction(a).label }}</button>
            </li>
            <li v-if="!rows.length" class="empty">这里没有待办，休息一下</li>
          </ul>
        </section>

        <div class="charts">
          <section class="card card-pad">
            <div class="row between">
              <h2 class="sec">入库趋势（个）</h2>
              <span class="pill gray">本周</span>
            </div>
            <TrendChart :points="data.trend" unit="入库" />
          </section>
          <section class="card card-pad">
            <h2 class="sec">资产状态分布</h2>
            <DonutChart :slices="data.distribution" caption="本周资产" />
          </section>
        </div>
      </div>

      <aside class="right stack">
        <section class="card card-pad"><MiniCalendar today="2026-11-16" :events="data.events" /></section>
        <section v-if="plan" class="card card-pad">
          <h2 class="sec">本周入库进度</h2>
          <div class="row between plan">
            <span class="muted">计划入库 {{ plan.planned }} 个资产</span>
            <b>{{ planPct }}%</b>
          </div>
          <div class="pbar"><i :style="{ width: planPct + '%' }" /></div>
          <p class="muted small">已入库 {{ plan.stored }} 个，剩余 {{ plan.reviewing }} 个在核对与分类阶段</p>
        </section>
        <section class="card card-pad">
          <h2 class="sec">近期事项</h2>
          <ul class="events">
            <li v-for="e in data.events" :key="e.date">
              <span class="edate"><b>{{ e.date.slice(3) }}</b><small>{{ e.date.slice(0, 2) }} 月</small></span>
              <span class="grow">{{ e.title }}</span>
              <span class="pill purple">{{ e.tag }}</span>
            </li>
          </ul>
          <RouterLink v-if="auth.can('agent:cron')" to="/agent/cron" class="more">查看定时任务 <AppIcon name="arrowRight" :size="14" /></RouterLink>
        </section>
      </aside>
    </div>

    <UploadDialog v-if="uploading" @close="uploading = false" @done="uploading = false; load()" />
  </div>
</template>

<style scoped>
.cols { display: grid; grid-template-columns: minmax(0, 1fr) 320px; gap: 20px; align-items: start; }
.kpis { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 14px; }
.kpi { border-radius: 18px; padding: 18px 16px; display: flex; gap: 14px; align-items: center; border: 1px solid rgba(255, 255, 255, 0.9); box-shadow: 0 8px 20px rgba(108, 99, 190, 0.1); }
.kpi .tile { flex: none; width: 54px; height: 54px; border-radius: 16px; color: #fff; display: flex; align-items: center; justify-content: center; }
.kpi.purple { background: linear-gradient(135deg, #f1eeff, #e7e2fe); }
.kpi.purple .tile { background: linear-gradient(135deg, #9e92fb, #6d5df0); box-shadow: inset 0 2px 0 rgba(255, 255, 255, 0.45), 0 8px 16px rgba(109, 93, 240, 0.35); }
.kpi.green { background: linear-gradient(135deg, #eaf8f0, #ddf3e6); }
.kpi.green .tile { background: linear-gradient(135deg, #6fd9a2, #2fae6e); box-shadow: inset 0 2px 0 rgba(255, 255, 255, 0.45), 0 8px 16px rgba(47, 174, 110, 0.35); }
.kpi.amber { background: linear-gradient(135deg, #fff8e6, #ffefc9); }
.kpi.amber .tile { background: linear-gradient(135deg, #ffd66b, #f2b33d); box-shadow: inset 0 2px 0 rgba(255, 255, 255, 0.45), 0 8px 16px rgba(242, 179, 61, 0.35); }
.kpi.pink { background: linear-gradient(135deg, #ffeff2, #fde1e7); }
.kpi.pink .tile { background: linear-gradient(135deg, #ff9aae, #f0607c); box-shadow: inset 0 2px 0 rgba(255, 255, 255, 0.45), 0 8px 16px rgba(240, 96, 124, 0.35); }
.kpi p { margin: 0; }
.kname { font-size: 13px; color: var(--text-2); }
.kval { font-size: 28px; font-weight: 700; line-height: 1.2; }
.kdelta { font-size: 12px; color: var(--muted); }
.kdelta .up { color: var(--ok); }
.kdelta .down { color: var(--bad); }
.pad-x { padding: 0 20px; }
.tasks { list-style: none; margin: 0; padding: 4px 20px 10px; }
.tasks li { display: flex; align-items: center; gap: 14px; padding: 12px 0; border-bottom: 1px solid #f0eff8; }
.tasks li:last-child { border-bottom: 0; }
.check { flex: none; width: 20px; height: 20px; border-radius: 50%; border: 2px solid #cfcbe6; display: flex; align-items: center; justify-content: center; color: #fff; }
.check.done { background: #2fae6e; border-color: #2fae6e; }
.tname { display: block; font-weight: 500; }
.tasks small { font-size: 12px; }
.date { display: inline-flex; align-items: center; gap: 4px; color: var(--muted); font-size: 12.5px; white-space: nowrap; }
.act { min-width: 56px; text-align: right; }
.charts { display: grid; grid-template-columns: minmax(0, 1.3fr) minmax(0, 1fr); gap: 16px; }
.charts .sec { margin-bottom: 14px; }
.between { justify-content: space-between; }
.plan { margin: 12px 0 8px; }
.plan b { font-size: 22px; }
.pbar { height: 8px; border-radius: 999px; background: #eeedf7; overflow: hidden; }
.pbar i { display: block; height: 100%; border-radius: inherit; background: linear-gradient(90deg, #6fd9a2, #2fae6e); }
.small { font-size: 12.5px; margin: 10px 0 0; }
.events { list-style: none; margin: 10px 0 6px; padding: 0; }
.events li { display: flex; align-items: center; gap: 12px; padding: 9px 0; border-bottom: 1px solid #f0eff8; font-size: 13px; }
.edate { flex: none; width: 42px; height: 42px; border-radius: 12px; background: var(--soft); display: flex; flex-direction: column; align-items: center; justify-content: center; line-height: 1.1; }
.edate b { font-size: 15px; }
.edate small { font-size: 10.5px; color: var(--muted); }
.more { display: inline-flex; align-items: center; gap: 4px; font-size: 13px; margin-top: 4px; }
@media (max-width: 1280px) {
  .cols { grid-template-columns: 1fr; }
  .right { display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); }
}
@media (max-width: 960px) {
  .kpis { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .charts { grid-template-columns: 1fr; }
}
@media (max-width: 560px) {
  .tasks li { flex-wrap: wrap; }
  .date { display: none; }
}
</style>
