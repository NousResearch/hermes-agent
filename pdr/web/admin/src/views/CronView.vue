<script setup lang="ts">
import { onMounted, ref } from 'vue'
import { api, type CronJob, type GatewayPlatform } from '../api'
import AppIcon from '../components/AppIcon.vue'
import { errorText, toast } from '../lib/toast'

const jobs = ref<CronJob[]>([])
const platforms = ref<GatewayPlatform[]>([])
const running = ref('')

const LAST: Record<CronJob['lastStatus'], [string, string]> = { ok: ['成功', 'green'], failed: ['失败', 'pink'], never: ['未执行', 'gray'] }
const PLAT: Record<GatewayPlatform['status'], [string, string]> = { online: ['在线', 'green'], offline: ['离线', 'pink'], disabled: ['未启用', 'gray'] }

async function flip(j: CronJob): Promise<void> {
  try {
    await api.toggleCron(j.id, !j.enabled)
    j.enabled = !j.enabled
    toast(`${j.name}已${j.enabled ? '启用' : '暂停'}`)
  } catch (e) {
    toast(errorText(e))
  }
}

async function run(j: CronJob): Promise<void> {
  running.value = j.id
  try {
    Object.assign(j, await api.runCron(j.id))
    toast(`${j.name} 已执行完成`)
  } catch (e) {
    toast(errorText(e))
  } finally {
    running.value = ''
  }
}

onMounted(async () => {
  try {
    const r = await api.cronJobs()
    jobs.value = r.jobs
    platforms.value = r.platforms
  } catch (e) {
    toast(errorText(e))
  }
})
</script>

<template>
  <div class="stack">
    <div class="page-head">
      <div>
        <h1>定时任务 · 消息网关</h1>
        <p>定时任务由智能体按计划执行并投递结果；消息网关是智能体对外的接入渠道，用户端问答经 api_server 接入。</p>
      </div>
    </div>

    <section class="card">
      <div class="card-head"><h2 class="sec">定时任务</h2><span class="muted small">时区 Asia/Shanghai</span></div>
      <div class="tbl-wrap pad">
        <table class="tbl">
          <thead><tr><th>任务</th><th>执行计划</th><th>结果投递</th><th>下次执行</th><th>上次执行</th><th>启用</th><th>操作</th></tr></thead>
          <tbody>
            <tr v-for="j in jobs" :key="j.id" :class="{ off: !j.enabled }">
              <td><b>{{ j.name }}</b></td>
              <td>{{ j.scheduleText }}<br /><span class="mono faint">{{ j.schedule }}</span></td>
              <td>{{ j.deliver }}</td>
              <td class="muted">{{ j.enabled ? j.nextRun : '—' }}</td>
              <td><span class="muted">{{ j.lastRun }}</span> <span class="pill" :class="LAST[j.lastStatus][1]">{{ LAST[j.lastStatus][0] }}</span></td>
              <td><span class="switch"><input type="checkbox" :checked="j.enabled" :aria-label="`启用 ${j.name}`" @change="flip(j)" /><span /></span></td>
              <td>
                <button type="button" class="btn" :disabled="running === j.id" @click="run(j)">
                  <AppIcon name="play" :size="13" />{{ running === j.id ? '执行中…' : '立即执行' }}
                </button>
              </td>
            </tr>
          </tbody>
        </table>
      </div>
    </section>

    <section>
      <h2 class="sec h2">消息网关</h2>
      <div class="plats">
        <div v-for="p in platforms" :key="p.name" class="card plat">
          <div class="row">
            <span class="pico" :class="p.status"><AppIcon :name="p.name === 'api_server' ? 'globe' : 'message'" :size="16" /></span>
            <b class="grow">{{ p.label }}</b>
            <span class="pill" :class="PLAT[p.status][1]">{{ PLAT[p.status][0] }}</span>
          </div>
          <p class="muted">{{ p.note }}</p>
          <small>活跃会话 {{ p.sessions }}</small>
        </div>
      </div>
    </section>
  </div>
</template>

<style scoped>
.pad { padding: 4px 20px 12px; }
.small { font-size: 12.5px; }
tr.off td { color: var(--faint); }
.h2 { margin-bottom: 12px; }
.plats { display: grid; grid-template-columns: repeat(auto-fill, minmax(240px, 1fr)); gap: 14px; }
.plat { padding: 16px; border-radius: 16px; display: flex; flex-direction: column; gap: 8px; }
.plat p { margin: 0; font-size: 13px; }
.plat small { font-size: 12.5px; color: var(--text-2); }
.pico { width: 32px; height: 32px; border-radius: 10px; display: flex; align-items: center; justify-content: center; background: #eef0f5; color: var(--faint); }
.pico.online { background: var(--ok-soft); color: var(--ok); }
.pico.offline { background: var(--bad-soft); color: var(--bad); }
</style>
