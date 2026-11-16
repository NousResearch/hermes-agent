<script setup lang="ts">
import { computed, onMounted, ref, watch } from 'vue'
import { api, type AuditLog } from '../api'
import { errorText, toast } from '../lib/toast'

const MODULES = ['', '登录', '内容任务', '人工核对', '知识库', '智能体', '权限管理']
const module = ref('')
const rows = ref<AuditLog[]>([])
const kw = ref('')
const RESULT: Record<AuditLog['result'], [string, string]> = { ok: ['成功', 'green'], denied: ['拒绝（无权限）', 'pink'], failed: ['失败', 'amber'] }
const shown = computed(() => rows.value.filter((r) => !kw.value.trim() || [r.user, r.action, r.target].some((s) => s.includes(kw.value.trim()))))

async function load(): Promise<void> {
  try {
    rows.value = await api.auditLogs(module.value)
  } catch (e) {
    toast(errorText(e))
  }
}

watch(module, load)
onMounted(load)
</script>

<template>
  <div class="stack">
    <div class="page-head">
      <div>
        <h1>审计日志</h1>
        <p>登录、内容处理、配置与权限变更都会留痕；没有权限的访问会以“拒绝”记录。</p>
      </div>
    </div>
    <section class="card">
      <div class="card-head">
        <div class="tabs" role="tablist">
          <button v-for="m in MODULES" :key="m" type="button" role="tab" :aria-selected="module === m" @click="module = m">{{ m || '全部' }}</button>
        </div>
        <input v-model="kw" class="fld" type="search" placeholder="搜索操作人、操作或对象" aria-label="搜索审计日志" />
      </div>
      <div class="tbl-wrap pad">
        <table class="tbl">
          <thead><tr><th>时间</th><th>操作人</th><th>模块</th><th>操作</th><th>对象</th><th>IP</th><th>结果</th></tr></thead>
          <tbody>
            <tr v-for="r in shown" :key="r.id">
              <td class="muted nowrap">{{ r.at }}</td>
              <td class="nowrap">{{ r.user }}</td>
              <td class="nowrap">{{ r.module }}</td>
              <td>{{ r.action }}</td>
              <td class="target">{{ r.target }}</td>
              <td class="mono faint">{{ r.ip }}</td>
              <td><span class="pill" :class="RESULT[r.result][1]">{{ RESULT[r.result][0] }}</span></td>
            </tr>
          </tbody>
        </table>
        <p v-if="!shown.length" class="empty">没有符合条件的记录</p>
      </div>
    </section>
  </div>
</template>

<style scoped>
.pad { padding: 4px 20px 12px; }
.nowrap { white-space: nowrap; }
.target { color: var(--text-2); min-width: 220px; }
</style>
