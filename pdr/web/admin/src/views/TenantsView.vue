<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import { api, type Tenant } from '../api'
import { errorText, toast } from '../lib/toast'

const tenants = ref<Tenant[]>([])
const STATUS: Record<Tenant['status'], [string, string]> = { active: ['正常', 'green'], trial: ['试用中', 'blue'], suspended: ['已暂停', 'pink'] }
const totals = computed(() => ({
  users: tenants.value.reduce((n, t) => n + t.userQty, 0),
  storage: tenants.value.reduce((n, t) => n + t.storageUsedGb, 0),
  active: tenants.value.filter((t) => t.status !== 'suspended').length
}))

onMounted(async () => {
  try {
    tenants.value = await api.tenants()
  } catch (e) {
    toast(errorText(e))
  }
})
</script>

<template>
  <div class="stack">
    <div class="page-head">
      <div>
        <h1>租户管理</h1>
        <p>平台级功能：每个租户的数据在数据库中按租户隔离（行级安全），智能体使用独立的 profile。</p>
      </div>
    </div>
    <div class="stats">
      <div class="card stat"><small>租户</small><b>{{ tenants.length }}</b><span class="muted">{{ totals.active }} 个在用</span></div>
      <div class="card stat"><small>账号总数</small><b>{{ totals.users }}</b></div>
      <div class="card stat"><small>已用存储</small><b>{{ totals.storage.toFixed(1) }} GB</b></div>
    </div>
    <section class="card">
      <div class="tbl-wrap pad">
        <table class="tbl">
          <thead><tr><th>租户</th><th>编码</th><th>套餐</th><th>管理员</th><th>账号数</th><th>存储用量</th><th>开通日期</th><th>状态</th></tr></thead>
          <tbody>
            <tr v-for="t in tenants" :key="t.code">
              <td><b>{{ t.name }}</b></td>
              <td class="mono">{{ t.code }}</td>
              <td>{{ t.plan }}</td>
              <td>{{ t.adminName }}</td>
              <td>{{ t.userQty }}</td>
              <td>
                <span class="usage"><i :style="{ width: (t.storageUsedGb / t.storageQuotaGb) * 100 + '%' }" /></span>
                <span class="muted small">{{ t.storageUsedGb }} / {{ t.storageQuotaGb }} GB</span>
              </td>
              <td class="muted">{{ t.createdAt }}</td>
              <td><span class="pill" :class="STATUS[t.status][1]">{{ STATUS[t.status][0] }}</span></td>
            </tr>
          </tbody>
        </table>
      </div>
    </section>
  </div>
</template>

<style scoped>
.stats { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 14px; }
.stat { padding: 16px 18px; display: flex; flex-direction: column; gap: 4px; }
.stat small { color: var(--muted); font-size: 12.5px; }
.stat b { font-size: 24px; }
.stat span { font-size: 12.5px; }
.pad { padding: 12px 20px; }
.usage { display: inline-block; width: 90px; height: 6px; border-radius: 999px; background: #eeedf7; overflow: hidden; vertical-align: middle; margin-right: 8px; }
.usage i { display: block; height: 100%; background: linear-gradient(90deg, #9e92fb, #5443d6); }
.small { font-size: 12.5px; }
</style>
