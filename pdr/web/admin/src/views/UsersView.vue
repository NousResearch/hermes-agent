<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import { api, type RegionNode, type UserRow } from '../api'
import AppIcon from '../components/AppIcon.vue'
import { errorText, toast } from '../lib/toast'

const users = ref<UserRow[]>([])
const regions = ref<RegionNode[]>([])
const type = ref<UserRow['userType']>('admin')
const region = ref('')
const kw = ref('')

// 选中某个区域时，包含其全部下级区域
const regionSet = computed(() => {
  if (!region.value) return null
  const out = new Set<string>()
  const walk = (n: RegionNode, inside: boolean) => {
    const hit = inside || n.name === region.value
    if (hit) out.add(n.name)
    n.children?.forEach((c) => walk(c, hit))
  }
  regions.value.forEach((n) => walk(n, false))
  return out
})
const rows = computed(() => users.value.filter((u) =>
  u.userType === type.value && (!regionSet.value || regionSet.value.has(u.region)) &&
  (!kw.value.trim() || u.name.includes(kw.value.trim()) || u.account.includes(kw.value.trim()))))

const flat = computed(() => {
  const out: { name: string; level: number }[] = []
  const walk = (n: RegionNode, level: number) => {
    out.push({ name: n.name, level })
    n.children?.forEach((c) => walk(c, level + 1))
  }
  regions.value.forEach((n) => walk(n, 0))
  return out
})

async function flip(u: UserRow): Promise<void> {
  const next = u.status === 'active' ? 'disabled' : 'active'
  try {
    await api.setUserStatus(u.account, next)
    u.status = next
    toast(`${u.name} 已${next === 'active' ? '启用' : '停用'}`)
  } catch (e) {
    toast(errorText(e))
  }
}

onMounted(async () => {
  try {
    ;[users.value, regions.value] = await Promise.all([api.users(), api.regions()])
  } catch (e) {
    toast(errorText(e))
  }
})
</script>

<template>
  <div class="stack">
    <div class="page-head">
      <div>
        <h1>用户 · 区域</h1>
        <p>管理账号登录管理端，问答账号只能登录用户端，两类账号互不通用。每个账号归属一个区域，数据范围按区域展开。</p>
      </div>
    </div>

    <div class="cols">
      <aside class="card tree">
        <h2 class="sec">区域</h2>
        <button type="button" class="node" :aria-current="!region ? 'true' : undefined" @click="region = ''">
          <AppIcon name="building" :size="14" />示例旅行社（全部）
        </button>
        <button
          v-for="n in flat"
          :key="n.name"
          type="button"
          class="node"
          :style="{ paddingLeft: 12 + (n.level + 1) * 16 + 'px' }"
          :aria-current="region === n.name ? 'true' : undefined"
          @click="region = n.name"
        >
          <AppIcon name="pin" :size="13" />{{ n.name }}
        </button>
      </aside>

      <section class="card">
        <div class="card-head">
          <div class="tabs" role="tablist">
            <button type="button" role="tab" :aria-selected="type === 'admin'" @click="type = 'admin'">
              管理账号<span class="qty">{{ users.filter((u) => u.userType === 'admin').length }}</span>
            </button>
            <button type="button" role="tab" :aria-selected="type === 'portal'" @click="type = 'portal'">
              问答账号（用户端）<span class="qty">{{ users.filter((u) => u.userType === 'portal').length }}</span>
            </button>
          </div>
          <input v-model="kw" class="fld" type="search" placeholder="搜索姓名或账号" aria-label="搜索姓名或账号" />
        </div>
        <div class="tbl-wrap pad">
          <table class="tbl">
            <thead><tr><th>姓名</th><th>账号</th><th>角色</th><th>归属区域</th><th>手机</th><th>最近登录</th><th>状态</th></tr></thead>
            <tbody>
              <tr v-for="u in rows" :key="u.account" :class="{ off: u.status === 'disabled' }">
                <td><span class="row"><span class="avatar">{{ u.name.slice(0, 1) }}</span>{{ u.name }}</span></td>
                <td class="mono">{{ u.account }}</td>
                <td><span v-for="r in u.roleNames" :key="r" class="pill purple">{{ r }}</span></td>
                <td>{{ u.region }}</td>
                <td class="muted">{{ u.phone }}</td>
                <td class="muted">{{ u.lastLoginAt }}</td>
                <td>
                  <label class="row">
                    <span class="switch"><input type="checkbox" :checked="u.status === 'active'" :aria-label="`启用 ${u.name}`" @change="flip(u)" /><span /></span>
                    <span class="small">{{ u.status === 'active' ? '正常' : '已停用' }}</span>
                  </label>
                </td>
              </tr>
            </tbody>
          </table>
          <p v-if="!rows.length" class="empty">该区域下没有{{ type === 'admin' ? '管理' : '问答' }}账号</p>
        </div>
      </section>
    </div>
  </div>
</template>

<style scoped>
.cols { display: grid; grid-template-columns: 260px minmax(0, 1fr); gap: 16px; align-items: start; }
.tree { padding: 16px 10px; display: flex; flex-direction: column; gap: 2px; }
.tree .sec { margin: 0 8px 8px; }
.node { display: flex; align-items: center; gap: 6px; padding: 7px 12px; border-radius: 9px; border: 0; background: none; text-align: left; font-size: 13.5px; color: var(--text-2); }
.node:hover { background: #f6f5fd; }
.node[aria-current='true'] { background: var(--accent-soft); color: #3f2fc0; font-weight: 500; }
.pad { padding: 4px 20px 12px; }
tr.off td { color: var(--faint); }
.small { font-size: 12.5px; }
@media (max-width: 1000px) { .cols { grid-template-columns: 1fr; } }
</style>
