<script setup lang="ts">
import { computed, onMounted, ref, watch } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { api, type Notice } from '../api'
import { moduleOf, TINTS, visibleModules } from '../lib/nav'
import { toastText } from '../lib/toast'
import { useAuth } from '../stores/auth'
import AppIcon from './AppIcon.vue'

const auth = useAuth()
const route = useRoute()
const router = useRouter()

const collapsed = ref(false)
const keyword = ref('')
const notices = ref<Notice[]>([])
const bellOpen = ref(false)
const userOpen = ref(false)

const modules = computed(() => visibleModules(auth.can))
const current = computed(() => {
  const key = moduleOf(route.path)?.key
  return modules.value.find((m) => m.key === key)
})
const subPages = computed(() => (current.value && current.value.pages.length > 1 ? current.value.pages : []))
const unread = computed(() => notices.value.filter((n) => !n.read).length)
const initial = computed(() => (auth.user?.name || '?').slice(0, 1))
const greetingName = computed(() => auth.user?.name || '')

function tileStyle(tint: keyof typeof TINTS, on: boolean): string {
  if (on) return 'background:linear-gradient(135deg,#8E80FA,#5443D6);color:#fff;box-shadow:0 4px 10px rgba(84,67,214,0.35)'
  const [bg, fg] = TINTS[tint]
  return `background:${bg};color:${fg}`
}

function search(): void {
  const kw = keyword.value.trim()
  if (!kw) return
  if (auth.can('content:asset')) router.push({ path: '/content/tasks', query: { keyword: kw } })
}

function openNotice(n: Notice): void {
  n.read = true
  bellOpen.value = false
  if (n.link) router.push(n.link)
}

async function logout(): Promise<void> {
  userOpen.value = false
  await auth.logout()
  router.replace('/login')
}

function closeMenus(e: MouseEvent): void {
  const t = e.target as HTMLElement
  if (!t.closest('.menu-anchor')) {
    bellOpen.value = false
    userOpen.value = false
  }
}

onMounted(async () => {
  document.addEventListener('click', closeMenus)
  try {
    notices.value = await api.notices()
  } catch {
    notices.value = []
  }
})
watch(() => route.path, () => {
  bellOpen.value = false
  userOpen.value = false
})
</script>

<template>
  <div class="shell" :class="{ collapsed }">
    <nav class="side" aria-label="主导航">
      <div class="brand">
        <span class="logo"><AppIcon name="check" :size="18" :stroke="2.4" /></span>
        <span class="brand-text"><b>PDR</b><small>管理端</small></span>
      </div>
      <RouterLink
        v-for="m in modules"
        :key="m.key"
        :to="m.pages[0].path"
        class="side-a"
        :class="{ on: current?.key === m.key }"
        :title="m.name"
        :aria-current="current?.key === m.key ? 'page' : undefined"
      >
        <span class="sico" :style="tileStyle(m.tint, current?.key === m.key)"><AppIcon :name="m.icon" :size="16" /></span>
        <span class="side-label">{{ m.name }}</span>
      </RouterLink>

      <div class="side-foot">
        <div v-if="auth.can('content:dashboard')" class="usage">
          <span class="usage-ico"><AppIcon name="database" :size="18" /></span>
          <b>本月用量</b>
          <p>存储 38 / 100 GB<br />解析 12,480 页 · 音视频 640 分钟</p>
          <div class="bar"><i style="width: 38%" /></div>
          <RouterLink to="/content/board" class="usage-btn">查看用量明细</RouterLink>
        </div>
        <button type="button" class="collapse" @click="collapsed = !collapsed">
          <AppIcon :name="collapsed ? 'expand' : 'collapse'" :size="14" />
          <span class="side-label">收起侧栏</span>
        </button>
      </div>
    </nav>

    <main class="main">
      <header class="top">
        <label class="search">
          <AppIcon name="search" :size="16" />
          <span class="sr-only">搜索</span>
          <input v-model="keyword" type="search" placeholder="搜索资产编号或文件名…" @keydown.enter="search" />
        </label>
        <div class="top-right">
          <span v-if="api.mode === 'mock'" class="demo">演示模式 · 示例数据</span>
          <div class="menu-anchor">
            <button type="button" class="ib bell" :aria-expanded="bellOpen" aria-label="通知" @click="bellOpen = !bellOpen; userOpen = false">
              <AppIcon name="bell" :size="18" />
              <span v-if="unread" class="badge">{{ unread }}</span>
            </button>
            <div v-if="bellOpen" class="menu notices" role="menu">
              <p class="menu-title">通知</p>
              <button v-for="n in notices" :key="n.id" type="button" class="notice-row" :class="{ read: n.read }" role="menuitem" @click="openNotice(n)">
                <span class="dot" />
                <span class="grow"><span class="nt">{{ n.title }}</span><small>{{ n.at }}</small></span>
              </button>
              <p v-if="!notices.length" class="empty">暂无通知</p>
            </div>
          </div>
          <span class="divider" />
          <div class="menu-anchor">
            <button type="button" class="me" :aria-expanded="userOpen" @click="userOpen = !userOpen; bellOpen = false">
              <span class="me-avatar">{{ initial }}</span>
              <span class="me-text"><b>{{ greetingName }}</b><small>{{ auth.user?.roleName }}</small></span>
              <AppIcon name="chevronDown" :size="14" />
            </button>
            <div v-if="userOpen" class="menu user-menu" role="menu">
              <div class="me-card">
                <b>{{ auth.user?.name }}</b>
                <small>{{ auth.user?.tenantName }} · {{ auth.user?.roleName }}</small>
                <small>数据范围：{{ auth.user?.dataScopeText }}</small>
              </div>
              <button type="button" class="menu-item" role="menuitem" @click="logout">
                <AppIcon name="logout" :size="15" />{{ api.mode === 'mock' ? '退出（可换一个演示账号）' : '退出登录' }}
              </button>
            </div>
          </div>
        </div>
      </header>

      <div v-if="subPages.length" class="subnav tabs" role="tablist">
        <RouterLink
          v-for="p in subPages"
          :key="p.path"
          :to="p.path"
          role="tab"
          :aria-selected="route.path === p.path"
          class="subtab"
        >{{ p.name }}</RouterLink>
      </div>

      <div class="content">
        <slot />
      </div>
    </main>

    <div v-if="toastText" class="toast" role="status">{{ toastText }}</div>
  </div>
</template>

<style scoped>
.shell { display: flex; gap: 16px; min-height: 100vh; padding: 16px; }
.side {
  flex: none; width: 232px; padding: 20px 14px 14px; border-radius: 22px;
  background: linear-gradient(180deg, var(--nav-top), var(--nav-bottom));
  box-shadow: var(--shadow-lg), inset 0 1px 0 #fff; display: flex; flex-direction: column; gap: 6px;
  position: sticky; top: 16px; height: calc(100vh - 32px); overflow-y: auto; transition: width 0.18s;
}
.brand { display: flex; align-items: center; gap: 10px; padding: 0 8px 18px; }
.logo {
  width: 36px; height: 36px; border-radius: 11px; display: flex; align-items: center; justify-content: center; color: #fff; flex: none;
  background: linear-gradient(135deg, #8e80fa, #5443d6); box-shadow: 0 6px 14px rgba(84, 67, 214, 0.35);
}
.brand-text { display: flex; flex-direction: column; line-height: 1.2; }
.brand-text b { font-size: 17px; color: var(--accent); letter-spacing: 0.5px; }
.brand-text small { font-size: 12px; color: var(--muted); }
.side-a { display: flex; align-items: center; gap: 12px; padding: 8px 12px; border-radius: 14px; color: var(--text-2); font-size: 14px; }
.side-a:hover { background: rgba(255, 255, 255, 0.75); color: var(--text-2); }
.side-a.on { background: #fff; color: var(--text); font-weight: 500; box-shadow: 0 6px 16px rgba(108, 99, 190, 0.18); }
.sico { width: 30px; height: 30px; border-radius: 9px; display: flex; align-items: center; justify-content: center; flex: none; }
.side-foot { margin-top: auto; display: flex; flex-direction: column; gap: 12px; padding-top: 16px; }
.usage {
  background: #fff; border-radius: 18px; padding: 16px 14px; text-align: center; box-shadow: 0 8px 20px rgba(108, 99, 190, 0.12);
  display: flex; flex-direction: column; align-items: center; gap: 6px;
}
.usage-ico {
  width: 44px; height: 44px; border-radius: 13px; display: flex; align-items: center; justify-content: center; color: #fff;
  background: linear-gradient(135deg, #ffd66b, #f2b33d); box-shadow: 0 6px 14px rgba(242, 179, 61, 0.35);
}
.usage b { font-size: 14px; }
.usage p { margin: 0; font-size: 12px; color: var(--muted); line-height: 1.6; }
.bar { width: 100%; height: 6px; border-radius: 999px; background: #eeedf7; overflow: hidden; margin: 4px 0; }
.bar i { display: block; height: 100%; border-radius: inherit; background: linear-gradient(90deg, #8e80fa, #5443d6); }
.usage-btn {
  margin-top: 2px; padding: 7px 14px; border-radius: 10px; font-size: 13px; color: #fff;
  background: linear-gradient(135deg, var(--accent-2), var(--accent)); box-shadow: 0 6px 14px rgba(84, 67, 214, 0.3);
}
.usage-btn:hover { color: #fff; }
.collapse { display: flex; align-items: center; gap: 6px; border: 0; background: none; color: var(--muted); font-size: 12.5px; padding: 4px 10px; }
.collapse:hover { color: var(--text); }

.shell.collapsed .side { width: 78px; }
.shell.collapsed .side-label, .shell.collapsed .brand-text, .shell.collapsed .usage { display: none; }
.shell.collapsed .side-a { justify-content: center; padding: 8px; }

.main {
  flex: 1; min-width: 0; background: rgba(255, 255, 255, 0.55); border-radius: 26px; padding: 20px 24px 32px;
  box-shadow: inset 0 1px 0 #fff, 0 10px 30px rgba(108, 99, 190, 0.08); border: 1px solid rgba(255, 255, 255, 0.8);
}
.top { display: flex; align-items: center; justify-content: space-between; gap: 16px; margin-bottom: 14px; }
.search {
  display: flex; align-items: center; gap: 10px; width: min(420px, 100%); height: 44px; padding: 0 16px; border-radius: 14px;
  background: #fff; color: var(--faint); box-shadow: 0 4px 14px rgba(108, 99, 190, 0.08);
}
.search input { flex: 1; border: 0; outline: none; background: none; font-size: 13.5px; color: var(--text); min-width: 0; }
.top-right { display: flex; align-items: center; gap: 12px; }
.demo { font-size: 12px; color: var(--warn); background: var(--warn-soft); border-radius: 8px; padding: 3px 9px; white-space: nowrap; }
.menu-anchor { position: relative; }
.bell { position: relative; width: 38px; height: 38px; }
.badge {
  position: absolute; top: 2px; right: 1px; min-width: 16px; height: 16px; padding: 0 4px; border-radius: 999px;
  background: #f0607c; color: #fff; font-size: 10.5px; line-height: 16px; text-align: center; box-shadow: 0 0 0 2px #fff;
}
.divider { width: 1px; height: 28px; background: var(--line); }
.me { display: flex; align-items: center; gap: 10px; border: 0; background: none; padding: 2px; color: var(--muted); }
.me-avatar {
  width: 40px; height: 40px; border-radius: 50%; display: flex; align-items: center; justify-content: center; color: #fff;
  font-weight: 600; background: linear-gradient(135deg, #b9b0fb, #7b6cf6); box-shadow: 0 4px 10px rgba(84, 67, 214, 0.3);
}
.me-text { display: flex; flex-direction: column; align-items: flex-start; line-height: 1.3; }
.me-text b { font-size: 14px; color: var(--text); }
.me-text small { font-size: 12px; color: var(--muted); }
.menu {
  position: absolute; right: 0; top: calc(100% + 8px); z-index: 30; background: #fff; border: 1px solid var(--line);
  border-radius: 14px; box-shadow: var(--shadow-lg); padding: 8px; min-width: 240px;
}
.menu-title { margin: 4px 8px 6px; font-size: 12.5px; color: var(--muted); }
.notices { width: 340px; }
.notice-row { display: flex; gap: 10px; width: 100%; text-align: left; border: 0; background: none; padding: 8px; border-radius: 10px; }
.notice-row:hover { background: #f6f5fd; }
.notice-row .dot { flex: none; width: 7px; height: 7px; margin-top: 7px; border-radius: 50%; background: #f0607c; }
.notice-row.read .dot { background: #d9d6ec; }
.notice-row .nt { display: block; font-size: 13px; color: var(--text); }
.notice-row small { color: var(--faint); font-size: 12px; }
.me-card { display: flex; flex-direction: column; gap: 2px; padding: 8px 10px 10px; border-bottom: 1px solid var(--line); margin-bottom: 6px; }
.me-card small { color: var(--muted); font-size: 12px; }
.menu-item { display: flex; align-items: center; gap: 8px; width: 100%; border: 0; background: none; padding: 8px 10px; border-radius: 10px; font-size: 13px; text-align: left; }
.menu-item:hover { background: #f6f5fd; }
.subnav { margin-bottom: 18px; }
.subtab { padding: 10px 0; font-size: 13.5px; color: #4a4e6a; white-space: nowrap; border-bottom: 2px solid transparent; margin-bottom: -1px; }
.subtab[aria-selected='true'] { color: var(--accent); font-weight: 500; border-bottom-color: var(--accent); }
.content { min-width: 0; }

@media (max-width: 1100px) {
  .side { width: 78px; }
  .side-label, .brand-text, .usage { display: none; }
  .side-a { justify-content: center; padding: 8px; }
  .collapse { display: none; }
}
@media (max-width: 720px) {
  .shell { flex-direction: column; padding: 10px; gap: 10px; }
  .side { position: static; width: auto; height: auto; flex-direction: row; overflow-x: auto; padding: 10px; border-radius: 18px; }
  .brand, .side-foot { display: none; }
  .main { padding: 14px 14px 24px; border-radius: 20px; }
  .demo, .me-text, .divider { display: none; }
  .notices { width: min(340px, calc(100vw - 40px)); }
}
</style>
