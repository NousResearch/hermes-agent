<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { useAuth } from '../stores/auth'
import { useChat } from '../stores/chat'
import { groupByTime } from '../lib/text'
import AppIcon from './AppIcon.vue'
import type { IconName } from '../lib/icons'

defineProps<{ open: boolean }>()
const emit = defineEmits<{ close: [] }>()

const auth = useAuth()
const chat = useChat()
const route = useRoute()
const router = useRouter()
const keyword = ref('')
const searchEl = ref<HTMLInputElement | null>(null)
const menuOpen = ref(false)

const NAV: { to: string; name: string; icon: IconName }[] = [
  { to: '/', name: '首页', icon: 'home' },
  { to: '/explore', name: '发现', icon: 'explore' },
  { to: '/saved', name: '收藏', icon: 'bookmark' },
  { to: '/history', name: '历史', icon: 'history' }
]

const groups = computed(() => {
  const k = keyword.value.trim()
  return groupByTime(k ? chat.sessions.filter((s) => s.title.includes(k)) : chat.sessions)
})
const initial = computed(() => (auth.user?.userName || '?').slice(0, 1))

function isActive(to: string): boolean {
  return to === '/' ? route.path === '/' : route.path.startsWith(to)
}

function onKey(e: KeyboardEvent): void {
  if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'k') {
    e.preventDefault()
    searchEl.value?.focus()
  }
}
onMounted(() => window.addEventListener('keydown', onKey))
onBeforeUnmount(() => window.removeEventListener('keydown', onKey))

async function logout(): Promise<void> {
  menuOpen.value = false
  chat.reset()
  await auth.logout()
  router.replace('/login')
}
</script>

<template>
  <aside class="sidebar" :class="{ open }" aria-label="侧边栏">
    <div class="brand">
      <span class="logo" aria-hidden="true">
        <svg width="22" height="22" viewBox="0 0 24 24"><path d="M4 7.5 12 3l8 4.5v9L12 21l-8-4.5z" fill="url(#lg)" /><path d="M8.5 12.2 11 14.6l4.6-5" fill="none" stroke="#fff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" /><defs><linearGradient id="lg" x1="4" y1="3" x2="20" y2="21"><stop stop-color="#8a7bf6" /><stop offset="1" stop-color="#4b4be0" /></linearGradient></defs></svg>
      </span>
      <span class="brand-name">{{ auth.user?.tenantName || 'PDR' }}</span>
      <button class="icon-btn close" type="button" aria-label="关闭侧边栏" @click="emit('close')"><AppIcon name="x" /></button>
    </div>

    <label class="search">
      <AppIcon name="search" :size="15" />
      <span class="sr-only">搜索历史会话</span>
      <input ref="searchEl" v-model="keyword" placeholder="搜索会话" />
      <kbd aria-hidden="true">⌘K</kbd>
    </label>

    <nav class="nav" aria-label="主导航">
      <RouterLink v-for="n in NAV" :key="n.to" :to="n.to" class="nav-item" :class="{ active: isActive(n.to) }" :aria-current="isActive(n.to) ? 'page' : undefined" @click="emit('close')">
        <AppIcon :name="n.icon" :size="17" />{{ n.name }}
      </RouterLink>
    </nav>

    <div class="recent">
      <section v-for="g in groups" :key="g.label" class="group">
        <h2>{{ g.label }}</h2>
        <RouterLink v-for="s in g.items" :key="s.id" :to="`/c/${s.id}`" class="recent-item" :class="{ active: route.params.id === s.id }" :title="s.title" @click="emit('close')">{{ s.title }}</RouterLink>
      </section>
      <p v-if="!groups.length" class="empty">{{ keyword ? '没有匹配的会话' : '还没有对话，去首页问第一个问题吧' }}</p>
    </div>

    <div class="user">
      <span class="avatar" aria-hidden="true">{{ initial }}</span>
      <span class="who">
        <b>{{ auth.user?.userName }}</b>
        <span>{{ auth.user?.account }}</span>
      </span>
      <button class="icon-btn" type="button" aria-label="账号菜单" :aria-expanded="menuOpen" @click="menuOpen = !menuOpen"><AppIcon name="upDown" :size="15" /></button>
      <div v-if="menuOpen" class="menu" role="menu">
        <button type="button" role="menuitem" @click="logout"><AppIcon name="logout" :size="15" />退出登录</button>
      </div>
    </div>
  </aside>
</template>

<style scoped>
.sidebar {
  width: 252px; flex: none; display: flex; flex-direction: column; gap: 6px; padding: 18px 14px 14px;
  background: var(--panel); border: 1px solid var(--line); border-radius: var(--radius);
}
.brand { display: flex; align-items: center; gap: 8px; padding: 0 6px 12px; }
.brand-name { font-size: 16px; font-weight: 700; letter-spacing: 0.5px; color: #24245a; }
.close { display: none; margin-left: auto; }
.search {
  display: flex; align-items: center; gap: 8px; height: 38px; padding: 0 8px 0 12px; margin-bottom: 10px;
  background: var(--surface); border: 1px solid var(--line); border-radius: 11px; color: var(--muted);
  box-shadow: 0 1px 2px rgba(20, 20, 40, 0.04), 0 0 0 3px #f0f0f4;
}
.search input { flex: 1; min-width: 0; border: 0; outline: none; background: transparent; font-size: 13px; color: var(--text); }
.search kbd { font: 11px/1 var(--font); padding: 4px 6px; border-radius: 6px; background: var(--accent-soft); color: var(--accent); }
.nav { display: flex; flex-direction: column; gap: 2px; }
.nav-item { display: flex; align-items: center; gap: 10px; height: 36px; padding: 0 10px; border-radius: 9px; color: var(--text-2); font-size: 14px; }
.nav-item:hover { background: #efeff4; color: var(--text); }
.nav-item.active { background: var(--surface); color: var(--text); font-weight: 500; box-shadow: 0 1px 3px rgba(20, 20, 40, 0.06); }
.recent { flex: 1; min-height: 0; overflow-y: auto; margin-top: 14px; padding-right: 2px; }
.group h2 { margin: 12px 4px 4px; font-size: 12px; font-weight: 500; color: var(--muted); }
.recent-item {
  display: block; padding: 7px 10px; border-radius: 9px; font-size: 13.5px; color: var(--text-2);
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
.recent-item:hover { background: #efeff4; color: var(--text); }
.recent-item.active { background: var(--surface); color: var(--text); box-shadow: 0 1px 3px rgba(20, 20, 40, 0.06); }
.empty { margin: 12px 6px; font-size: 12.5px; color: var(--muted); }
.user {
  position: relative; display: flex; align-items: center; gap: 10px; padding: 10px; margin-top: 8px;
  background: var(--surface); border: 1px solid var(--line); border-radius: 12px;
}
.avatar { width: 32px; height: 32px; border-radius: 50%; display: flex; align-items: center; justify-content: center; background: linear-gradient(135deg, #c9c4ff, #6d64f0); color: #fff; font-weight: 700; flex: none; }
.who { flex: 1; min-width: 0; display: flex; flex-direction: column; line-height: 1.35; }
.who b { font-size: 13px; font-weight: 600; }
.who span { font-size: 11.5px; color: var(--muted); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.menu {
  position: absolute; left: 0; right: 0; bottom: calc(100% + 6px); padding: 6px; background: var(--surface);
  border: 1px solid var(--line); border-radius: 12px; box-shadow: 0 12px 30px rgba(20, 20, 40, 0.12);
}
.menu button { display: flex; align-items: center; gap: 8px; width: 100%; height: 34px; padding: 0 10px; border: 0; border-radius: 8px; background: transparent; font-size: 13px; color: var(--danger); }
.menu button:hover { background: #fbeff1; }

@media (max-width: 820px) {
  .sidebar { position: fixed; z-index: 30; top: 8px; bottom: 8px; left: 8px; transform: translateX(calc(-100% - 16px)); transition: transform 0.2s ease; box-shadow: 0 20px 50px rgba(20, 20, 40, 0.2); }
  .sidebar.open { transform: none; }
  .close { display: inline-flex; }
}
</style>
