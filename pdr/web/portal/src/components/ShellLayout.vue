<script setup lang="ts">
import { onMounted, ref } from 'vue'
import { useRouter } from 'vue-router'
import { api } from '../api'
import { useAuth } from '../stores/auth'
import { useChat } from '../stores/chat'
import AppIcon from './AppIcon.vue'
import AppSidebar from './AppSidebar.vue'
import ScopeMenu from './ScopeMenu.vue'

defineProps<{ showScope?: boolean }>()
const auth = useAuth()
const chat = useChat()
const router = useRouter()
const sideOpen = ref(false)
const bootError = ref('')

onMounted(async () => {
  if (chat.scopes.length) return
  try {
    await chat.bootstrap()
  } catch (e) {
    bootError.value = (e as Error).message
  }
})

function newChat(): void {
  chat.reset()
  router.push('/')
}
</script>

<template>
  <div class="frame">
    <AppSidebar :open="sideOpen" @close="sideOpen = false" />
    <div v-if="sideOpen" class="scrim" aria-hidden="true" @click="sideOpen = false" />
    <main class="main">
      <div class="bg" aria-hidden="true" />
      <header class="top">
        <button class="icon-btn menu" type="button" aria-label="打开侧边栏" @click="sideOpen = true"><AppIcon name="menu" :size="18" /></button>
        <ScopeMenu v-if="showScope" />
        <span class="spacer" />
        <span v-if="api.mode === 'mock'" class="mock" title="后端未连接时使用内置示例数据；设置 VITE_PORTAL_MOCK=false 后连接真实接口">演示模式 · 示例数据</span>
        <button class="btn-ink" type="button" @click="newChat"><AppIcon name="plus" :size="15" :stroke="2" />新对话</button>
        <span class="avatar" :title="auth.user?.userName" aria-hidden="true">{{ (auth.user?.userName || '?').slice(0, 1) }}</span>
      </header>
      <p v-if="bootError" class="boot-error" role="alert">加载失败：{{ bootError }}</p>
      <slot />
    </main>
  </div>
</template>

<style scoped>
.frame {
  height: 100vh; display: flex; gap: 12px; padding: 12px; margin: 0 auto; max-width: 1680px;
  background: var(--frame);
}
.main {
  position: relative; flex: 1; min-width: 0; display: flex; flex-direction: column; overflow: hidden;
  background: var(--panel); border: 1px solid var(--line); border-radius: var(--radius);
}
.bg {
  position: absolute; inset: 0; pointer-events: none;
  background:
    radial-gradient(520px 260px at 50% 78%, rgba(150, 140, 255, 0.10), transparent 70%),
    radial-gradient(360px 200px at 66% 82%, rgba(140, 230, 190, 0.10), transparent 70%),
    radial-gradient(360px 200px at 34% 82%, rgba(255, 200, 230, 0.10), transparent 70%);
}
.top { position: relative; z-index: 2; display: flex; align-items: center; gap: 10px; padding: 16px 20px 8px; }
.spacer { flex: 1; }
.mock { font-size: 12px; color: #8a5a00; background: #fff3d6; border-radius: 6px; padding: 3px 8px; }
.menu { display: none; }
.avatar {
  width: 34px; height: 34px; border-radius: 10px; display: flex; align-items: center; justify-content: center;
  background: linear-gradient(135deg, #ffe1c7, #ffb98a); color: #6b3410; font-weight: 700; box-shadow: 0 1px 3px rgba(20, 20, 40, 0.1);
}
.boot-error { position: relative; z-index: 2; margin: 8px 20px; padding: 10px 12px; border-radius: 10px; background: #fdecef; color: var(--danger); font-size: 13px; }
.scrim { position: fixed; inset: 0; z-index: 20; background: rgba(20, 20, 30, 0.25); }
@media (max-width: 820px) {
  .frame { padding: 0; gap: 0; }
  .main { border-radius: 0; border: 0; }
  .menu { display: inline-flex; }
  .top { padding: 10px 12px 4px; }
  .mock { display: none; }
}
</style>
