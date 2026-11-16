<script setup lang="ts">
import { onMounted, ref } from 'vue'
import { api, type SavedAnswer } from '../api'
import ShellLayout from '../components/ShellLayout.vue'

const items = ref<SavedAnswer[]>([])
const loading = ref(true)

onMounted(async () => {
  try {
    items.value = await api.savedAnswers()
  } finally {
    loading.value = false
  }
})

const plain = (s: string) => s.replace(/\[\d+\]/g, '').replace(/\*\*/g, '')
const day = (iso: string) => new Date(iso).toLocaleDateString('zh-CN', { month: 'long', day: 'numeric' })
</script>

<template>
  <ShellLayout>
    <div class="page">
      <h1>收藏</h1>
      <p class="sub">在回答下方点收藏图标，答案会出现在这里</p>
      <p v-if="loading" class="sub">加载中…</p>
      <p v-else-if="!items.length" class="empty">还没有收藏的回答</p>
      <RouterLink v-for="s in items" :key="s.messageId" :to="`/c/${s.sessionId}`" class="item">
        <span class="q">{{ s.question }}</span>
        <span class="a">{{ plain(s.answer) }}</span>
        <span class="d">收藏于 {{ day(s.savedAt) }}</span>
      </RouterLink>
    </div>
  </ShellLayout>
</template>

<style scoped>
.page { position: relative; z-index: 1; flex: 1; overflow-y: auto; padding: 8px 28px 32px; max-width: 860px; width: 100%; margin: 0 auto; display: flex; flex-direction: column; gap: 10px; }
h1 { margin: 8px 0 0; font-size: 24px; }
.sub { margin: 0 0 8px; color: var(--muted); font-size: 13.5px; }
.empty { margin: 40px 0; text-align: center; color: var(--muted); }
.item { display: flex; flex-direction: column; gap: 4px; padding: 14px 16px; background: var(--surface); border: 1px solid var(--line); border-radius: 14px; color: var(--text); }
.item:hover { border-color: #cfcdf5; }
.q { font-weight: 600; font-size: 14px; }
.a { font-size: 13.5px; color: var(--text-2); display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden; }
.d { font-size: 12px; color: var(--muted); }
@media (max-width: 820px) { .page { padding: 8px 14px 24px; } }
</style>
