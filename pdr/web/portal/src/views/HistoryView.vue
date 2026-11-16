<script setup lang="ts">
import { computed, ref } from 'vue'
import { useChat } from '../stores/chat'
import { groupByTime } from '../lib/text'
import ShellLayout from '../components/ShellLayout.vue'
import AppIcon from '../components/AppIcon.vue'

const chat = useChat()
const keyword = ref('')
const pending = ref('')

const groups = computed(() => {
  const k = keyword.value.trim()
  return groupByTime(k ? chat.sessions.filter((s) => s.title.includes(k)) : chat.sessions)
})
const time = (iso: string) => new Date(iso).toLocaleString('zh-CN', { month: 'numeric', day: 'numeric', hour: '2-digit', minute: '2-digit' })

async function confirmDelete(id: string): Promise<void> {
  if (pending.value !== id) {
    pending.value = id
    return
  }
  pending.value = ''
  await chat.remove(id)
}
</script>

<template>
  <ShellLayout>
    <div class="page">
      <h1>历史</h1>
      <label class="search">
        <AppIcon name="search" :size="15" />
        <span class="sr-only">搜索历史会话</span>
        <input v-model="keyword" placeholder="按标题搜索" />
      </label>
      <section v-for="g in groups" :key="g.label">
        <h2>{{ g.label }}</h2>
        <div v-for="s in g.items" :key="s.id" class="row">
          <RouterLink :to="`/c/${s.id}`" class="title">{{ s.title }}</RouterLink>
          <span class="time">{{ time(s.updatedAt) }}</span>
          <button class="icon-btn del" type="button" :aria-label="pending === s.id ? `确认删除“${s.title}”` : `删除“${s.title}”`" @click="confirmDelete(s.id)">
            <span v-if="pending === s.id" class="confirm">确认删除</span>
            <AppIcon v-else name="trash" :size="15" />
          </button>
        </div>
      </section>
      <p v-if="!groups.length" class="empty">{{ keyword ? '没有匹配的会话' : '还没有历史会话' }}</p>
    </div>
  </ShellLayout>
</template>

<style scoped>
.page { position: relative; z-index: 1; flex: 1; overflow-y: auto; padding: 8px 28px 32px; max-width: 860px; width: 100%; margin: 0 auto; }
h1 { margin: 8px 0 14px; font-size: 24px; }
.search { display: flex; align-items: center; gap: 8px; height: 40px; padding: 0 12px; background: var(--surface); border: 1px solid var(--line); border-radius: 12px; color: var(--muted); }
.search input { flex: 1; border: 0; outline: none; background: transparent; color: var(--text); }
h2 { margin: 20px 0 6px; font-size: 12.5px; font-weight: 500; color: var(--muted); }
.row { display: flex; align-items: center; gap: 12px; padding: 4px 4px 4px 12px; border-radius: 10px; }
.row:hover { background: rgba(255, 255, 255, 0.8); }
.title { flex: 1; min-width: 0; padding: 8px 0; color: var(--text); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.time { font-size: 12px; color: var(--muted); }
.del { width: auto; min-width: 32px; padding: 0 6px; }
.confirm { font-size: 12px; color: var(--danger); }
.empty { margin: 40px 0; text-align: center; color: var(--muted); }
@media (max-width: 820px) { .page { padding: 8px 14px 24px; } .time { display: none; } }
</style>
