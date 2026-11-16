<script setup lang="ts">
import { computed } from 'vue'
import { useRouter } from 'vue-router'
import { useChat } from '../stores/chat'
import ShellLayout from '../components/ShellLayout.vue'
import AppIcon from '../components/AppIcon.vue'

const chat = useChat()
const router = useRouter()

const groups = computed(() => {
  const map = new Map<string, typeof chat.questions>()
  for (const q of chat.questions) map.set(q.category, [...(map.get(q.category) || []), q])
  return [...map.entries()]
})

async function ask(text: string, scopeCode?: string): Promise<void> {
  chat.reset()
  if (scopeCode) chat.scopeCode = scopeCode
  const id = await chat.ask(text)
  if (id) router.push(`/c/${id}`)
}
</script>

<template>
  <ShellLayout>
    <div class="page">
      <h1>发现</h1>
      <p class="sub">常见问题，点一下就能直接提问</p>
      <section v-for="[cat, items] in groups" :key="cat" class="group">
        <h2>{{ cat }}</h2>
        <div class="cards">
          <button v-for="q in items" :key="q.id" class="card" type="button" @click="ask(q.text, q.scopeCode)">
            <span>{{ q.text }}</span>
            <AppIcon name="arrowRight" :size="15" />
          </button>
        </div>
      </section>
      <p v-if="!groups.length" class="sub">暂无推荐问题</p>
    </div>
  </ShellLayout>
</template>

<style scoped>
.page { position: relative; z-index: 1; flex: 1; overflow-y: auto; padding: 8px 28px 32px; max-width: 920px; width: 100%; margin: 0 auto; }
h1 { margin: 8px 0 4px; font-size: 24px; }
.sub { margin: 0 0 18px; color: var(--muted); font-size: 13.5px; }
.group h2 { margin: 18px 0 10px; font-size: 14px; color: var(--text-2); }
.cards { display: grid; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr)); gap: 10px; }
.card {
  display: flex; justify-content: space-between; align-items: center; gap: 10px; padding: 14px 16px; text-align: left;
  background: var(--surface); border: 1px solid var(--line); border-radius: 14px; font-size: 14px; color: var(--text);
  box-shadow: 0 1px 2px rgba(20, 20, 40, 0.04);
}
.card:hover { border-color: #cfcdf5; box-shadow: 0 0 0 3px #efeffd; }
.card :deep(svg) { color: var(--accent); flex: none; }
@media (max-width: 820px) { .page { padding: 8px 14px 24px; } }
</style>
