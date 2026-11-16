<script setup lang="ts">
import { onBeforeUnmount, onMounted, ref } from 'vue'
import { useChat } from '../stores/chat'
import AppIcon from './AppIcon.vue'
import GlowOrb from './GlowOrb.vue'

const chat = useChat()
const open = ref(false)
const root = ref<HTMLElement | null>(null)

function pick(code: string): void {
  chat.scopeCode = code
  open.value = false
}
function onDoc(e: MouseEvent): void {
  if (root.value && !root.value.contains(e.target as Node)) open.value = false
}
onMounted(() => document.addEventListener('click', onDoc))
onBeforeUnmount(() => document.removeEventListener('click', onDoc))
</script>

<template>
  <div ref="root" class="scope">
    <button class="trigger" type="button" aria-haspopup="listbox" :aria-expanded="open" @click="open = !open">
      <GlowOrb :size="20" />
      <span>{{ chat.scope?.name || '资料范围' }}</span>
      <AppIcon name="chevronDown" :size="15" />
    </button>
    <ul v-if="open" class="list" role="listbox" aria-label="资料范围">
      <li v-for="s in chat.scopes" :key="s.code" role="option" :aria-selected="s.code === chat.scopeCode">
        <button type="button" @click="pick(s.code)">
          <span class="name">{{ s.name }}<AppIcon v-if="s.code === chat.scopeCode" name="check" :size="14" /></span>
          <span class="desc">{{ s.desc }}</span>
        </button>
      </li>
    </ul>
  </div>
</template>

<style scoped>
.scope { position: relative; }
.trigger {
  display: inline-flex; align-items: center; gap: 8px; height: 36px; padding: 0 10px 0 8px;
  background: var(--surface); border: 1px solid var(--line); border-radius: 10px; font-size: 13.5px; font-weight: 500;
  box-shadow: 0 1px 2px rgba(20, 20, 40, 0.05);
}
.trigger:hover { border-color: var(--line-strong); }
.list {
  position: absolute; z-index: 20; top: calc(100% + 6px); left: 0; width: 260px; margin: 0; padding: 6px; list-style: none;
  background: var(--surface); border: 1px solid var(--line); border-radius: 12px; box-shadow: 0 16px 40px rgba(20, 20, 40, 0.12);
}
.list button { display: flex; flex-direction: column; align-items: flex-start; width: 100%; padding: 8px 10px; border: 0; border-radius: 8px; background: transparent; text-align: left; }
.list button:hover { background: #f3f3f7; }
.name { display: flex; align-items: center; gap: 6px; font-size: 13.5px; font-weight: 500; color: var(--text); }
.name :deep(svg) { color: var(--accent); }
.desc { font-size: 12px; color: var(--muted); }
</style>
