<script setup lang="ts">
import { computed } from 'vue'
import { useChat } from '../stores/chat'
import AppIcon from './AppIcon.vue'

const chat = useChat()
const c = computed(() => chat.citation)
const isMedia = computed(() => c.value?.kind === 'video' || c.value?.kind === 'audio')
const KIND: Record<string, string> = { pdf: 'PDF 文档', word: 'Word 文档', excel: 'Excel 表格', markdown: 'Markdown 文档', image: '图片', video: '视频', audio: '音频' }
</script>

<template>
  <aside v-if="c" class="drawer" aria-label="出处预览">
    <header>
      <h2>出处 [{{ c.n }}]</h2>
      <button class="icon-btn" type="button" aria-label="关闭出处预览" @click="chat.citation = null"><AppIcon name="x" /></button>
    </header>
    <p class="title">{{ c.title }}</p>
    <p class="meta">{{ KIND[c.kind] }} · {{ c.location }}</p>
    <div v-if="isMedia" class="player">
      <AppIcon name="play" :size="22" />
      <span>播放 {{ c.location }} 片段</span>
    </div>
    <blockquote>{{ c.quote }}</blockquote>
    <p class="foot">只展示与问题相关的片段。</p>
  </aside>
</template>

<style scoped>
.drawer {
  width: 340px; flex: none; display: flex; flex-direction: column; gap: 10px; padding: 16px;
  background: var(--surface); border: 1px solid var(--line); border-radius: var(--radius); box-shadow: var(--shadow-card);
  align-self: flex-start; position: sticky; top: 0;
}
header { display: flex; justify-content: space-between; align-items: center; }
h2 { margin: 0; font-size: 15px; }
.title { margin: 0; font-size: 14px; font-weight: 600; }
.meta { margin: 0; font-size: 12.5px; color: var(--muted); }
.player {
  height: 160px; border-radius: 12px; background: radial-gradient(circle at 30% 30%, #3a3a52, #16161f); color: #e6e6f0;
  display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 8px; font-size: 12.5px;
}
blockquote { margin: 0; padding: 12px 14px; border-radius: 12px; background: #f5f5f9; font-size: 14px; line-height: 1.8; color: var(--text-2); }
.foot { margin: 0; font-size: 12px; color: var(--muted); }
@media (max-width: 1100px) {
  .drawer { position: fixed; z-index: 25; right: 12px; top: 12px; bottom: 12px; width: min(360px, calc(100vw - 24px)); box-shadow: 0 24px 60px rgba(20, 20, 40, 0.2); overflow-y: auto; }
}
</style>
