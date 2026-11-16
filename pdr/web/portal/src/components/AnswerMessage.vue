<script setup lang="ts">
import { computed, ref } from 'vue'
import type { ChatMessage, Citation } from '../api'
import { useChat } from '../stores/chat'
import { toBlocks } from '../lib/text'
import AppIcon from './AppIcon.vue'
import GlowOrb from './GlowOrb.vue'

const props = defineProps<{ message: ChatMessage; last: boolean }>()
const emit = defineEmits<{ followUp: [text: string] }>()
const chat = useChat()
const copied = ref(false)

const blocks = computed(() => toBlocks(props.message.content))
const working = computed(() => props.message.status === 'searching' || props.message.status === 'streaming')
const KIND: Record<Citation['kind'], string> = { pdf: 'PDF', word: 'Word', excel: 'Excel', markdown: 'Markdown', image: '图片', video: '视频', audio: '音频' }

function open(n: number): void {
  chat.citation = props.message.citations?.find((c) => c.n === n) || null
}

async function copy(): Promise<void> {
  try {
    await navigator.clipboard.writeText(props.message.content.replace(/\[\d+\]/g, '').replace(/\*\*/g, ''))
    copied.value = true
    setTimeout(() => (copied.value = false), 1500)
  } catch {
    copied.value = false
  }
}
</script>

<template>
  <article class="answer" :aria-busy="working">
    <GlowOrb :size="28" />
    <div class="body">
      <p class="status">
        <template v-if="message.status === 'searching'"><span class="dots" aria-hidden="true"><i /><i /><i /></span>正在检索资料…</template>
        <template v-else-if="message.retrievedQty">已检索 {{ message.retrievedQty }} 份资料</template>
        <template v-else-if="message.status !== 'error'">未检索到相关资料</template>
      </p>

      <div class="content" :class="{ error: message.status === 'error' }">
        <template v-for="(b, i) in blocks" :key="i">
          <p :class="b.kind">
            <span v-if="b.kind === 'li'" class="marker">{{ b.marker }}</span>
            <span class="line">
              <template v-for="(p, j) in b.parts" :key="j">
                <strong v-if="p.t === 'bold'">{{ p.v }}</strong>
                <button v-else-if="p.t === 'cite'" class="cite" type="button" :aria-label="`查看出处 ${p.n}`" @click="open(p.n)">{{ p.n }}</button>
                <template v-else>{{ p.v }}</template>
              </template>
              <span v-if="working && i === blocks.length - 1" class="caret" aria-hidden="true" />
            </span>
          </p>
        </template>
        <p v-if="message.status === 'stopped'" class="note">已停止生成</p>
      </div>

      <div v-if="message.citations?.length" class="sources">
        <button v-for="c in message.citations" :key="c.n" class="source" :class="{ on: chat.citation?.chunkNo === c.chunkNo }" type="button" @click="open(c.n)">
          <span class="src-kind">[{{ c.n }}] {{ KIND[c.kind] }}</span>
          <span class="src-title">{{ c.title }}</span>
          <span class="src-loc">{{ c.location }}</span>
        </button>
      </div>

      <div v-if="!working" class="actions">
        <button class="icon-btn" type="button" :aria-label="copied ? '已复制' : '复制回答'" :title="copied ? '已复制' : '复制'" @click="copy"><AppIcon :name="copied ? 'check' : 'copy'" :size="15" /></button>
        <button class="icon-btn" type="button" aria-label="重新回答" title="重新回答" :disabled="chat.busy" @click="chat.regenerate(message.id)"><AppIcon name="refresh" :size="15" /></button>
        <button class="icon-btn" type="button" aria-label="回答有帮助" title="有帮助" :aria-pressed="message.feedback === 1" @click="chat.setFeedback(message, 1)"><AppIcon name="thumbUp" :size="15" /></button>
        <button class="icon-btn" type="button" aria-label="回答没帮助" title="没帮助" :aria-pressed="message.feedback === -1" @click="chat.setFeedback(message, -1)"><AppIcon name="thumbDown" :size="15" /></button>
        <button class="icon-btn" type="button" :aria-label="message.saved ? '取消收藏' : '收藏回答'" :title="message.saved ? '取消收藏' : '收藏'" :aria-pressed="!!message.saved" @click="chat.toggleSaved(message)"><AppIcon name="bookmark" :size="15" /></button>
        <span v-if="message.feedback === -1" class="hint">已记录，我们会改进这条回答</span>
      </div>

      <div v-if="last && !working && message.followUps?.length" class="follow">
        <button v-for="f in message.followUps" :key="f" class="chip" type="button" @click="emit('followUp', f)">{{ f }}<AppIcon name="arrowRight" :size="13" /></button>
      </div>
    </div>
  </article>
</template>

<style scoped>
.answer { display: flex; gap: 14px; align-items: flex-start; }
.answer > :first-child { margin-top: 2px; }
.body { flex: 1; min-width: 0; display: flex; flex-direction: column; gap: 10px; }
.status { margin: 4px 0 0; font-size: 12.5px; color: var(--muted); display: flex; align-items: center; gap: 8px; }
.dots { display: inline-flex; gap: 3px; }
.dots i { width: 5px; height: 5px; border-radius: 50%; background: var(--accent); opacity: 0.4; animation: blink 1.2s infinite; }
.dots i:nth-child(2) { animation-delay: 0.2s; }
.dots i:nth-child(3) { animation-delay: 0.4s; }
@keyframes blink { 40% { opacity: 1; } }
.content { font-size: 15px; line-height: 1.85; color: var(--text); }
.content p { margin: 0 0 6px; }
.content p.li { display: flex; gap: 6px; padding-left: 4px; }
.marker { flex: none; color: var(--muted); }
.line { flex: 1; min-width: 0; }
.content.error { color: var(--danger); }
.note { font-size: 12.5px; color: var(--muted); }
.cite {
  display: inline-flex; align-items: center; justify-content: center; min-width: 18px; height: 18px; padding: 0 4px; margin: 0 2px;
  border: 0; border-radius: 5px; background: var(--accent-soft); color: var(--accent); font-size: 11px; font-weight: 600; vertical-align: 2px;
}
.cite:hover { background: #dedcfd; }
.caret { display: inline-block; width: 7px; height: 15px; margin-left: 2px; vertical-align: -2px; border-radius: 2px; background: var(--accent); animation: blink 1s infinite; }
.sources { display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 8px; }
.source {
  display: flex; flex-direction: column; align-items: flex-start; gap: 2px; min-width: 0; padding: 10px 12px; text-align: left;
  background: var(--surface); border: 1px solid var(--line); border-radius: 12px;
}
.source:hover, .source.on { border-color: #c9c8f7; box-shadow: 0 0 0 3px #efeffd; }
.src-kind { font-size: 11.5px; color: var(--accent); font-weight: 500; }
.src-title { width: 100%; font-size: 13px; color: var(--text); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.src-loc { font-size: 12px; color: var(--muted); }
.actions { display: flex; align-items: center; gap: 2px; }
.actions .icon-btn:disabled { opacity: 0.4; cursor: not-allowed; }
.hint { margin-left: 8px; font-size: 12px; color: var(--muted); }
.follow { display: flex; flex-wrap: wrap; gap: 8px; }
.follow .chip { height: 32px; font-size: 13px; }
</style>
