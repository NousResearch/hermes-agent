<script setup lang="ts">
import { computed, nextTick, onBeforeUnmount, ref } from 'vue'
import { useChat } from '../stores/chat'
import AppIcon from './AppIcon.vue'

const props = defineProps<{ variant: 'hero' | 'dock'; maxChars?: number }>()
const emit = defineEmits<{ send: [text: string] }>()

const chat = useChat()
const text = ref('')
const box = ref<HTMLTextAreaElement | null>(null)
const fileEl = ref<HTMLInputElement | null>(null)
const uploading = ref(false)
const listening = ref(false)
const limit = computed(() => props.maxChars ?? 500)
const canSend = computed(() => text.value.trim().length > 0 && !chat.busy)

interface SpeechRec {
  lang: string
  interimResults: boolean
  onresult: ((e: { results: ArrayLike<ArrayLike<{ transcript: string }>> }) => void) | null
  onend: (() => void) | null
  start(): void
  stop(): void
}
type SpeechCtor = new () => SpeechRec
const w = window as unknown as { SpeechRecognition?: SpeechCtor; webkitSpeechRecognition?: SpeechCtor }
const Speech = w.SpeechRecognition || w.webkitSpeechRecognition
let rec: SpeechRec | null = null

function autosize(): void {
  const el = box.value
  if (!el) return
  el.style.height = 'auto'
  el.style.height = `${Math.min(el.scrollHeight, props.variant === 'hero' ? 220 : 180)}px`
}

function submit(): void {
  if (!canSend.value) return
  emit('send', text.value.trim())
  text.value = ''
  nextTick(autosize)
}

function onKeydown(e: KeyboardEvent): void {
  if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) {
    e.preventDefault()
    submit()
  }
}

function toggleVoice(): void {
  if (!Speech) return
  if (listening.value) {
    rec?.stop()
    return
  }
  rec = new Speech()
  rec.lang = 'zh-CN'
  rec.interimResults = false
  rec.onresult = (e) => {
    text.value = (text.value + e.results[0][0].transcript).slice(0, limit.value)
    nextTick(autosize)
  }
  rec.onend = () => (listening.value = false)
  listening.value = true
  rec.start()
}
onBeforeUnmount(() => rec?.stop())

async function onFiles(e: Event): Promise<void> {
  const files = Array.from((e.target as HTMLInputElement).files || [])
  uploading.value = true
  try {
    for (const f of files) await chat.attach(f)
  } finally {
    uploading.value = false
    if (fileEl.value) fileEl.value.value = ''
  }
}

defineExpose({ focus: () => box.value?.focus(), fill: (v: string) => { text.value = v; nextTick(() => { autosize(); box.value?.focus() }) } })
</script>

<template>
  <form class="composer" :class="variant" @submit.prevent="submit">
    <div class="input-row">
      <AppIcon name="sparkle" :size="16" class="spark" />
      <label class="sr-only" for="composer-input">输入问题</label>
      <textarea
        id="composer-input"
        ref="box"
        v-model="text"
        :maxlength="limit"
        :rows="variant === 'hero' ? 3 : 1"
        placeholder="输入问题，或描述你想查找的资料…"
        @input="autosize"
        @keydown="onKeydown"
      />
    </div>

    <ul v-if="chat.attachments.length" class="files" aria-label="本次对话的附件">
      <li v-for="a in chat.attachments" :key="a.id"><AppIcon name="file" :size="13" />{{ a.name }}</li>
    </ul>

    <div class="tools">
      <input ref="fileEl" class="sr-only" type="file" multiple tabindex="-1" aria-hidden="true" @change="onFiles" />
      <button class="chip square" type="button" :aria-label="uploading ? '正在上传附件' : '添加附件（仅本次对话有效）'" :title="'添加附件（仅本次对话有效）'" @click="fileEl?.click()">
        <AppIcon name="paperclip" :size="14" />
      </button>
      <button class="chip" type="button" :aria-pressed="chat.deep" title="召回更多资料，回答更完整，耗时稍长" @click="chat.deep = !chat.deep">
        <AppIcon name="layers" :size="14" />深度检索
      </button>
      <span class="chip static" :title="chat.scope?.desc">
        <AppIcon name="book" :size="14" />{{ chat.scope?.name || '全部可用资料' }}
      </span>
      <span class="count" aria-live="polite">{{ text.length }}/{{ limit }}</span>
      <button v-if="chat.busy" class="action stop" type="button" aria-label="停止生成" @click="chat.stop()"><AppIcon name="stop" :size="14" /></button>
      <button v-else-if="text.trim()" class="action" type="submit" aria-label="发送" :disabled="!canSend"><AppIcon name="send" :size="16" :stroke="2.2" /></button>
      <button
        v-else
        class="action"
        type="button"
        :aria-label="listening ? '停止语音输入' : '语音输入'"
        :aria-pressed="listening"
        :disabled="!Speech"
        :title="Speech ? '语音输入' : '当前浏览器不支持语音输入'"
        @click="toggleVoice"
      >
        <AppIcon name="mic" :size="16" />
      </button>
    </div>
  </form>
</template>

<style scoped>
.composer {
  width: 100%; display: flex; flex-direction: column; gap: 10px; padding: 16px 16px 12px;
  background: rgba(255, 255, 255, 0.92); border: 1px solid var(--line); border-radius: 18px; box-shadow: var(--shadow-card);
  backdrop-filter: blur(8px);
}
.composer.hero { min-height: 164px; }
.composer:focus-within { border-color: #cfcdf5; }
.input-row { flex: 1; display: flex; gap: 10px; align-items: flex-start; }
.spark { color: var(--accent); margin-top: 3px; flex: none; }
textarea { flex: 1; resize: none; border: 0; outline: none; background: transparent; font-size: 14.5px; line-height: 1.6; color: var(--text); }
textarea::placeholder { color: #8a8a97; }
.files { display: flex; flex-wrap: wrap; gap: 6px; margin: 0; padding: 0 0 0 26px; list-style: none; }
.files li { display: inline-flex; align-items: center; gap: 4px; padding: 2px 8px; border-radius: 6px; background: #f2f2f6; font-size: 12px; color: var(--text-2); }
.tools { display: flex; align-items: center; gap: 6px; flex-wrap: wrap; }
.chip.square { width: 30px; padding: 0; justify-content: center; }
.chip.static { cursor: default; }
.count { margin-left: auto; font-size: 12px; color: var(--muted); }
.action {
  width: 34px; height: 34px; border: 0; border-radius: 10px; display: inline-flex; align-items: center; justify-content: center;
  color: #fff; background: linear-gradient(160deg, #8a86ff, #5b5bea);
  box-shadow: 0 1px 0 rgba(255, 255, 255, 0.35) inset, 0 6px 14px rgba(91, 91, 234, 0.35);
}
.action:disabled { opacity: 0.45; cursor: not-allowed; box-shadow: none; }
.action[aria-pressed='true'] { background: linear-gradient(160deg, #ff8fa3, #e5486a); }
.action.stop { background: var(--ink); box-shadow: none; }
</style>
