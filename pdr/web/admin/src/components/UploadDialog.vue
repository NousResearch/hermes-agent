<script setup lang="ts">
import { ref } from 'vue'
import { api } from '../api'
import { errorText, toast } from '../lib/toast'
import AppIcon from './AppIcon.vue'

const emit = defineEmits<{ close: []; done: [] }>()

const files = ref<{ name: string; size: number }[]>([])
const over = ref(false)
const busy = ref(false)
const ACCEPT = '.pdf,.doc,.docx,.xls,.xlsx,.csv,.md,.txt,.png,.jpg,.jpeg,.webp,.mp4,.mov,.mp3,.wav,.m4a'

function add(list: FileList | null): void {
  if (!list) return
  for (const f of Array.from(list)) {
    if (!files.value.some((x) => x.name === f.name)) files.value.push({ name: f.name, size: f.size })
  }
}

function drop(e: DragEvent): void {
  over.value = false
  add(e.dataTransfer?.files ?? null)
}

async function submit(): Promise<void> {
  busy.value = true
  try {
    const created = await api.upload(files.value)
    toast(`已提交 ${created.length} 个文件，进入解析队列`)
    emit('done')
  } catch (e) {
    toast(errorText(e))
  } finally {
    busy.value = false
  }
}

const sizeText = (n: number) => (n > 1 << 20 ? (n / (1 << 20)).toFixed(1) + ' MB' : Math.max(1, Math.round(n / 1024)) + ' KB')
</script>

<template>
  <div class="mask" @click.self="emit('close')">
    <section class="dialog card" role="dialog" aria-modal="true" aria-labelledby="upload-title">
      <header class="card-head">
        <h2 id="upload-title" class="sec">上传文件</h2>
        <button type="button" class="ib" aria-label="关闭" @click="emit('close')"><AppIcon name="x" /></button>
      </header>
      <div class="body">
        <label class="drop" :class="{ over }" @dragover.prevent="over = true" @dragleave="over = false" @drop.prevent="drop">
          <AppIcon name="upload" :size="26" />
          <b>拖入文件，或点击选择</b>
          <small>支持 PDF、Word、Excel、Markdown、图片、视频、音频；上传后自动进入 MinerU 解析</small>
          <input type="file" multiple :accept="ACCEPT" class="sr-only" @change="add(($event.target as HTMLInputElement).files)" />
        </label>
        <ul v-if="files.length" class="files">
          <li v-for="(f, i) in files" :key="f.name">
            <AppIcon name="file" />
            <span class="grow ellipsis">{{ f.name }}</span>
            <small class="muted">{{ sizeText(f.size) }}</small>
            <button type="button" class="ib" :aria-label="`移除 ${f.name}`" @click="files.splice(i, 1)"><AppIcon name="x" :size="14" /></button>
          </li>
        </ul>
        <p class="muted note">文件归属你的数据范围；解析失败会按指数退避自动重试，超过最大次数后在任务列表中人工“重新执行”。</p>
      </div>
      <footer class="foot">
        <button type="button" class="btn" @click="emit('close')">取消</button>
        <button type="button" class="btn-p sm" :disabled="!files.length || busy" @click="submit">
          {{ busy ? '上传中…' : `开始上传${files.length ? `（${files.length}）` : ''}` }}
        </button>
      </footer>
    </section>
  </div>
</template>

<style scoped>
.mask { position: fixed; inset: 0; z-index: 40; background: rgba(35, 38, 63, 0.28); display: flex; align-items: center; justify-content: center; padding: 16px; }
.dialog { width: min(560px, 100%); border-radius: 20px; }
.body { padding: 18px 20px; display: flex; flex-direction: column; gap: 12px; }
.drop {
  display: flex; flex-direction: column; align-items: center; gap: 6px; padding: 28px 16px; border-radius: 16px; cursor: pointer; text-align: center;
  border: 2px dashed #d5d0f3; background: #faf9ff; color: var(--accent);
}
.drop.over { border-color: var(--accent); background: var(--accent-soft); }
.drop b { color: var(--text); font-size: 14px; }
.drop small { color: var(--muted); font-size: 12.5px; }
.files { list-style: none; margin: 0; padding: 0; display: flex; flex-direction: column; gap: 4px; max-height: 200px; overflow-y: auto; }
.files li { display: flex; align-items: center; gap: 10px; padding: 6px 8px; border-radius: 10px; background: #f8f7fd; font-size: 13px; }
.note { margin: 0; font-size: 12.5px; }
.foot { display: flex; justify-content: flex-end; gap: 10px; padding: 14px 20px; border-top: 1px solid var(--line); }
</style>
