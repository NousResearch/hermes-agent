<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import { api, type LabelItem } from '../api'
import AppIcon from '../components/AppIcon.vue'
import { MODALITY_TINT } from '../lib/format'
import { errorText, toast } from '../lib/toast'

const items = ref<LabelItem[]>([])
const show = ref<'pending' | 'done'>('pending')
const picked = ref<Record<string, string>>({})
const loaded = ref(false)

const groups = computed(() => {
  const map = new Map<string, { no: string; name: string; modality: LabelItem['modality']; items: LabelItem[]; total: number; left: number }>()
  for (const it of items.value) {
    const g = map.get(it.assetNo) ?? { no: it.assetNo, name: it.assetName, modality: it.modality, items: [], total: 0, left: 0 }
    g.total += 1
    if (it.status === 'pending') g.left += 1
    if (it.status === show.value) g.items.push(it)
    map.set(it.assetNo, g)
  }
  return [...map.values()]
})
const pendingQty = computed(() => items.value.filter((i) => i.status === 'pending').length)

async function load(): Promise<void> {
  try {
    items.value = (await api.reviewQueue()).label
    for (const it of items.value) picked.value[it.id] = it.label ?? it.modelLabel
  } catch (e) {
    toast(errorText(e))
  } finally {
    loaded.value = true
  }
}

async function save(it: LabelItem): Promise<void> {
  try {
    await api.saveLabel(it.id, picked.value[it.id])
    it.label = picked.value[it.id]
    it.status = 'done'
    toast(`已标注为「${it.label}」`)
  } catch (e) {
    toast(errorText(e))
  }
}

async function acceptAll(list: LabelItem[]): Promise<void> {
  for (const it of list.filter((x) => x.conf >= 0.9 && x.status === 'pending')) {
    picked.value[it.id] = it.modelLabel
    await save(it)
  }
}

onMounted(load)
</script>

<template>
  <div class="stack">
    <div class="page-head">
      <div>
        <h1>标注工作台</h1>
        <p>模型已给出预标注，确认或改选后保存；一个资产全部确认后自动进入分类。只显示分配给你或在你数据范围内的任务。</p>
      </div>
      <div class="tabs" role="tablist">
        <button type="button" role="tab" :aria-selected="show === 'pending'" @click="show = 'pending'">待确认<span class="qty">{{ pendingQty }}</span></button>
        <button type="button" role="tab" :aria-selected="show === 'done'" @click="show = 'done'">已完成<span class="qty">{{ items.length - pendingQty }}</span></button>
      </div>
    </div>

    <section v-for="g in groups" :key="g.no" class="card">
      <div class="card-head">
        <div class="row">
          <span class="pill" :class="MODALITY_TINT[g.modality]">{{ g.modality }}</span>
          <b>{{ g.name }}</b>
          <span class="mono faint">{{ g.no }}</span>
        </div>
        <div class="row">
          <span class="muted small">进度 {{ g.total - g.left }} / {{ g.total }}</span>
          <button v-if="show === 'pending' && g.items.length" type="button" class="btn" @click="acceptAll(g.items)">
            <AppIcon name="sparkle" :size="14" />采纳置信度 ≥ 0.90 的建议
          </button>
        </div>
      </div>
      <ul class="items">
        <li v-for="it in g.items" :key="it.id">
          <div class="target">
            <AppIcon :name="it.modality === '音频' ? 'audio' : 'image'" :size="15" />
            <span class="mono">{{ it.target }}</span>
          </div>
          <p class="content">“{{ it.content }}”</p>
          <div class="row wrap">
            <span class="muted small">模型建议：<b>{{ it.modelLabel }}</b>
              <span :class="it.conf < 0.8 ? 'bad' : ''">（{{ it.conf.toFixed(2) }}）</span></span>
            <span class="grow" />
            <div class="opts" role="radiogroup" :aria-label="`${it.target} 的标签`">
              <button
                v-for="o in it.options"
                :key="o"
                type="button"
                role="radio"
                class="opt"
                :aria-checked="picked[it.id] === o"
                :disabled="it.status === 'done'"
                @click="picked[it.id] = o"
              >{{ o }}</button>
            </div>
            <button v-if="it.status === 'pending'" type="button" class="btn-p sm" @click="save(it)">保存</button>
            <span v-else class="pill green"><AppIcon name="check" :size="12" />已标注</span>
          </div>
        </li>
        <li v-if="!g.items.length" class="empty">{{ show === 'pending' ? '这个资产已全部标注完成' : '还没有已完成的标注' }}</li>
      </ul>
    </section>

    <section v-if="loaded && !groups.length" class="card empty">没有分配给你的标注任务</section>
  </div>
</template>

<style scoped>
.small { font-size: 12.5px; }
.items { list-style: none; margin: 0; padding: 0; }
.items li { padding: 14px 20px; border-bottom: 1px solid #f0eff8; display: flex; flex-direction: column; gap: 8px; }
.items li:last-child { border-bottom: 0; }
.target { display: inline-flex; align-items: center; gap: 6px; color: var(--muted); font-size: 12.5px; }
.content { margin: 0; font-size: 14.5px; line-height: 1.7; }
.bad { color: #9b1c12; }
.opts { display: flex; gap: 6px; flex-wrap: wrap; }
.opt { height: 30px; padding: 0 12px; border-radius: 999px; border: 1px solid var(--field); background: #fff; font-size: 13px; color: var(--text-2); }
.opt[aria-checked='true'] { background: var(--accent-soft); border-color: #b9b0fb; color: var(--accent); font-weight: 500; }
.opt:disabled[aria-checked='true'] { opacity: 1; }
</style>
