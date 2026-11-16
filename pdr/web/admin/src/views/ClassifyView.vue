<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import { api, type CategoryNode, type ClassifyItem } from '../api'
import AppIcon from '../components/AppIcon.vue'
import { MODALITY_TINT } from '../lib/format'
import { errorText, toast } from '../lib/toast'

const items = ref<ClassifyItem[]>([])
const tree = ref<CategoryNode[]>([])
const choice = ref<Record<string, string>>({})
const loaded = ref(false)

const leaves = computed(() => tree.value.flatMap((c) => (c.children ?? []).map((s) => ({ code: s.code, name: `${c.name} / ${s.name}`, group: c.name }))))

async function load(): Promise<void> {
  try {
    const q = await api.reviewQueue()
    items.value = q.classify.filter((c) => !c.chosen)
    tree.value = q.categories
    for (const it of items.value) choice.value[it.asset.no] = it.suggestions[0]?.code ?? ''
  } catch (e) {
    toast(errorText(e))
  } finally {
    loaded.value = true
  }
}

async function confirm(it: ClassifyItem): Promise<void> {
  const code = choice.value[it.asset.no]
  try {
    await api.confirmCategory(it.asset.no, code)
    toast(`已归类「${leaves.value.find((l) => l.code === code)?.name}」，开始入库`)
    items.value = items.value.filter((x) => x !== it)
  } catch (e) {
    toast(errorText(e))
  }
}

onMounted(load)
</script>

<template>
  <div class="stack">
    <div class="page-head">
      <div>
        <h1>分类确认</h1>
        <p>模型根据内容给出分类建议；确认后资产按分类入库，并进入对应知识库。置信度达到自动采纳阈值的资产不会出现在这里。</p>
      </div>
      <span class="pill amber">待确认 {{ items.length }}</span>
    </div>

    <div class="grid">
      <section v-for="it in items" :key="it.asset.no" class="card card-pad item">
        <div class="row">
          <span class="pill" :class="MODALITY_TINT[it.asset.modality]">{{ it.asset.modality }}</span>
          <b class="ellipsis">{{ it.asset.name }}</b>
        </div>
        <p class="meta mono faint">{{ it.asset.no }} · {{ it.asset.region }} · {{ it.asset.uploadedAt }}</p>
        <p class="summary"><span class="lbl">内容摘要</span>{{ it.summary }}</p>
        <div class="sugs" role="radiogroup" :aria-label="`${it.asset.name} 的分类`">
          <label v-for="s in it.suggestions" :key="s.code" class="sug" :class="{ on: choice[it.asset.no] === s.code }">
            <input v-model="choice[it.asset.no]" type="radio" :value="s.code" :name="it.asset.no" />
            <span class="grow">{{ s.name }}</span>
            <span class="bar"><i :style="{ width: s.conf * 100 + '%' }" /></span>
            <b>{{ s.conf.toFixed(2) }}</b>
          </label>
        </div>
        <label class="row other">
          <span class="lbl">或选择其他分类</span>
          <select v-model="choice[it.asset.no]" class="fld grow">
            <optgroup v-for="c in tree" :key="c.code" :label="c.name">
              <option v-for="s in c.children" :key="s.code" :value="s.code">{{ c.name }} / {{ s.name }}</option>
            </optgroup>
          </select>
        </label>
        <div class="row end">
          <button type="button" class="btn-p sm" @click="confirm(it)"><AppIcon name="check" :size="14" />确认分类并入库</button>
        </div>
      </section>
    </div>

    <section v-if="loaded && !items.length" class="card empty">没有待确认分类的资产</section>
  </div>
</template>

<style scoped>
.grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(420px, 1fr)); gap: 16px; }
.item { display: flex; flex-direction: column; gap: 10px; }
.meta { margin: -4px 0 0; }
.summary { margin: 0; font-size: 13.5px; line-height: 1.7; background: var(--soft); border-radius: 12px; padding: 10px 12px; }
.summary .lbl { display: block; font-size: 12px; margin-bottom: 2px; }
.sugs { display: flex; flex-direction: column; gap: 6px; }
.sug { display: flex; align-items: center; gap: 10px; padding: 9px 12px; border-radius: 12px; border: 1px solid var(--line); font-size: 13.5px; cursor: pointer; }
.sug.on { border-color: #b9b0fb; background: #faf9ff; }
.sug input { accent-color: var(--accent); }
.bar { width: 90px; height: 6px; border-radius: 999px; background: #eeedf7; overflow: hidden; }
.bar i { display: block; height: 100%; background: linear-gradient(90deg, #9e92fb, #5443d6); }
.other { gap: 10px; }
.end { justify-content: flex-end; }
@media (max-width: 560px) { .grid { grid-template-columns: 1fr; } }
</style>
