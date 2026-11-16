<script setup lang="ts">
import { computed, onMounted, ref, watch } from 'vue'
import { api, type KnowledgeBase, type KnowledgeDoc } from '../api'
import AppIcon from '../components/AppIcon.vue'
import { MODALITY_TINT } from '../lib/format'
import { errorText, toast } from '../lib/toast'
import { useAuth } from '../stores/auth'

const auth = useAuth()
const bases = ref<KnowledgeBase[]>([])
const cur = ref('')
const docs = ref<KnowledgeDoc[]>([])
const kw = ref('')

const manage = computed(() => auth.can('sys:role'))
const current = computed(() => bases.value.find((b) => b.code === cur.value))
const filtered = computed(() => docs.value.filter((d) => !kw.value.trim() || d.name.includes(kw.value.trim())))
const totals = computed(() => ({
  docs: bases.value.reduce((n, b) => n + b.docQty, 0),
  chunks: bases.value.reduce((n, b) => n + b.chunkQty, 0),
  open: bases.value.filter((b) => b.enabled).length
}))

async function toggle(b: KnowledgeBase): Promise<void> {
  try {
    await api.toggleKnowledgeBase(b.code, !b.enabled)
    b.enabled = !b.enabled
    toast(b.enabled ? `「${b.name}」已对问答开放` : `「${b.name}」已停用，问答不再检索`)
  } catch (e) {
    toast(errorText(e))
  }
}

watch(cur, async (code) => {
  if (!code) return
  try {
    docs.value = await api.knowledgeDocs(code)
  } catch (e) {
    toast(errorText(e))
  }
})

onMounted(async () => {
  try {
    bases.value = await api.knowledgeBases()
    cur.value = bases.value[0]?.code ?? ''
  } catch (e) {
    toast(errorText(e))
  }
})
</script>

<template>
  <div class="stack">
    <div class="page-head">
      <div>
        <h1>知识库</h1>
        <p>入库并完成分类的资产按分类进入知识库，用户端问答只检索已启用的知识库，并按提问人的区域权限过滤。</p>
      </div>
      <div class="row stats">
        <span><b>{{ totals.docs }}</b>文档</span>
        <span><b>{{ totals.chunks.toLocaleString() }}</b>检索片段</span>
        <span><b>{{ totals.open }}</b>个对问答开放</span>
      </div>
    </div>

    <div class="cols">
      <div class="bases">
        <button
          v-for="b in bases"
          :key="b.code"
          type="button"
          class="kb card"
          :class="{ on: cur === b.code }"
          :aria-pressed="cur === b.code"
          @click="cur = b.code"
        >
          <span class="row top">
            <span class="kico"><AppIcon name="book" :size="18" /></span>
            <b class="grow">{{ b.name }}</b>
            <span class="pill" :class="b.enabled ? 'green' : 'gray'">{{ b.enabled ? '问答可用' : '已停用' }}</span>
          </span>
          <span class="desc">{{ b.desc }}</span>
          <span class="row nums">
            <span>{{ b.docQty }} 个文档</span><span>{{ b.chunkQty.toLocaleString() }} 个片段</span><span>{{ b.scopeText }}</span>
          </span>
        </button>
      </div>

      <section v-if="current" class="card">
        <div class="card-head">
          <div>
            <h2 class="sec">{{ current.name }}</h2>
            <p class="muted small">最近更新 {{ current.updatedAt }} · 可见范围：{{ current.scopeText }}</p>
          </div>
          <div class="row">
            <input v-model="kw" class="fld" type="search" placeholder="搜索文档" aria-label="搜索文档" />
            <label v-if="manage" class="row toggle">
              <span class="lbl">对问答开放</span>
              <span class="switch"><input type="checkbox" :checked="current.enabled" @change="toggle(current)" /><span /></span>
            </label>
          </div>
        </div>
        <div class="tbl-wrap pad">
          <table class="tbl">
            <thead><tr><th>文档</th><th>模态</th><th>片段数</th><th>区域</th><th>入库时间</th><th>被引用次数</th></tr></thead>
            <tbody>
              <tr v-for="d in filtered" :key="d.assetNo">
                <td><b class="dname">{{ d.name }}</b><span class="mono faint">{{ d.assetNo }}</span></td>
                <td><span class="pill" :class="MODALITY_TINT[d.modality]">{{ d.modality }}</span></td>
                <td>{{ d.chunkQty }}</td>
                <td>{{ d.region }}</td>
                <td class="muted">{{ d.storedAt }}</td>
                <td><span class="hits"><i :style="{ width: Math.min(100, d.hitQty / 3.2) + '%' }" /></span>{{ d.hitQty }}</td>
              </tr>
            </tbody>
          </table>
          <p v-if="!filtered.length" class="empty">没有符合条件的文档（只显示你的数据范围内的文档）</p>
        </div>
      </section>
    </div>
  </div>
</template>

<style scoped>
.stats { gap: 18px; color: var(--muted); font-size: 13px; }
.stats b { font-size: 20px; color: var(--text); margin-right: 4px; }
.cols { display: grid; grid-template-columns: 340px minmax(0, 1fr); gap: 16px; align-items: start; }
.bases { display: flex; flex-direction: column; gap: 12px; }
.kb { display: flex; flex-direction: column; gap: 8px; padding: 16px; text-align: left; border-radius: 16px; }
.kb:hover { border-color: #cfcbe6; }
.kb.on { border-color: var(--accent); box-shadow: 0 0 0 3px rgba(84, 67, 214, 0.12), var(--shadow); }
.top { gap: 10px; }
.kico { width: 34px; height: 34px; border-radius: 10px; background: var(--warn-soft); color: var(--warn); display: flex; align-items: center; justify-content: center; }
.desc { font-size: 13px; color: var(--muted); line-height: 1.6; }
.nums { gap: 12px; font-size: 12px; color: var(--faint); flex-wrap: wrap; }
.small { font-size: 12.5px; margin: 2px 0 0; }
.toggle { gap: 8px; }
.pad { padding: 4px 20px 12px; }
.dname { display: block; font-weight: 500; }
.hits { display: inline-block; width: 60px; height: 6px; border-radius: 999px; background: #eeedf7; margin-right: 8px; vertical-align: middle; overflow: hidden; }
.hits i { display: block; height: 100%; background: linear-gradient(90deg, #ffd66b, #f2b33d); }
@media (max-width: 1100px) { .cols { grid-template-columns: 1fr; } }
</style>
