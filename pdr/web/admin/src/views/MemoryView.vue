<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import { api, type MemoryEntry, type MemoryState } from '../api'
import AppIcon from '../components/AppIcon.vue'
import { errorText, toast } from '../lib/toast'

const state = ref<MemoryState | null>(null)
const persona = ref('')
const tab = ref<'pending' | 'memory' | 'user'>('pending')

const entries = computed(() => state.value?.entries ?? [])
const shown = computed(() => entries.value.filter((e) => (tab.value === 'pending' ? e.status === 'pending' : e.status === 'active' && e.kind === tab.value)))
const used = (kind: MemoryEntry['kind']) => entries.value.filter((e) => e.kind === kind && e.status === 'active').reduce((n, e) => n + e.content.length, 0)
const dirty = computed(() => state.value && persona.value !== state.value.persona)

async function savePersona(): Promise<void> {
  try {
    await api.savePersona(persona.value)
    if (state.value) state.value.persona = persona.value
    toast('人格设定已保存，下一会话生效')
  } catch (e) {
    toast(errorText(e))
  }
}

async function decide(e: MemoryEntry, approve: boolean): Promise<void> {
  try {
    await api.decideMemory(e.id, approve)
    if (approve) e.status = 'active'
    else if (state.value) state.value.entries = state.value.entries.filter((x) => x.id !== e.id)
    toast(approve ? '已批准写入记忆' : '已拒绝')
  } catch (err) {
    toast(errorText(err))
  }
}

async function remove(e: MemoryEntry): Promise<void> {
  try {
    await api.deleteMemory(e.id)
    if (state.value) state.value.entries = state.value.entries.filter((x) => x.id !== e.id)
    toast('已删除这条记忆')
  } catch (err) {
    toast(errorText(err))
  }
}

onMounted(async () => {
  try {
    state.value = await api.memory()
    persona.value = state.value.persona
  } catch (e) {
    toast(errorText(e))
  }
})
</script>

<template>
  <div v-if="state" class="stack">
    <div class="page-head">
      <div>
        <h1>记忆 · 人格</h1>
        <p>人格设定（SOUL.md）决定智能体的身份与回答原则；长期记忆与用户画像会注入系统提示词，后台写入需要审批。</p>
      </div>
    </div>

    <div class="cols">
      <section class="card">
        <div class="card-head">
          <h2 class="sec">人格设定</h2>
          <span class="mono faint">SOUL.md</span>
        </div>
        <div class="pad">
          <textarea v-model="persona" class="fld persona" rows="14" aria-label="人格设定" />
          <div class="row foot">
            <span class="muted small">{{ persona.length }} 字 · 下一会话生效</span>
            <span class="grow" />
            <button type="button" class="btn" :disabled="!dirty" @click="persona = state.persona">撤销</button>
            <button type="button" class="btn-p sm" :disabled="!dirty" @click="savePersona">保存</button>
          </div>
        </div>
      </section>

      <section class="card">
        <div class="card-head">
          <h2 class="sec">长期记忆</h2>
          <div class="quota">
            <span>记忆 {{ used('memory') }} / {{ state.memoryCharLimit }} 字</span>
            <span>用户画像 {{ used('user') }} / {{ state.userCharLimit }} 字</span>
          </div>
        </div>
        <div class="tabs pad-x" role="tablist">
          <button type="button" role="tab" :aria-selected="tab === 'pending'" @click="tab = 'pending'">
            待审批<span class="qty">{{ entries.filter((e) => e.status === 'pending').length }}</span>
          </button>
          <button type="button" role="tab" :aria-selected="tab === 'memory'" @click="tab = 'memory'">记忆</button>
          <button type="button" role="tab" :aria-selected="tab === 'user'" @click="tab = 'user'">用户画像</button>
        </div>
        <ul class="entries">
          <li v-for="e in shown" :key="e.id">
            <p>{{ e.content }}</p>
            <div class="row wrap">
              <span class="pill" :class="e.kind === 'memory' ? 'purple' : 'blue'">{{ e.kind === 'memory' ? '记忆' : '用户画像' }}</span>
              <span class="muted small">{{ e.source }} · {{ e.createdAt }}</span>
              <span class="grow" />
              <template v-if="e.status === 'pending'">
                <button type="button" class="btn" @click="decide(e, false)">拒绝</button>
                <button type="button" class="btn-p sm" @click="decide(e, true)"><AppIcon name="check" :size="14" />批准</button>
              </template>
              <button v-else type="button" class="btn btn-danger" @click="remove(e)"><AppIcon name="trash" :size="14" />删除</button>
            </div>
          </li>
          <li v-if="!shown.length" class="empty">{{ tab === 'pending' ? '没有待审批的记忆写入' : '暂无内容' }}</li>
        </ul>
      </section>
    </div>
  </div>
</template>

<style scoped>
.cols { display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); gap: 16px; align-items: start; }
.pad { padding: 16px 20px; }
.pad-x { padding: 0 20px; }
.persona { width: 100%; font-size: 13.5px; }
.foot { margin-top: 12px; }
.small { font-size: 12.5px; }
.quota { display: flex; gap: 12px; font-size: 12.5px; color: var(--muted); flex-wrap: wrap; }
.entries { list-style: none; margin: 0; padding: 0; }
.entries li { padding: 14px 20px; border-bottom: 1px solid #f0eff8; display: flex; flex-direction: column; gap: 8px; }
.entries p { margin: 0; font-size: 14px; line-height: 1.7; }
@media (max-width: 1100px) { .cols { grid-template-columns: 1fr; } }
</style>
