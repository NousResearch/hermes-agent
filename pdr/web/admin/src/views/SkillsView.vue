<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import { api, type Skill, type Toolset } from '../api'
import AppIcon from '../components/AppIcon.vue'
import { errorText, toast } from '../lib/toast'

const toolsets = ref<Toolset[]>([])
const skills = ref<Skill[]>([])
const category = ref('全部')

const categories = computed(() => ['全部', ...new Set(skills.value.map((s) => s.category))])
const shownSkills = computed(() => skills.value.filter((s) => category.value === '全部' || s.category === category.value))
const enabledTools = computed(() => toolsets.value.filter((t) => t.enabled).reduce((n, t) => n + t.toolQty, 0))
const SOURCE: Record<Skill['source'], string> = { 内置: 'purple', 可选: 'blue', 租户自建: 'green' }

async function flipTool(t: Toolset): Promise<void> {
  try {
    await api.toggleToolset(t.name, !t.enabled)
    t.enabled = !t.enabled
    toast(`${t.label}已${t.enabled ? '启用' : '停用'}，下一会话生效`)
  } catch (e) {
    toast(errorText(e))
  }
}

async function flipSkill(s: Skill): Promise<void> {
  try {
    await api.toggleSkill(s.name, !s.enabled)
    s.enabled = !s.enabled
    toast(`技能 ${s.name} 已${s.enabled ? '启用' : '停用'}`)
  } catch (e) {
    toast(errorText(e))
  }
}

onMounted(async () => {
  try {
    const r = await api.toolsets()
    toolsets.value = r.toolsets
    skills.value = r.skills
  } catch (e) {
    toast(errorText(e))
  }
})
</script>

<template>
  <div class="stack">
    <div class="page-head">
      <div>
        <h1>技能 · 工具集</h1>
        <p>工具集决定智能体能调用哪些工具；技能是可复用的做事说明。修改只影响本租户，下一会话生效。</p>
      </div>
      <span class="pill purple">已启用 {{ toolsets.filter((t) => t.enabled).length }} 个工具集 · {{ enabledTools }} 个工具</span>
    </div>

    <section>
      <h2 class="sec h2">工具集</h2>
      <div class="tools">
        <div v-for="t in toolsets" :key="t.name" class="card tool" :class="{ off: !t.enabled }">
          <div class="row">
            <span class="tico"><AppIcon name="layers" :size="16" /></span>
            <span class="grow"><b>{{ t.label }}</b><span class="mono faint">{{ t.name }}</span></span>
            <span class="switch"><input type="checkbox" :checked="t.enabled" :aria-label="`启用 ${t.label}`" @change="flipTool(t)" /><span /></span>
          </div>
          <p>{{ t.desc }}</p>
          <small class="muted">{{ t.toolQty }} 个工具</small>
        </div>
      </div>
    </section>

    <section class="card">
      <div class="card-head">
        <h2 class="sec">技能</h2>
        <div class="tabs" role="tablist">
          <button v-for="c in categories" :key="c" type="button" role="tab" :aria-selected="category === c" @click="category = c">{{ c }}</button>
        </div>
      </div>
      <div class="tbl-wrap pad">
        <table class="tbl">
          <thead><tr><th>技能</th><th>分类</th><th>说明</th><th>来源</th><th>启用</th></tr></thead>
          <tbody>
            <tr v-for="s in shownSkills" :key="s.name">
              <td class="mono">{{ s.name }}</td>
              <td>{{ s.category }}</td>
              <td class="muted">{{ s.desc }}</td>
              <td><span class="pill" :class="SOURCE[s.source]">{{ s.source }}</span></td>
              <td><span class="switch"><input type="checkbox" :checked="s.enabled" :aria-label="`启用 ${s.name}`" @change="flipSkill(s)" /><span /></span></td>
            </tr>
          </tbody>
        </table>
      </div>
    </section>
  </div>
</template>

<style scoped>
.h2 { margin-bottom: 12px; }
.tools { display: grid; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr)); gap: 14px; }
.tool { padding: 16px; display: flex; flex-direction: column; gap: 8px; border-radius: 16px; }
.tool.off { background: #fbfbfe; }
.tool.off b, .tool.off p { color: var(--muted); }
.tico { width: 32px; height: 32px; border-radius: 10px; background: var(--accent-soft); color: var(--accent); display: flex; align-items: center; justify-content: center; flex: none; }
.tool.off .tico { background: #eef0f5; color: var(--faint); }
.tool b { display: block; }
.tool .mono { font-size: 12px; }
.tool p { margin: 0; font-size: 13px; line-height: 1.6; color: var(--text-2); }
.pad { padding: 4px 20px 12px; }
</style>
