<script setup lang="ts">
import { computed, onMounted, ref, watch } from 'vue'
import { api, type ConfigChange, type ConfigChangeLog, type ConfigItem, type ConfigSchema, type ConfigScope, type ConfigValue } from '../api'
import { RELOAD_TINT, showValue } from '../lib/format'
import { errorText, toast } from '../lib/toast'
import AppIcon from './AppIcon.vue'

const props = defineProps<{
  /** agent：config.yaml 的 hermes 段（智能体）；pdr：PDR 自身各段 */
  part: 'agent' | 'pdr'
  scopes: ConfigScope[]
}>()

const scope = ref<ConfigScope>(props.scopes[0])
const schema = ref<ConfigSchema | null>(null)
const groupKey = ref('')
const keyword = ref('')
const edits = ref<Record<string, string | boolean>>({})
const errors = ref<Record<string, string>>({})
const logs = ref<ConfigChangeLog[] | null>(null)
const saving = ref(false)

const groups = computed(() => (schema.value?.groups ?? []).filter((g) => g.scope === props.part))
const group = computed(() => groups.value.find((g) => g.key === groupKey.value) ?? groups.value[0])
const items = computed<ConfigItem[]>(() => {
  const kw = keyword.value.trim().toLowerCase()
  if (!kw) return group.value?.items ?? []
  return groups.value.flatMap((g) => g.items)
    .filter((i) => i.label.toLowerCase().includes(kw) || i.path.toLowerCase().includes(kw) || i.desc.toLowerCase().includes(kw))
    .slice(0, 80)
})
const itemQty = computed(() => groups.value.reduce((n, g) => n + g.items.length, 0))
const byPath = computed(() => new Map(groups.value.flatMap((g) => g.items).map((i) => [i.path, i])))
const changedPaths = computed(() => Object.keys(edits.value).filter((p) => {
  const it = byPath.value.get(p)
  return it && toText(it) !== String(edits.value[p])
}))

function toText(it: ConfigItem): string | boolean {
  if (it.kind === 'bool') return it.value === true
  if (it.kind === 'json') return JSON.stringify(it.value ?? null, null, 0)
  return it.value === null || it.value === undefined ? '' : String(it.value)
}

function current(it: ConfigItem): string | boolean {
  return it.path in edits.value ? edits.value[it.path] : toText(it)
}

function update(it: ConfigItem, v: string | boolean): void {
  edits.value[it.path] = v
  delete errors.value[it.path]
}

function reset(it: ConfigItem): void {
  delete edits.value[it.path]
  delete errors.value[it.path]
}

function parse(it: ConfigItem, raw: string | boolean): ConfigValue {
  if (it.kind === 'bool') return raw === true
  const text = String(raw).trim()
  if (it.kind === 'json') return JSON.parse(text || 'null')
  if (it.kind === 'number') {
    if (!text) {
      if (it.nullable || it.value === null) return null
      throw new Error('不能为空')
    }
    const n = Number(text)
    if (Number.isNaN(n)) throw new Error('请输入数字')
    if (it.min !== undefined && n < it.min) throw new Error(`不能小于 ${it.min}`)
    if (it.max !== undefined && n > it.max) throw new Error(`不能大于 ${it.max}`)
    return n
  }
  if (!text && (it.nullable || it.value === null)) return null
  return text
}

async function load(): Promise<void> {
  try {
    schema.value = await api.configSchema(scope.value)
    if (!groups.value.some((g) => g.key === groupKey.value)) groupKey.value = groups.value[0]?.key ?? ''
  } catch (e) {
    toast(errorText(e))
  }
}

async function save(): Promise<void> {
  const changes: ConfigChange[] = []
  errors.value = {}
  for (const p of changedPaths.value) {
    const it = byPath.value.get(p)!
    try {
      changes.push({ path: p, label: it.label, before: it.value, after: parse(it, edits.value[p]) })
    } catch (e) {
      errors.value[p] = e instanceof SyntaxError ? 'JSON 格式不正确' : errorText(e)
    }
  }
  if (Object.keys(errors.value).length) return toast('有配置项填写不正确，请检查标红的项')
  saving.value = true
  try {
    const entry = await api.saveConfig(scope.value, changes)
    edits.value = {}
    await load()
    toast(scope.value === 'platform'
      ? `已写入 config.yaml（${changes.length} 项），SHA-256 ${entry.sha256?.slice(0, 12)}…`
      : `已保存 ${changes.length} 项租户覆盖，不改 config.yaml`)
  } catch (e) {
    toast(errorText(e))
  } finally {
    saving.value = false
  }
}

async function openLog(): Promise<void> {
  try {
    logs.value = await api.configLog()
  } catch (e) {
    toast(errorText(e))
  }
}

watch(scope, () => {
  edits.value = {}
  load()
})
onMounted(load)
</script>

<template>
  <div class="editor">
    <div class="page-head">
      <div>
        <h1>{{ part === 'agent' ? '智能体配置' : '系统配置' }}</h1>
        <p>
          {{ part === 'agent' ? '智能体引擎' : 'PDR 系统' }}全部 {{ itemQty }} 项配置，{{ groups.length }} 个分组；中文名、说明、取值范围与生效方式来自 config.yaml 的元数据行。
        </p>
      </div>
      <div v-if="scopes.length > 1" class="seg" role="radiogroup" aria-label="配置层级">
        <button type="button" role="radio" :aria-checked="scope === 'platform'" @click="scope = 'platform'">平台级（写入 config.yaml）</button>
        <button type="button" role="radio" :aria-checked="scope === 'tenant'" @click="scope = 'tenant'">租户覆盖（示例旅行社）</button>
      </div>
    </div>

    <div class="cols">
      <aside class="card side">
        <input v-model="keyword" class="fld search" type="search" placeholder="搜索中文名、配置键或说明" aria-label="搜索配置项" />
        <p class="gtitle">{{ part === 'agent' ? '智能体' : 'PDR 系统' }}（{{ groups.length }} 组）</p>
        <div class="glist">
          <button
            v-for="g in groups"
            :key="g.key"
            type="button"
            class="gbtn"
            :aria-current="!keyword && g.key === group?.key ? 'true' : undefined"
            @click="groupKey = g.key; keyword = ''"
          >
            <span class="ellipsis">{{ g.name }}</span>
            <small>{{ g.items.length }}</small>
          </button>
        </div>
      </aside>

      <section class="card main">
        <div class="card-head">
          <div class="row">
            <h2 class="sec">{{ keyword ? `搜索“${keyword}”` : group?.name }}</h2>
            <span v-if="!keyword && group" class="mono faint">{{ group.items[0]?.path.split('.').slice(0, -1).join('.') }}</span>
          </div>
          <span class="muted small">{{ keyword ? `找到 ${items.length} 项` : `本组共 ${items.length} 项` }}</span>
        </div>
        <p class="notice" :class="scope === 'platform' ? 'warn' : 'info'">
          <template v-if="scope === 'platform'">平台级：保存后写回 config.yaml（保留注释），记录变更并附写入后文件的 SHA-256；“重启”生效的项需要重启服务。</template>
          <template v-else>租户覆盖视图：未覆盖的项显示平台值；保存后只写入数据库覆盖表，并重新生成该租户的智能体配置，不改 config.yaml。</template>
        </p>

        <ul class="items">
          <li v-for="it in items" :key="it.path" :class="{ changed: changedPaths.includes(it.path), err: errors[it.path] }">
            <div class="info">
              <div class="row wrap">
                <b>{{ it.label }}</b>
                <span class="pill" :class="RELOAD_TINT[it.reload] || 'gray'">{{ it.reload }}</span>
                <span v-if="changedPaths.includes(it.path)" class="pill blue">已修改</span>
              </div>
              <span class="mono faint path">{{ it.path }}</span>
              <p class="desc">{{ it.desc }}</p>
              <p v-if="changedPaths.includes(it.path)" class="was muted">原值：{{ showValue(it.value) }}
                <button type="button" class="btn-link" @click="reset(it)">撤销</button></p>
            </div>
            <div class="ctl">
              <label v-if="it.kind === 'bool'" class="row">
                <span class="switch"><input type="checkbox" :checked="current(it) === true" :aria-label="it.label" @change="update(it, ($event.target as HTMLInputElement).checked)" /><span /></span>
                <span>{{ current(it) === true ? '开启' : '关闭' }}</span>
              </label>
              <select v-else-if="it.kind === 'enum'" class="fld" :value="current(it)" :aria-label="it.label" @change="update(it, ($event.target as HTMLSelectElement).value)">
                <option v-if="it.value === null || it.value === ''" value="">（未设置）</option>
                <option v-for="o in it.options" :key="o.value" :value="o.value">{{ o.label }}</option>
              </select>
              <textarea
                v-else-if="it.kind === 'json'"
                class="fld mono"
                rows="2"
                :value="String(current(it))"
                :aria-label="it.label"
                @input="update(it, ($event.target as HTMLTextAreaElement).value)"
              />
              <input
                v-else
                class="fld"
                :inputmode="it.kind === 'number' ? 'decimal' : undefined"
                :value="String(current(it))"
                :placeholder="it.nullable || it.value === null ? '留空表示不设置' : ''"
                :aria-label="it.label"
                @input="update(it, ($event.target as HTMLInputElement).value)"
              />
              <small class="muted">范围：{{ it.range || '—' }}</small>
              <small v-if="errors[it.path]" class="bad">{{ errors[it.path] }}</small>
            </div>
          </li>
          <li v-if="!items.length" class="empty">没有匹配的配置项</li>
        </ul>

        <div class="foot">
          <span class="pill green"><AppIcon name="check" :size="12" />config.yaml 的 SHA-256 与最近一次变更记录一致</span>
          <span class="grow" />
          <button type="button" class="btn" @click="openLog"><AppIcon name="history" :size="14" />查看变更记录</button>
          <button type="button" class="btn-p sm" :disabled="!changedPaths.length || saving" @click="save">
            {{ saving ? '保存中…' : `保存${changedPaths.length ? `（${changedPaths.length} 项）` : ''}` }}
          </button>
        </div>
      </section>
    </div>

    <div v-if="logs" class="mask" @click.self="logs = null">
      <section class="card dialog" role="dialog" aria-modal="true" aria-labelledby="log-title">
        <header class="card-head">
          <h2 id="log-title" class="sec">配置变更记录</h2>
          <button type="button" class="ib" aria-label="关闭" @click="logs = null"><AppIcon name="x" /></button>
        </header>
        <ul class="logs">
          <li v-for="(l, i) in logs" :key="i">
            <div class="row">
              <b>{{ l.user }}</b>
              <span class="pill" :class="l.scope === 'platform' ? 'amber' : 'purple'">{{ l.scope === 'platform' ? '平台级' : '租户覆盖' }}</span>
              <span class="grow" />
              <span class="muted small">{{ l.at }}</span>
            </div>
            <p v-for="c in l.changes" :key="c.path" class="chg">
              {{ c.label }} <span class="mono faint">{{ c.path }}</span>：{{ showValue(c.before) }} → <b>{{ showValue(c.after) }}</b>
            </p>
            <p v-if="l.sha256" class="mono faint small">SHA-256 {{ l.sha256 }}</p>
          </li>
          <li v-if="!logs.length" class="empty">还没有变更</li>
        </ul>
      </section>
    </div>
  </div>
</template>

<style scoped>
.seg { display: inline-flex; border: 1px solid var(--field); border-radius: 10px; overflow: hidden; background: #fff; }
.seg button { border: 0; background: #fff; padding: 0 14px; height: 34px; font-size: 13px; }
.seg button[aria-checked='true'] { background: var(--accent); color: #fff; }
.cols { display: grid; grid-template-columns: 260px minmax(0, 1fr); gap: 16px; align-items: start; }
.side { padding: 14px; position: sticky; top: 16px; max-height: calc(100vh - 32px); display: flex; flex-direction: column; }
.search { width: 100%; margin-bottom: 10px; }
.gtitle { margin: 4px 6px 6px; font-size: 12.5px; color: var(--muted); }
.glist { overflow-y: auto; display: flex; flex-direction: column; gap: 2px; }
.gbtn { display: flex; justify-content: space-between; align-items: center; gap: 8px; padding: 7px 10px; border-radius: 9px; border: 0; background: none; text-align: left; font-size: 13.5px; color: var(--text); }
.gbtn small { color: var(--faint); font-size: 12px; }
.gbtn:hover { background: #f6f5fd; }
.gbtn[aria-current='true'] { background: var(--accent-soft); color: #3f2fc0; font-weight: 500; }
.small { font-size: 12.5px; }
.main > .notice { margin: 14px 20px 0; }
.items { list-style: none; margin: 0; padding: 6px 0; }
.items li { display: grid; grid-template-columns: minmax(0, 1fr) 300px; gap: 20px; padding: 14px 20px; border-bottom: 1px solid #f0eff8; }
.items li.changed { background: #f8f7ff; box-shadow: inset 3px 0 0 var(--accent); }
.items li.err { box-shadow: inset 3px 0 0 var(--bad); }
.items li.empty { display: block; }
.info { display: flex; flex-direction: column; gap: 4px; min-width: 0; }
.path { font-size: 12px; word-break: break-all; }
.desc { margin: 0; font-size: 13px; color: var(--text-2); line-height: 1.65; }
.was { margin: 0; font-size: 12.5px; }
.ctl { display: flex; flex-direction: column; gap: 6px; }
.ctl .fld { width: 100%; }
.ctl small { font-size: 12px; }
.bad { color: var(--bad); }
.foot { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; padding: 14px 20px; border-top: 1px solid var(--line); }
.mask { position: fixed; inset: 0; z-index: 40; background: rgba(35, 38, 63, 0.28); display: flex; align-items: center; justify-content: center; padding: 16px; }
.dialog { width: min(640px, 100%); max-height: 80vh; display: flex; flex-direction: column; }
.logs { list-style: none; margin: 0; padding: 6px 20px 16px; overflow-y: auto; }
.logs li { padding: 12px 0; border-bottom: 1px solid #f0eff8; display: flex; flex-direction: column; gap: 4px; }
.chg { margin: 0; font-size: 13px; }
@media (max-width: 1100px) {
  .cols { grid-template-columns: 1fr; }
  .side { position: static; max-height: 280px; }
  .items li { grid-template-columns: 1fr; }
}
</style>
