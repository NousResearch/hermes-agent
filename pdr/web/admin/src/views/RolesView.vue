<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import { api, type DataScope, type Perm, type PermGroup, type Role } from '../api'
import AppIcon from '../components/AppIcon.vue'
import { errorText, toast } from '../lib/toast'

const roles = ref<Role[]>([])
const catalog = ref<PermGroup[]>([])
const cur = ref('')
const draft = ref<Role | null>(null)
const tab = ref<'func' | 'scope' | 'col'>('func')
const saving = ref(false)

const SCOPES: { v: DataScope; name: string }[] = [
  { v: '1', name: '1 · 本租户全部' }, { v: '2', name: '2 · 本区域及下级' }, { v: '3', name: '3 · 本区域' },
  { v: '4', name: '4 · 本人创建' }, { v: '5', name: '5 · 仅分配给本人' }
]
const COL_PERMS = [{ v: '0', name: '0 · 不可见' }, { v: '1', name: '1 · 脱敏' }, { v: '2', name: '2 · 只读' }, { v: '3', name: '3 · 可编辑' }]

const dirty = computed(() => {
  const saved = roles.value.find((r) => r.code === cur.value)
  return !!draft.value && JSON.stringify(saved) !== JSON.stringify(draft.value)
})
const assetScope = computed(() => draft.value?.dataScopes.find((s) => s.resource === 'content_asset')?.scope ?? '1')
// 以“归属上海市”的成员为例，展示所选数据范围展开后实际可见的区域
const preview = computed(() => {
  const s = assetScope.value
  return [
    { name: '中国', level: 0, on: s === '1' },
    { name: '上海市（归属区域）', level: 1, on: ['1', '2', '3'].includes(s) },
    { name: '黄浦区', level: 2, on: ['1', '2'].includes(s) },
    { name: '浦东新区', level: 2, on: ['1', '2'].includes(s) },
    { name: '安徽省', level: 1, on: s === '1' },
    { name: '黄山市 · 黄山景区 / 宏村', level: 2, on: s === '1' }
  ]
})
const scopeNote = computed(() => ({
  '1': '可见本租户全部区域的数据',
  '2': '“本区域及下级”展开后实际可见的区域如下',
  '3': '只看归属区域本身，不含下级区域',
  '4': '不按区域，只能看到本人创建的数据',
  '5': '不按区域，只能看到分配给本人的任务'
}[assetScope.value]))

function has(code: Perm): boolean {
  const p = draft.value?.perms
  return p === '*' || !!p?.includes(code)
}

function toggle(code: Perm): void {
  if (!draft.value || draft.value.perms === '*') return
  const set = new Set(draft.value.perms)
  if (set.has(code)) set.delete(code)
  else set.add(code)
  draft.value.perms = [...set]
}

function select(code: string): void {
  cur.value = code
  const r = roles.value.find((x) => x.code === code)
  draft.value = r ? JSON.parse(JSON.stringify(r)) : null
}

function create(): void {
  const base = roles.value[roles.value.length - 1]
  const code = `custom_${roles.value.length + 1}`
  const role: Role = {
    code, name: '新角色', builtin: false, userQty: 0, desc: '自定义角色', perms: ['content:dashboard'],
    dataScopes: base.dataScopes.map((s) => ({ ...s, scope: '3' })), columns: base.columns.map((c) => ({ ...c, perm: '0' }))
  }
  roles.value.push(role)
  select(code)
  tab.value = 'func'
}

async function save(): Promise<void> {
  if (!draft.value) return
  saving.value = true
  try {
    await api.saveRole(draft.value)
    const i = roles.value.findIndex((r) => r.code === draft.value!.code)
    roles.value[i] = JSON.parse(JSON.stringify(draft.value))
    toast(`角色「${draft.value.name}」已保存；成员下次请求时生效，数据范围同时写入数据库会话变量`)
  } catch (e) {
    toast(errorText(e))
  } finally {
    saving.value = false
  }
}

onMounted(async () => {
  try {
    ;[roles.value, catalog.value] = await Promise.all([api.roles(), api.permCatalog()])
    select(roles.value.find((r) => r.code === 'content_reviewer')?.code ?? roles.value[0]?.code ?? '')
  } catch (e) {
    toast(errorText(e))
  }
})
</script>

<template>
  <div class="stack">
    <div class="page-head">
      <div>
        <h1>角色与权限</h1>
        <p>功能权限决定能看到哪些菜单和按钮；数据范围决定能看到哪些区域的数据；列权限决定字段是否可见或脱敏。</p>
      </div>
      <button type="button" class="btn-p" @click="create"><AppIcon name="plus" :size="16" />新建角色</button>
    </div>

    <div class="cols">
      <aside class="card list">
        <button
          v-for="r in roles"
          :key="r.code"
          type="button"
          class="role"
          :aria-current="cur === r.code ? 'true' : undefined"
          @click="select(r.code)"
        >
          <b>{{ r.name }}</b>
          <small>{{ r.builtin ? '系统内置' : '自定义' }} · {{ r.userQty }} 人</small>
        </button>
      </aside>

      <section v-if="draft" class="card main">
        <div class="head">
          <div class="row wrap">
            <input v-if="!draft.builtin" v-model="draft.name" class="fld name" aria-label="角色名称" />
            <h2 v-else class="sec">{{ draft.name }}</h2>
            <span class="mono faint">{{ draft.code }}</span>
            <span class="pill" :class="draft.builtin ? 'purple' : 'green'">{{ draft.builtin ? '系统内置' : '自定义' }} · {{ draft.userQty }} 人</span>
          </div>
          <p class="muted">{{ draft.desc }}</p>
        </div>
        <div class="tabs pad-x" role="tablist">
          <button type="button" role="tab" :aria-selected="tab === 'func'" @click="tab = 'func'">功能权限</button>
          <button type="button" role="tab" :aria-selected="tab === 'scope'" @click="tab = 'scope'">数据范围</button>
          <button type="button" role="tab" :aria-selected="tab === 'col'" @click="tab = 'col'">列权限</button>
        </div>

        <div v-if="tab === 'func'" class="body perms">
          <fieldset v-for="g in catalog" :key="g.name" class="pgroup">
            <legend>{{ g.name }}</legend>
            <label v-for="p in g.perms" :key="p.code" class="perm">
              <input type="checkbox" :checked="has(p.code)" @change="toggle(p.code)" />
              <span>{{ p.name }}</span>
              <span class="mono faint">{{ p.code }}</span>
            </label>
          </fieldset>
        </div>

        <div v-else-if="tab === 'scope'" class="body scope">
          <table class="tbl">
            <thead><tr><th>资源类型</th><th>数据范围</th><th>适用操作</th></tr></thead>
            <tbody>
              <tr v-for="s in draft.dataScopes" :key="s.resource">
                <td>{{ s.resourceName }}<br /><span class="mono faint">{{ s.resource }}</span></td>
                <td>
                  <select v-model="s.scope" class="fld" :aria-label="`${s.resourceName} 的数据范围`">
                    <option v-for="o in SCOPES" :key="o.v" :value="o.v">{{ o.name }}</option>
                  </select>
                </td>
                <td>{{ s.action }}</td>
              </tr>
            </tbody>
          </table>
          <aside class="preview">
            <b>归属区域预览（以归属上海市的成员为例）</b>
            <p class="muted">内容资产：{{ scopeNote }}</p>
            <ul>
              <li v-for="n in preview" :key="n.name" :style="{ paddingLeft: n.level * 18 + 'px' }" :class="{ on: n.on }">
                <AppIcon :name="n.on ? 'eye' : 'lock'" :size="13" />{{ n.name }}<small v-if="!n.on">（不可见）</small>
              </li>
            </ul>
            <p class="faint small">同时写入数据库会话变量，PostgreSQL 行级安全会再校验一次。</p>
          </aside>
        </div>

        <div v-else class="body">
          <label class="row ctable">
            <span class="lbl">受控表</span>
            <select class="fld" aria-label="受控表">
              <option>ods.ods_fin_invoice_extract_di（发票抽取）</option>
            </select>
          </label>
          <table class="tbl">
            <thead><tr><th>字段</th><th>中文名</th><th>列权限</th><th>脱敏规则</th></tr></thead>
            <tbody>
              <tr v-for="c in draft.columns" :key="c.col">
                <td class="mono">{{ c.col }}</td>
                <td>{{ c.name }}</td>
                <td>
                  <select v-model="c.perm" class="fld" :aria-label="`${c.name} 的列权限`">
                    <option v-for="o in COL_PERMS" :key="o.v" :value="o.v">{{ o.name }}</option>
                  </select>
                </td>
                <td class="muted">{{ c.perm === '1' ? c.mask : '—' }}</td>
              </tr>
            </tbody>
          </table>
        </div>

        <div class="foot">
          <span v-if="dirty" class="pill blue">有未保存的修改</span>
          <span class="grow" />
          <button type="button" class="btn" :disabled="!dirty" @click="select(cur)">取消</button>
          <button type="button" class="btn-p sm" :disabled="!dirty || saving" @click="save">{{ saving ? '保存中…' : '保存' }}</button>
        </div>
      </section>
    </div>
  </div>
</template>

<style scoped>
.cols { display: grid; grid-template-columns: 240px minmax(0, 1fr); gap: 16px; align-items: start; }
.list { padding: 10px; display: flex; flex-direction: column; gap: 2px; }
.role { display: flex; flex-direction: column; gap: 2px; align-items: flex-start; padding: 10px 12px; border-radius: 10px; border: 0; background: none; text-align: left; }
.role:hover { background: #f6f5fd; }
.role[aria-current='true'] { background: var(--accent-soft); color: #3f2fc0; }
.role small { color: var(--muted); font-size: 12px; }
.head { padding: 16px 20px 8px; }
.head p { margin: 6px 0 0; font-size: 13px; }
.name { font-weight: 600; width: 200px; }
.pad-x { padding: 0 20px; }
.body { padding: 16px 20px; }
.perms { display: grid; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr)); gap: 14px; }
.pgroup { border: 1px solid var(--line); border-radius: 14px; padding: 10px 14px 12px; margin: 0; }
.pgroup legend { padding: 0 6px; font-weight: 600; font-size: 13.5px; }
.perm { display: flex; align-items: center; gap: 8px; padding: 5px 0; font-size: 13.5px; cursor: pointer; }
.perm input { accent-color: var(--accent); width: 16px; height: 16px; }
.perm .mono { margin-left: auto; font-size: 11.5px; }
.scope { display: grid; grid-template-columns: minmax(0, 1fr) 320px; gap: 18px; align-items: start; }
.preview { border: 1px solid var(--line); border-radius: 14px; padding: 14px 16px; font-size: 13px; }
.preview p { margin: 4px 0 8px; }
.preview ul { list-style: none; margin: 0; padding: 0; display: flex; flex-direction: column; gap: 6px; }
.preview li { display: flex; align-items: center; gap: 6px; color: var(--faint); }
.preview li.on { color: var(--accent); font-weight: 500; }
.small { font-size: 12px; margin-top: 10px !important; }
.ctable { margin-bottom: 12px; }
.foot { display: flex; align-items: center; gap: 10px; padding: 14px 20px; border-top: 1px solid var(--line); }
@media (max-width: 1100px) {
  .cols, .scope { grid-template-columns: 1fr; }
}
</style>
