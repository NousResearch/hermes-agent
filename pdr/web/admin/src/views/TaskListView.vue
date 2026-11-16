<script setup lang="ts">
import { computed, onMounted, ref, watch } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { api, type Asset, type Modality, type Stage, type StageCount } from '../api'
import AppIcon from '../components/AppIcon.vue'
import UploadDialog from '../components/UploadDialog.vue'
import { MODALITIES, MODALITY_TINT, nextAction, STAGES, stageName, STATUS } from '../lib/format'
import { errorText, toast } from '../lib/toast'
import { useAuth } from '../stores/auth'

const auth = useAuth()
const route = useRoute()
const router = useRouter()

const stage = ref<Stage | 'all'>('all')
const modality = ref<Modality | ''>('')
const keyword = ref(typeof route.query.keyword === 'string' ? route.query.keyword : '')
const rows = ref<Asset[]>([])
const counts = ref<StageCount[]>([])
const all = ref<Asset[]>([])
const uploading = ref(false)
const loading = ref(true)

const tabs = computed(() => [{ key: 'all' as const, name: '全部' }, ...STAGES].map((s) => ({
  ...s, qty: counts.value.find((c) => c.stage === s.key)?.qty ?? 0
})))

// 顶部概况：与原型一致的四类需要关注的数量
const overview = computed(() => {
  const a = all.value
  return [
    { name: '解析中 / 待重试', qty: a.filter((x) => x.stage === 'parse' && x.status !== 'failed').length, hint: 'MinerU basic 档位', tint: 'purple' },
    { name: '待人工核对', qty: a.filter((x) => x.stage === 'review').length, hint: `含 ${a.filter((x) => x.stage === 'review' && x.modality === '视频').length} 个视频`, tint: 'amber' },
    { name: '待标注 / 待分类', qty: a.filter((x) => x.stage === 'label' || x.stage === 'classify').length, hint: '模型预标注已就绪', tint: 'green' },
    { name: '失败待处理', qty: a.filter((x) => x.status === 'failed').length, hint: '已用尽自动重试次数', tint: 'pink' }
  ]
})
const failed = computed(() => all.value.filter((a) => a.status === 'failed'))

async function load(): Promise<void> {
  loading.value = true
  try {
    const [page, everything] = await Promise.all([
      api.assets({ stage: stage.value, modality: modality.value, keyword: keyword.value }),
      api.assets({ stage: 'all' })
    ])
    rows.value = page.rows
    counts.value = page.stages
    all.value = everything.rows
  } catch (e) {
    toast(errorText(e))
  } finally {
    loading.value = false
  }
}

async function act(a: Asset): Promise<void> {
  const next = nextAction(a)
  if (next.retry) return retry([a])
  if (next.path) router.push({ path: next.path, query: { no: a.no } })
  else toast(`${a.name}：${a.note}`)
}

async function retry(list: Asset[]): Promise<void> {
  try {
    for (const a of list) await api.retry(a.no)
    toast(`已重新执行 ${list.length} 个失败任务`)
    load()
  } catch (e) {
    toast(errorText(e))
  }
}

watch([stage, modality], load)
let timer: ReturnType<typeof setTimeout> | undefined
watch(keyword, () => {
  clearTimeout(timer)
  timer = setTimeout(load, 250)
})
onMounted(load)
</script>

<template>
  <div class="stack">
    <div class="row between">
      <h1 class="h">任务概况</h1>
      <span class="scope"><AppIcon name="pin" :size="15" />数据范围：{{ auth.user?.dataScopeText }}</span>
    </div>
    <div class="overview">
      <div v-for="o in overview" :key="o.name" class="ov" :class="o.tint">
        <p>{{ o.name }}</p>
        <b>{{ o.qty }}</b>
        <small>{{ o.hint }}</small>
      </div>
    </div>

    <div class="row between wrap">
      <h1 class="h">内容任务</h1>
      <div class="row">
        <button v-if="auth.can('content:retry')" type="button" class="btn" :disabled="!failed.length" @click="retry(failed)">
          <AppIcon name="refresh" :size="14" />重试失败任务{{ failed.length ? `（${failed.length}）` : '' }}
        </button>
        <button v-if="auth.can('content:upload')" type="button" class="btn-p sm" @click="uploading = true">
          <AppIcon name="upload" :size="15" />上传文件
        </button>
      </div>
    </div>

    <section class="card">
      <div class="tabs pad-x" role="tablist" aria-label="按阶段筛选">
        <button v-for="t in tabs" :key="t.key" type="button" role="tab" :aria-selected="stage === t.key" @click="stage = t.key">
          {{ t.name }}<span class="qty">{{ t.qty }}</span>
        </button>
      </div>
      <div class="filters">
        <label class="row">
          <span class="lbl">模态</span>
          <select v-model="modality" class="fld">
            <option value="">全部</option>
            <option v-for="m in MODALITIES" :key="m" :value="m">{{ m }}</option>
          </select>
        </label>
        <label class="row grow">
          <span class="lbl">搜索</span>
          <input v-model="keyword" class="fld grow" type="search" placeholder="资产编号 / 文件名" />
        </label>
      </div>
      <div class="tbl-wrap pad-x">
        <table class="tbl">
          <thead>
            <tr>
              <th>资产名称</th><th>模态</th><th>区域</th><th>阶段</th><th>状态</th><th>说明</th><th>处理人</th><th>更新时间</th><th>操作</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="a in rows" :key="a.no">
              <td>
                <span class="aname">{{ a.name }}</span>
                <span class="mono faint">{{ a.no }} · {{ a.sizeText }}</span>
              </td>
              <td><span class="pill" :class="MODALITY_TINT[a.modality]">{{ a.modality }}</span></td>
              <td class="nowrap">{{ a.region }}</td>
              <td class="nowrap">{{ stageName(a.stage) }}</td>
              <td><span class="pill" :class="STATUS[a.status][1]">{{ STATUS[a.status][0] }}</span></td>
              <td class="note">
                {{ a.note }}
                <span v-if="a.category" class="pill gray">{{ a.category }}</span>
              </td>
              <td class="nowrap"><span class="row owner"><span class="avatar">{{ a.owner.slice(0, 1) }}</span>{{ a.owner }}</span></td>
              <td class="nowrap muted">{{ a.updatedAt }}</td>
              <td class="nowrap">
                <button
                  type="button"
                  class="btn-link"
                  :disabled="nextAction(a).retry && !auth.can('content:retry')"
                  @click="act(a)"
                >{{ nextAction(a).label }}</button>
              </td>
            </tr>
          </tbody>
        </table>
        <p v-if="!loading && !rows.length" class="empty">该阶段暂无任务</p>
      </div>
      <div class="foot">
        <span class="muted">共 {{ rows.length }} 条（只显示数据范围内的资产）</span>
        <span class="muted">失败任务按指数退避自动重试，超过最大执行次数后需人工“重新执行”</span>
      </div>
    </section>

    <UploadDialog v-if="uploading" @close="uploading = false" @done="uploading = false; load()" />
  </div>
</template>

<style scoped>
.between { justify-content: space-between; }
.h { font-size: 17px; font-weight: 700; }
.scope { display: inline-flex; align-items: center; gap: 4px; font-size: 14px; color: var(--text-2); }
.overview { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 14px; }
.ov { border-radius: 18px; padding: 18px; border: 1px solid rgba(255, 255, 255, 0.9); box-shadow: 0 8px 20px rgba(108, 99, 190, 0.1); }
.ov p { margin: 0; font-size: 13px; color: var(--text-2); }
.ov b { display: block; font-size: 28px; line-height: 1.4; }
.ov small { color: var(--muted); font-size: 12.5px; }
.ov.purple { background: linear-gradient(135deg, #f1eeff, #e7e2fe); }
.ov.amber { background: linear-gradient(135deg, #fff8e6, #ffefc9); }
.ov.green { background: linear-gradient(135deg, #eaf8f0, #ddf3e6); }
.ov.pink { background: linear-gradient(135deg, #ffeff2, #fde1e7); }
.ov.pink b { color: var(--bad); }
.pad-x { padding: 0 20px; }
.filters { display: flex; gap: 16px; padding: 14px 20px; flex-wrap: wrap; }
.filters .fld { min-width: 120px; }
.aname { display: block; font-weight: 500; min-width: 200px; }
.note { color: var(--muted); font-size: 13px; min-width: 200px; }
.note .pill { margin-left: 4px; }
.nowrap { white-space: nowrap; }
.owner { gap: 6px; }
.foot { display: flex; justify-content: space-between; gap: 12px; flex-wrap: wrap; padding: 14px 20px; font-size: 12.5px; border-top: 1px solid var(--line); }
@media (max-width: 960px) { .overview { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
</style>
