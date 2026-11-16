<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { api, type BlockStatus, type BlockType, type ReviewBlock, type ReviewDoc } from '../api'
import AppIcon from '../components/AppIcon.vue'
import StageSteps from '../components/StageSteps.vue'
import { errorText, toast } from '../lib/toast'

const route = useRoute()
const router = useRouter()

const doc = ref<ReviewDoc | null>(null)
const blocks = ref<ReviewBlock[]>([])
const sel = ref('')
const editing = ref('')
const draft = ref('')
const tier = ref('standard')
const empty = ref(false)
const busy = ref(false)

const TABLE = ['线路', '天数', '成人价', '儿童价', '黄山 + 宏村', '4', '[成人价]', '[儿童价]', '杭州 + 乌镇', '3', '[成人价]', '[儿童价]']
const TYPE: Record<BlockType, string> = { title: '标题', text: '正文', table: '表格', image: '图片', footer: '页脚' }
const STATUS: { name: string; tint: string }[] = [
  { name: '未核对', tint: 'gray' }, { name: '已确认', tint: 'green' }, { name: '已修改', tint: 'purple' }, { name: '标记删除', tint: 'pink' }
]

const done = computed(() => blocks.value.filter((b) => b.status > 0).length)
const allDone = computed(() => done.value === blocks.value.length && blocks.value.length > 0)
const shown = (b: ReviewBlock) => b.edited ?? b.origin

function set(b: ReviewBlock, status: BlockStatus, text?: string): void {
  b.status = status
  if (text !== undefined) b.edited = text
  sel.value = b.id
}

function startEdit(b: ReviewBlock): void {
  if (b.fix) return set(b, 2, b.fix)
  editing.value = b.id
  draft.value = shown(b)
}

function saveEdit(b: ReviewBlock): void {
  set(b, draft.value.trim() === b.origin ? 1 : 2, draft.value.trim())
  editing.value = ''
}

async function load(): Promise<void> {
  let no = typeof route.query.no === 'string' ? route.query.no : ''
  try {
    if (!no) {
      const { rows } = await api.assets({ stage: 'review' })
      no = rows.find((a) => !['视频', '音频'].includes(a.modality))?.no || ''
      if (!no) {
        empty.value = true
        return
      }
    }
    doc.value = await api.reviewDoc(no)
    blocks.value = doc.value.blocks
    sel.value = blocks.value.find((b) => b.conf < 0.8)?.id || blocks.value[0]?.id || ''
  } catch (e) {
    toast(errorText(e))
    empty.value = true
  }
}

async function submit(action: 'draft' | 'submit' | 'reject'): Promise<void> {
  if (!doc.value) return
  busy.value = true
  try {
    await api.submitReview(doc.value.asset.no, blocks.value, action, tier.value)
    if (action === 'draft') return toast('草稿已保存')
    toast(action === 'submit' ? '核对已提交，进入分类' : `已驳回，使用 ${tier.value} 档位重新解析`)
    router.push('/content/tasks')
  } catch (e) {
    toast(errorText(e))
  } finally {
    busy.value = false
  }
}

onMounted(load)
</script>

<template>
  <div v-if="empty" class="card empty-card">
    <AppIcon name="check" :size="28" />
    <p>暂无待核对的文档。去 <RouterLink to="/content/tasks">任务列表</RouterLink> 看看其他任务。</p>
  </div>

  <div v-else-if="doc" class="stack">
    <nav class="crumb" aria-label="位置">
      <RouterLink to="/content/tasks">任务列表</RouterLink>
      <span>/</span>
      <span class="mono">{{ doc.asset.no }}</span>
      <b>{{ doc.asset.name }}</b>
    </nav>
    <StageSteps
      current="review"
      :notes="{ upload: doc.asset.uploadedAt, parse: `MinerU ${doc.tier} · ${doc.asset.attempts} 次成功` }"
      :skipped="['label']"
    />

    <div class="cols">
      <section class="card preview">
        <div class="row between head">
          <span class="muted">原件 · 第 {{ doc.page }} 页 / 共 {{ doc.pageCount }} 页</span>
          <span class="muted">点击右侧内容块可在原件中定位</span>
        </div>
        <div class="paper-bg">
          <div class="paper">
            <button
              v-for="(b, i) in blocks"
              :key="b.id"
              type="button"
              class="pblock"
              :class="[b.type, { sel: sel === b.id, low: b.conf < 0.8, del: b.status === 3 }]"
              :aria-label="`定位内容块 ${i + 1}`"
              @click="sel = b.id"
            >
              <span class="tag">{{ i + 1 }}</span>
              <span v-if="b.type === 'title'" class="ptitle">{{ shown(b) }}</span>
              <span v-else-if="b.type === 'table'" class="ptable">
                <span v-for="(c, ci) in TABLE" :key="ci">{{ c }}</span>
              </span>
              <span v-else-if="b.type === 'image'" class="pimg">[图片：宏村南湖]</span>
              <span v-else :class="b.type === 'footer' ? 'pfoot' : 'ptext'">{{ shown(b) }}</span>
            </button>
          </div>
        </div>
      </section>

      <section class="card blocks">
        <div class="card-head">
          <h2 class="sec">本页内容块</h2>
          <span class="muted small">已核对 {{ done }} / {{ blocks.length }} · 低置信度块已用橙色虚线框标出</span>
        </div>
        <ul>
          <li v-for="(b, i) in blocks" :key="b.id" :class="{ sel: sel === b.id }" @click="sel = b.id">
            <div class="row bhead">
              <span class="idx" :class="{ low: b.conf < 0.8, on: sel === b.id }">{{ i + 1 }}</span>
              <span class="pill gray">{{ TYPE[b.type] }}</span>
              <span class="conf" :class="{ bad: b.conf < 0.8 }">置信度 {{ b.conf.toFixed(2) }}</span>
              <span class="grow" />
              <span class="pill" :class="STATUS[b.status].tint">{{ STATUS[b.status].name }}</span>
            </div>
            <p v-if="b.edited !== undefined && b.edited !== b.origin" class="origin">原文：<s>{{ b.origin }}</s></p>
            <textarea v-if="editing === b.id" v-model="draft" class="fld" rows="3" aria-label="修改内容块文本" />
            <p v-else class="text" :class="{ del: b.status === 3 }">{{ shown(b) }}</p>
            <div class="row wrap acts">
              <template v-if="editing === b.id">
                <button type="button" class="btn" @click.stop="saveEdit(b)">保存修改</button>
                <button type="button" class="btn" @click.stop="editing = ''">取消</button>
              </template>
              <template v-else>
                <button type="button" class="btn" @click.stop="set(b, 1)"><AppIcon name="check" :size="14" />确认无误</button>
                <button v-if="b.type !== 'image' && b.type !== 'table'" type="button" class="btn" @click.stop="startEdit(b)">
                  <AppIcon name="edit" :size="14" />{{ b.fix ? '采纳修正：黔县 → 黟县' : '修改文本' }}
                </button>
                <button type="button" class="btn btn-danger" @click.stop="set(b, 3)"><AppIcon name="trash" :size="14" />标记删除</button>
              </template>
            </div>
          </li>
        </ul>
        <div class="foot">
          <label class="row">
            <span class="lbl">驳回并重新解析，档位</span>
            <select v-model="tier" class="fld">
              <option value="basic">basic</option>
              <option value="standard">standard</option>
              <option value="advanced">advanced</option>
            </select>
          </label>
          <button type="button" class="btn" :disabled="busy" @click="submit('reject')">驳回</button>
          <span class="grow" />
          <button type="button" class="btn" :disabled="busy" @click="submit('draft')">保存草稿</button>
          <button type="button" class="btn-p sm" :disabled="busy || !allDone" :title="allDone ? '' : '所有内容块核对后才能提交'" @click="submit('submit')">
            提交核对 <AppIcon name="arrowRight" :size="14" /> 分类
          </button>
        </div>
      </section>
    </div>
  </div>
</template>

<style scoped>
.between { justify-content: space-between; }
.crumb { display: flex; align-items: center; gap: 8px; font-size: 14px; flex-wrap: wrap; }
.crumb span { color: var(--faint); }
.crumb .mono { color: var(--text-2); }
.cols { display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1.05fr); gap: 16px; align-items: start; }
.preview { padding: 14px; }
.head { padding: 0 2px 10px; font-size: 13px; }
.paper-bg { background: #e8eaf0; border-radius: 12px; padding: 18px; }
.paper { background: #fff; margin: 0 auto; max-width: 440px; min-height: 520px; padding: 28px 30px; box-shadow: 0 2px 10px rgba(0, 0, 0, 0.08); display: flex; flex-direction: column; gap: 10px; }
.pblock { position: relative; display: block; width: 100%; text-align: left; border: 2px solid transparent; border-radius: 4px; background: none; padding: 6px 6px 6px 30px; font-size: 12.5px; color: #333; }
.pblock:hover { background: #f6f8fc; }
.pblock.low { border: 2px dashed #c2690e; }
.pblock.sel { border: 2px solid var(--accent); background: var(--accent-soft); }
.pblock.del { opacity: 0.35; text-decoration: line-through; }
.tag { position: absolute; left: 6px; top: 7px; width: 18px; height: 18px; border-radius: 5px; background: #eef1f5; color: #3a4456; font-size: 11px; display: flex; align-items: center; justify-content: center; }
.pblock.sel .tag { background: var(--accent); color: #fff; }
.ptitle { font-size: 17px; font-weight: 700; color: #1a1a2e; }
.ptext { line-height: 1.7; }
.ptable { display: grid; grid-template-columns: 1.4fr 0.7fr 1fr 1fr; border: 1px solid #c9ced8; font-size: 11.5px; }
.ptable span { padding: 4px 6px; border: 1px solid #e1e4ea; text-align: center; }
.ptable span:nth-child(-n + 4) { background: #f3f5f9; font-weight: 600; }
.pimg { display: flex; align-items: center; justify-content: center; height: 120px; background: #dfe5ee; color: #5b6576; border-radius: 2px; }
.pfoot { display: block; text-align: center; color: #888; font-size: 11px; }
.blocks ul { list-style: none; margin: 0; padding: 0; }
.blocks li { padding: 14px 20px; border-bottom: 1px solid #f0eff8; display: flex; flex-direction: column; gap: 8px; cursor: pointer; }
.blocks li.sel { background: #f7f6fe; box-shadow: inset 3px 0 0 var(--accent); }
.bhead { gap: 8px; }
.idx { width: 22px; height: 22px; border-radius: 6px; background: #eef1f5; color: #3a4456; font-size: 12px; display: inline-flex; align-items: center; justify-content: center; }
.idx.low { background: #fdf0e4; color: #8a3f06; }
.idx.on { background: var(--accent); color: #fff; }
.conf { font-size: 12.5px; color: var(--muted); }
.conf.bad { color: #9b1c12; font-weight: 500; }
.origin { margin: 0; font-size: 12.5px; color: var(--muted); }
.text { margin: 0; font-size: 14px; line-height: 1.7; }
.text.del { text-decoration: line-through; color: var(--faint); }
.acts .btn { height: 30px; font-size: 12.5px; }
.small { font-size: 12.5px; }
.foot { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; padding: 14px 20px; border-top: 1px solid var(--line); }
.empty-card { padding: 48px; text-align: center; color: var(--muted); display: flex; flex-direction: column; align-items: center; gap: 10px; }
@media (max-width: 1100px) { .cols { grid-template-columns: 1fr; } }
</style>
