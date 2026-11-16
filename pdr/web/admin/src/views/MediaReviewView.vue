<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { api, type MediaReview, type MediaSegment } from '../api'
import AppIcon from '../components/AppIcon.vue'
import StageSteps from '../components/StageSteps.vue'
import { errorText, toast } from '../lib/toast'

const route = useRoute()
const router = useRouter()

const media = ref<MediaReview | null>(null)
const segs = ref<MediaSegment[]>([])
const f = ref(0)
const editing = ref(-1)
const draft = ref('')
const empty = ref(false)
const busy = ref(false)

const fmt = (s: number) => `${String(Math.floor(s / 60)).padStart(2, '0')}:${String(s % 60).padStart(2, '0')}`
const frame = computed(() => media.value?.frames[f.value])
const progress = computed(() => (media.value && frame.value ? (frame.value.t / media.value.duration) * 100 : 0))
const done = computed(() => segs.value.filter((s) => s.status > 0).length)

function linked(i: number): string {
  const frames = media.value?.frames.filter((x) => x.seg === i) ?? []
  if (!frames.length) return ''
  const inside = frames.every((x) => x.t >= segs.value[i].a && x.t < segs.value[i].b)
  return `帧 ${frames.map((x) => fmt(x.t)).join('、')} · ${inside ? '时间重叠' : '最近邻'}`
}

function pickSeg(i: number): void {
  const idx = media.value?.frames.findIndex((x) => x.seg === i) ?? -1
  if (idx >= 0) f.value = idx
}

function edit(i: number): void {
  editing.value = i
  draft.value = segs.value[i].text
}

function saveEdit(i: number): void {
  const s = segs.value[i]
  s.status = draft.value.trim() === s.text ? 1 : 2
  s.text = draft.value.trim()
  editing.value = -1
}

async function load(): Promise<void> {
  let no = typeof route.query.no === 'string' ? route.query.no : ''
  try {
    if (!no) {
      const { rows } = await api.assets({ stage: 'review' })
      no = rows.find((a) => ['视频', '音频'].includes(a.modality))?.no || ''
      if (!no) {
        empty.value = true
        return
      }
    }
    media.value = await api.mediaReview(no)
    segs.value = media.value.segments
    f.value = 2
  } catch (e) {
    toast(errorText(e))
    empty.value = true
  }
}

async function submit(): Promise<void> {
  if (!media.value) return
  busy.value = true
  try {
    await api.submitMedia(media.value.asset.no, segs.value)
    toast('核对已提交，进入分类')
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
    <p>暂无待核对的音视频。去 <RouterLink to="/content/tasks">任务列表</RouterLink> 看看其他任务。</p>
  </div>

  <div v-else-if="media" class="stack">
    <nav class="crumb" aria-label="位置">
      <RouterLink to="/content/tasks">任务列表</RouterLink>
      <span>/</span>
      <span class="mono">{{ media.asset.no }}</span>
      <b>{{ media.asset.name }}</b>
    </nav>
    <StageSteps current="review" :notes="{ upload: media.asset.uploadedAt, parse: '转写 + 关键帧' }" :skipped="['label']" />

    <div class="cols">
      <section class="card card-pad">
        <div class="player">
          <div class="screen">
            <span class="ftag">{{ frame?.kind }} · {{ frame ? fmt(frame.t) : '' }}</span>
            <b>{{ frame?.label }}</b>
            <small>画面文字（OCR）：{{ frame?.ocr }}</small>
          </div>
          <div class="controls">
            <AppIcon name="play" :size="16" />
            <span class="track"><i :style="{ width: progress + '%' }" /></span>
            <span class="mono">{{ frame ? fmt(frame.t) : '00:00' }} / {{ fmt(media.duration) }}</span>
          </div>
        </div>
        <h2 class="sec sub">关键帧（{{ media.frames.length }}）</h2>
        <div class="frames">
          <button
            v-for="(x, i) in media.frames"
            :key="x.t"
            type="button"
            class="frame"
            :class="{ on: i === f }"
            @click="f = i"
          >
            <span class="thumb">{{ x.label }}</span>
            <span class="mono">{{ fmt(x.t) }}</span>
            <small>{{ x.kind }}</small>
          </button>
        </div>
      </section>

      <section class="card segs">
        <div class="card-head">
          <h2 class="sec">语音转写</h2>
          <span class="muted small">已核对 {{ done }} / {{ segs.length }} · 与关键帧按时间对齐</span>
        </div>
        <ul>
          <li v-for="(s, i) in segs" :key="s.a" :class="{ on: frame?.seg === i }" @click="pickSeg(i)">
            <span class="time mono">{{ fmt(s.a) }}–{{ fmt(s.b) }}</span>
            <div class="grow body">
              <div class="row meta">
                <span class="pill blue">{{ s.speaker }}</span>
                <span :class="s.conf < 0.8 ? 'bad' : 'muted'">置信 {{ s.conf.toFixed(2) }}</span>
                <span v-if="linked(i)" class="faint">{{ linked(i) }}</span>
                <span class="grow" />
                <span class="pill" :class="['gray', 'green', 'purple', 'pink'][s.status]">{{ ['未核对', '已确认', '已修改', '标记删除'][s.status] }}</span>
              </div>
              <textarea v-if="editing === i" v-model="draft" class="fld" rows="2" aria-label="修改转写文本" @click.stop />
              <p v-else class="text">{{ s.text }}</p>
              <div class="row acts">
                <template v-if="editing === i">
                  <button type="button" class="btn" @click.stop="saveEdit(i)">保存修改</button>
                  <button type="button" class="btn" @click.stop="editing = -1">取消</button>
                </template>
                <template v-else>
                  <button type="button" class="btn" @click.stop="s.status = 1"><AppIcon name="check" :size="14" />确认</button>
                  <button type="button" class="btn" @click.stop="edit(i)"><AppIcon name="edit" :size="14" />修改</button>
                </template>
              </div>
            </div>
          </li>
        </ul>
        <div class="foot">
          <span class="muted small">提示：方括号内为模型无法确定的名称，请对照画面补全。</span>
          <span class="grow" />
          <button type="button" class="btn-p sm" :disabled="busy || done < segs.length" @click="submit">
            提交核对 <AppIcon name="arrowRight" :size="14" /> 分类
          </button>
        </div>
      </section>
    </div>
  </div>
</template>

<style scoped>
.crumb { display: flex; align-items: center; gap: 8px; font-size: 14px; flex-wrap: wrap; }
.crumb span { color: var(--faint); }
.cols { display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); gap: 16px; align-items: start; }
.player { border-radius: 14px; overflow: hidden; background: #1d2033; }
.screen {
  height: 280px; display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 8px; color: #fff; position: relative;
  background: radial-gradient(400px 200px at 50% 40%, #3b4170, #1d2033);
}
.screen b { font-size: 22px; }
.screen small { color: #c4c7e0; }
.ftag { position: absolute; left: 14px; top: 12px; font-size: 12px; background: rgba(255, 255, 255, 0.12); padding: 2px 8px; border-radius: 6px; }
.controls { display: flex; align-items: center; gap: 12px; padding: 10px 14px; color: #fff; font-size: 12px; }
.track { flex: 1; height: 4px; border-radius: 999px; background: rgba(255, 255, 255, 0.2); }
.track i { display: block; height: 100%; border-radius: inherit; background: #9e92fb; }
.sub { margin: 16px 0 10px; font-size: 14px; }
.frames { display: grid; grid-template-columns: repeat(auto-fill, minmax(96px, 1fr)); gap: 10px; }
.frame { display: flex; flex-direction: column; gap: 4px; padding: 6px; border-radius: 10px; text-align: left; background: #fff; border: 2px solid #e6e4f2; font-size: 12px; }
.frame.on { border-color: var(--accent); }
.thumb { height: 52px; border-radius: 6px; background: #dce3ec; display: flex; align-items: center; justify-content: center; text-align: center; font-size: 11px; color: #4b5568; padding: 2px; }
.frame.on .thumb { background: #c9d8f5; }
.frame small { color: var(--muted); }
.segs ul { list-style: none; margin: 0; padding: 0; }
.segs li { display: flex; gap: 12px; padding: 12px 20px; border-bottom: 1px solid #f0eff8; cursor: pointer; }
.segs li.on { background: #f0f5fe; box-shadow: inset 3px 0 0 var(--accent); }
.time { flex: none; color: var(--muted); padding-top: 2px; }
.body { display: flex; flex-direction: column; gap: 6px; }
.meta { gap: 8px; font-size: 12.5px; flex-wrap: wrap; }
.bad { color: #9b1c12; font-weight: 500; }
.text { margin: 0; font-size: 14px; line-height: 1.7; }
.acts .btn { height: 30px; font-size: 12.5px; }
.small { font-size: 12.5px; }
.foot { display: flex; align-items: center; gap: 10px; padding: 14px 20px; border-top: 1px solid var(--line); flex-wrap: wrap; }
.empty-card { padding: 48px; text-align: center; color: var(--muted); display: flex; flex-direction: column; align-items: center; gap: 10px; }
@media (max-width: 1100px) { .cols { grid-template-columns: 1fr; } }
</style>
