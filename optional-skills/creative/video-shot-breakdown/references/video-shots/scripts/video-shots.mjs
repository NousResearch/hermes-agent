#!/usr/bin/env node
// video-shots — deterministic helpers for the video-shots skill (拉片 / 逐镜拆解).
// Zero dependencies on purpose: the skill must work in any directory without an
// npm install. Node 18+ (stdlib only) + ffmpeg/ffprobe on PATH.

import { execFileSync } from 'node:child_process';
import { existsSync, mkdirSync, readFileSync, readdirSync, realpathSync, rmSync, writeFileSync } from 'node:fs';
import { basename, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

/* ------------------------------------------------------------------ */
/* 常量                                                                */
/* ------------------------------------------------------------------ */
/*
 * 拉片的前提刻在骨子里：**镜头边界是量出来的，不是看出来的。**
 *
 *   切点   ← ffmpeg 场景检测（scene score），确定性，模型不参与
 *   时长   ← 切点相减，两位小数，模型不许估
 *   运动量 ← 逐帧差分的中位数，一条时间轴曲线（track.json）
 *   景别 / 类别 / 运镜 / 画面 ← 只有这四件事是模型的活
 *
 * 这条分界线是本 skill 的全部价值：能算的都算掉，模型只判断它真正
 * 该判断的东西，然后每一条判断都被代码当场对账。
 */

export const DEFAULT_PARAMS = {
  sceneThreshold: 0.3,     // ffmpeg 场景检测阈值，越低切得越碎
  minShotSeconds: 0.3,     // 短于它的片段并进上一镜（闪帧、字幕跳变）
  boundaryTolerance: 0.05, // 相邻镜头首尾相接的容差（秒）
  endTolerance: 0.25,      // 末镜收尾对总时长的容差（秒）
  cutTolerance: 0.1,       // 镜头边界对齐 seedCuts / manualCuts 的容差（秒）
  staticMaxMotion: 1.5,    // 实测运动中位数低于它 = 画面几乎没动
  busyMinMotion: 12,       // 高于它 = 画面动得厉害（只提示不拦，见 motion 门）
  motionGateMinSeconds: 1, // 短于它的镜头采样点太少，motion 门不查（值照给）
  minFrameChars: 12,       // 中文画面描述的最低字数
  minFrameWords: 8,        // 英文画面描述的最低词数（12 个字符只有两个单词，等于没设门）
  minRhythmChars: 8,       // 中文节奏理由的最低字数
  minRhythmWords: 5,       // 英文节奏理由的最低词数
  hookWindowSeconds: 5,    // 开篇多少秒内该出现钩子（只提示不拦）
  flatRun: 6,              // 连续多少镜同一个节奏角色算「节奏平」（只提示不拦）
  trackHz: 5,              // 运动曲线采样率（每秒几个点）
  frameDir: 'frames',      // 关键帧目录
};

export function paramsOf(doc) {
  return { ...DEFAULT_PARAMS, ...(doc?.params ?? {}) };
}

/** 景别枚举：depth 决定节奏带的颜色深浅（越近越深）。 */
export const SHOT_SIZES = {
  none: { zh: '无景别', en: 'n/a', depth: 0.08, color: '#e4e8d9' }, // 黑场、纯字卡、纯图形——画面里没有被取景的空间
  'extreme-wide': { zh: '大远景', en: 'extreme wide', depth: 0.22, color: '#dae0c8' },
  wide: { zh: '全景', en: 'wide', depth: 0.34, color: '#c4cfaa' },
  'medium-wide': { zh: '中远景', en: 'medium wide', depth: 0.46, color: '#b0c091' },
  medium: { zh: '中景', en: 'medium', depth: 0.58, color: '#94aa74' },
  'medium-close': { zh: '中近景', en: 'medium close', depth: 0.7, color: '#788f58' },
  close: { zh: '特写', en: 'close-up', depth: 0.85, color: '#526f45' },
  'extreme-close': { zh: '大特写', en: 'extreme close-up', depth: 1, color: '#345136' },
};

/** 镜头类别：这一镜在片子里干什么活。evidence 是该类别必须拿出的证据字段。 */
export const SHOT_CATEGORIES = {
  establishing: { zh: '定场', en: 'establishing' },
  subject: { zh: '主体', en: 'subject' },
  dialogue: { zh: '对话', en: 'dialogue', evidence: 'audio' },
  reaction: { zh: '反应', en: 'reaction', evidence: 'subjects' },
  insert: { zh: '插入特写', en: 'insert' },
  pov: { zh: '主观', en: 'POV' },
  empty: { zh: '空镜', en: 'empty', evidence: 'no-subjects' },
  product: { zh: '产品展示', en: 'product' },
  'text-card': { zh: '字卡', en: 'text card', evidence: 'onscreenText' },
  transition: { zh: '转场镜头', en: 'transition' },
  archive: { zh: '引用素材', en: 'archive' },
};

/**
 * 运镜枚举。motion 是这个运镜在像素上**必然**留下的痕迹：
 *   still   固定机位——只要没有大主体运动，帧间差就该接近 0
 *   subtle  轻微/局部变化——不设门，实测值给人看
 *   strong  整幅画面必然移动——实测接近 0 就一定是判错了（motion 门只拦这一向）
 */
export const CAMERA_MOVES = {
  static: { zh: '固定', en: 'static', motion: 'still' },
  'push-in': { zh: '推', en: 'push in', motion: 'strong' },
  'pull-out': { zh: '拉', en: 'pull out', motion: 'strong' },
  'zoom-in': { zh: '变焦推', en: 'zoom in', motion: 'strong' },
  'zoom-out': { zh: '变焦拉', en: 'zoom out', motion: 'strong' },
  'pan-left': { zh: '左摇', en: 'pan left', motion: 'strong' },
  'pan-right': { zh: '右摇', en: 'pan right', motion: 'strong' },
  'tilt-up': { zh: '上摇', en: 'tilt up', motion: 'strong' },
  'tilt-down': { zh: '下摇', en: 'tilt down', motion: 'strong' },
  'truck-left': { zh: '左移', en: 'truck left', motion: 'strong' },
  'truck-right': { zh: '右移', en: 'truck right', motion: 'strong' },
  'pedestal-up': { zh: '升', en: 'pedestal up', motion: 'strong' },
  'pedestal-down': { zh: '降', en: 'pedestal down', motion: 'strong' },
  tracking: { zh: '跟拍', en: 'tracking', motion: 'strong' },
  arc: { zh: '环绕', en: 'arc', motion: 'strong' },
  'whip-pan': { zh: '甩镜', en: 'whip pan', motion: 'strong' },
  handheld: { zh: '手持微晃', en: 'handheld', motion: 'subtle' },
  shake: { zh: '剧烈晃动', en: 'shake', motion: 'strong' },
  'rack-focus': { zh: '变焦点', en: 'rack focus', motion: 'subtle' },
  'micro-push': { zh: '微推', en: 'micro push', motion: 'subtle' },
  roll: { zh: '旋转', en: 'roll', motion: 'strong' },
  drone: { zh: '航拍移动', en: 'drone', motion: 'strong' },
};

/**
 * 节奏角色（`rhythm`）：这一镜在**观众的注意力曲线**上干什么活。
 *
 * 景别/类别/运镜回答「怎么拍的」，节奏回答「为什么观众还没划走」。
 * 词表按短视频真正留人的那几件事拆——每个值都对应一种可观察的观众反应，
 * 不是情绪形容词。判不准就留空（整片留空是允许的，半张表不行，门查）。
 */
export const RHYTHM_ROLES = {
  hook: { zh: '钩子', en: 'hook', color: '#d8e07a' },
  setup: { zh: '铺垫', en: 'setup', color: '#9fb488' },
  build: { zh: '递进', en: 'build', color: '#8fb0a0' },
  beat: { zh: '重音', en: 'beat', color: '#c9a15e' },
  turn: { zh: '转折', en: 'turn', color: '#d98060' },
  payoff: { zh: '兑现', en: 'payoff', color: '#c56a4e' },
  breath: { zh: '换气', en: 'breath', color: '#7f8f9c' },
  close: { zh: '收口', en: 'close', color: '#8a7fa0' },
};

/** 入点转场方式。省略 = cut（硬切）。 */
export const TRANSITIONS = {
  cut: { zh: '硬切', en: 'cut' },
  dissolve: { zh: '叠化', en: 'dissolve' },
  'fade-in': { zh: '淡入', en: 'fade in' },
  'fade-out': { zh: '淡出', en: 'fade out' },
  whip: { zh: '甩切', en: 'whip' },
  'match-cut': { zh: '匹配剪辑', en: 'match cut' },
  wipe: { zh: '划像', en: 'wipe' },
  morph: { zh: '特效转场', en: 'morph' },
};

/**
 * 画面描述的空话词表。拉片的画面栏要能拿去核对——一句「氛围感很强」
 * 既不能验证也不能复现，等于没写。门查到就拦，改成看得见的东西。
 */
export const VAGUE_WORDS = [
  '氛围感', '高级感', '视觉冲击', '令人', '唯美', '美不胜收', '大气磅礴',
  '震撼人心', '画面感十足', '很美', '非常美', '精美绝伦', '赏心悦目', '引人入胜',
];

/** 英文空话表。判据跟着描述本身的语言走，不跟着界面语言走。 */
export const VAGUE_WORDS_EN = [
  'atmospheric', 'atmosphere is', 'cinematic vibe', 'visually stunning', 'visually striking',
  'breathtaking', 'gorgeous', 'mesmerizing', 'captivating', 'evocative', 'aesthetically',
  'beautifully shot', 'stunning', 'epic feel', 'moody vibe',
];

/** 画面描述的废话开头：镜头表里每行都在写镜头，不用再声明一遍。 */
export const FILLER_OPENERS = [/^这一?个?镜头/, /^本镜头?/, /^该镜头/, /^此镜头/];
export const FILLER_OPENERS_EN = [
  /^this shot\b/i, /^the shot\b/i, /^in this shot\b/i, /^this scene\b/i, /^in this scene\b/i,
  /^we see\b/i, /^the camera shows\b/i, /^the (image|frame) shows\b/i, /^here we\b/i,
];

/** 有没有中日韩文字——用来判断该按「数字数」还是「数词数」查画面描述。 */
const CJK = /[㐀-鿿぀-ヿ가-힯]/;

const r2 = (n) => Math.round(n * 100) / 100;
const r1 = (n) => Math.round(n * 10) / 10;

export function fmtTime(sec) {
  const s = Math.max(0, Number(sec) || 0);
  const m = Math.floor(s / 60);
  const rest = s - m * 60;
  return `${String(m).padStart(2, '0')}:${rest.toFixed(2).padStart(5, '0')}`;
}

export const shotNo = (i) => `S${String(i + 1).padStart(2, '0')}`;

/* ------------------------------------------------------------------ */
/* ffmpeg 层：能量出来的都在这里量                                      */
/* ------------------------------------------------------------------ */

function run(bin, args) {
  return execFileSync(bin, args, { encoding: 'utf8', maxBuffer: 1 << 28, stdio: ['ignore', 'pipe', 'pipe'] });
}

/** ffprobe：时长、帧率、分辨率、有没有声音。 */
export function probe(video) {
  const out = run('ffprobe', [
    '-v', 'error', '-print_format', 'json',
    '-show_entries', 'format=duration',
    '-show_entries', 'stream=codec_type,codec_name,width,height,r_frame_rate',
    video,
  ]);
  const j = JSON.parse(out);
  const v = (j.streams ?? []).find((s) => s.codec_type === 'video');
  if (!v) throw new Error(`${video} 里没有视频流`);
  const [num, den] = String(v.r_frame_rate ?? '0/1').split('/').map(Number);
  const width = v.width ?? 0;
  const height = v.height ?? 0;
  return {
    durationSeconds: r2(Number(j.format?.duration ?? 0)),
    fps: den ? r2(num / den) : 0,
    width,
    height,
    aspect: aspectOf(width, height),
    codec: v.codec_name ?? '',
    hasAudio: (j.streams ?? []).some((s) => s.codec_type === 'audio'),
  };
}

function aspectOf(w, h) {
  if (!w || !h) return '';
  const g = (a, b) => (b ? g(b, a % b) : a);
  const d = g(w, h);
  return `${w / d}:${h / d}`;
}

/** ffmpeg 场景检测：返回切点时刻（秒）。这就是镜头边界的唯一来源。 */
export function detectCuts(video, threshold) {
  const out = run('ffmpeg', [
    '-v', 'error', '-i', video, '-an',
    '-vf', `scale=320:-2,select='gt(scene,${threshold})',metadata=print:file=-`,
    '-f', 'null', '-',
  ]);
  const cuts = [];
  for (const line of out.split('\n')) {
    const m = /pts_time:([0-9.]+)/.exec(line);
    if (m) cuts.push(r2(Number(m[1])));
  }
  return cuts;
}

/** 逐帧差分曲线：每秒 hz 个采样点，值越大画面变化越剧烈。 */
export function motionTrack(video, hz) {
  const out = run('ffmpeg', [
    '-v', 'error', '-i', video, '-an',
    '-vf', `fps=${hz},scale=64:36,tblend=all_mode=difference,signalstats,metadata=print:file=-:key=lavfi.signalstats.YAVG`,
    '-f', 'null', '-',
  ]);
  const values = [];
  for (const line of out.split('\n')) {
    const m = /signalstats\.YAVG=([0-9.]+)/.exec(line);
    if (m) values.push(r1(Number(m[1])));
  }
  return { hz, values };
}

/**
 * 一段区间的运动中位数。**两端各切掉一小段**——切点那一帧的差分必然爆表，
 * 不剔掉的话每个镜头看起来都在动。剔除区间按镜长自适应，短镜也能拿到值
 * （值照样给人看，但 motion 门不查短镜，见 motionGateMinSeconds）。
 */
export function medianMotion(track, start, end) {
  if (!track || !Array.isArray(track.values) || !track.hz) return null;
  const span = end - start;
  if (!(span > 0)) return null;
  const edge = Math.min(0.4, Math.max(0.1, span * 0.15));
  const from = Math.ceil((start + edge) * track.hz);
  const to = Math.floor((end - edge) * track.hz);
  const slice = track.values.slice(Math.max(0, from), Math.max(0, to) + 1).filter((v) => Number.isFinite(v));
  if (!slice.length) return null;
  const sorted = [...slice].sort((a, b) => a - b);
  const mid = sorted.length >> 1;
  return r2(sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2);
}

/* ------------------------------------------------------------------ */
/* seed：把视频拆成工作底稿                                            */
/* ------------------------------------------------------------------ */

export function buildSeed(meta, cuts, track, opts = {}) {
  const params = { ...DEFAULT_PARAMS, ...(opts.params ?? {}) };
  const total = meta.durationSeconds;
  const rawCuts = [...new Set(cuts.filter((t) => t > 0 && t < total))].sort((a, b) => a - b);

  // 短于 minShotSeconds 的碎片并进上一镜：闪帧、字幕跳变、转场中间帧，
  // 它们是检测的噪声不是剪辑意图。并掉的切点仍留在 seedCuts 里备查。
  const bounds = [0];
  for (const t of rawCuts) {
    if (t - bounds[bounds.length - 1] >= params.minShotSeconds) bounds.push(t);
  }
  if (total - bounds[bounds.length - 1] < params.minShotSeconds && bounds.length > 1) bounds.pop();
  bounds.push(total);

  const shots = [];
  for (let i = 0; i < bounds.length - 1; i += 1) {
    const start = r2(bounds[i]);
    const end = r2(bounds[i + 1]);
    shots.push({
      id: shotNo(i),
      start,
      end,
      seconds: r2(end - start),
      motion: medianMotion(track, start, end),
      size: '',
      category: '',
      camera: '',
      transitionIn: 'cut',
      subjects: [],
      frame: '',
      onscreenText: '',
      audio: '',
    });
  }

  return {
    source: opts.source ?? '',
    title: opts.title ?? '',
    lang: opts.lang === 'en' ? 'en' : 'zh',
    meta,
    params: opts.params ?? {},
    seedCuts: rawCuts,
    manualCuts: [],
    cast: [],
    shots,
  };
}

/* ------------------------------------------------------------------ */
/* recut：补刀与并刀                                                    */
/* ------------------------------------------------------------------ */
/*
 * 场景检测不是神。暗场对暗场、叠化、同机位换人，它都可能漏；手持晃动、
 * 闪光、字幕跳变，它又可能多切。所以补刀并刀是常规操作，但**绝不能用手改**
 * ——一次 split 要动 id、start、end、seconds、motion 和后面所有镜头的编号，
 * 手改必漏一处。这个命令把它变成确定性操作：
 *
 *   --split <秒>  在这个时刻加一刀（同时记进 manualCuts，boundary 门认它）
 *   --merge <秒>  把这个时刻的那一刀去掉，前后两镜并成一镜
 *
 * 边界没动过的镜头，标注**原样保留**；被拆被并的镜头，标注清空并在 note 里
 * 写明出身——这两半是不是一回事得重新看画面，不许把旧描述顺下去。
 */
export function recut(doc, { splits = [], merges = [], track = null } = {}) {
  const p = paramsOf(doc);
  const total = Number(doc?.meta?.durationSeconds) || 0;
  const old = doc.shots ?? [];
  const tol = p.cutTolerance;

  const bounds = new Set([0, total]);
  for (const s of old) { bounds.add(r2(Number(s.start))); bounds.add(r2(Number(s.end))); }
  for (const t of merges) {
    const hit = [...bounds].find((b) => b !== 0 && b !== total && Math.abs(b - t) <= tol);
    if (hit == null) throw new Error(`--merge ${t}：这个时刻上没有镜头边界（容差 ${tol} 秒）`);
    bounds.delete(hit);
  }
  for (const t of splits) {
    const at = r2(t);
    if (!(at > 0 && at < total)) throw new Error(`--split ${t}：超出片长 0–${total}`);
    if ([...bounds].some((b) => Math.abs(b - at) <= tol)) throw new Error(`--split ${t}：这里已经有一刀了`);
    bounds.add(at);
  }

  const sorted = [...bounds].sort((a, b) => a - b);
  const shots = [];
  for (let i = 0; i < sorted.length - 1; i += 1) {
    const start = sorted[i];
    const end = sorted[i + 1];
    const id = shotNo(i);
    const same = old.find((s) => Math.abs(Number(s.start) - start) <= 0.011 && Math.abs(Number(s.end) - end) <= 0.011);
    if (same) {
      shots.push({ ...same, id, motion: track ? medianMotion(track, start, end) : same.motion });
      continue;
    }
    const from = old.filter((s) => Number(s.start) < end - 0.011 && Number(s.end) > start + 0.011).map((s) => s.id);
    shots.push({
      id,
      start,
      end,
      seconds: r2(end - start),
      motion: medianMotion(track, start, end),
      size: '',
      category: '',
      camera: '',
      transitionIn: 'cut',
      subjects: [],
      frame: '',
      onscreenText: '',
      audio: '',
      note: msgs(doc?.lang)('recutNote', from.join('+') || '—'),
    });
  }

  const manualCuts = [...new Set([...(doc.manualCuts ?? []), ...splits.map(r2)])]
    .filter((t) => sorted.some((b) => Math.abs(b - t) <= 0.011))
    .sort((a, b) => a - b);

  return { ...doc, manualCuts, shots };
}

/* ------------------------------------------------------------------ */
/* 统计：报告里的每个数字都在这里算，不从 JSON 读                        */
/* ------------------------------------------------------------------ */

export function stats(doc) {
  const shots = doc?.shots ?? [];
  const secs = shots.map((s) => Number(s.seconds) || 0);
  const total = r2(secs.reduce((a, b) => a + b, 0));
  const sorted = [...secs].sort((a, b) => a - b);
  const median = sorted.length ? (sorted.length % 2 ? sorted[sorted.length >> 1] : (sorted[(sorted.length >> 1) - 1] + sorted[sorted.length >> 1]) / 2) : 0;
  const tally = (key, table) => {
    const map = new Map();
    for (const s of shots) {
      const k = s[key] || '—';
      const cur = map.get(k) ?? { key: k, zh: table[k]?.zh ?? k, count: 0, seconds: 0 };
      cur.count += 1;
      cur.seconds = r2(cur.seconds + (Number(s.seconds) || 0));
      map.set(k, cur);
    }
    return [...map.values()].sort((a, b) => b.seconds - a.seconds);
  };
  return {
    count: shots.length,
    totalSeconds: total,
    avgSeconds: shots.length ? r2(total / shots.length) : 0,
    medianSeconds: r2(median),
    minSeconds: sorted.length ? r2(sorted[0]) : 0,
    maxSeconds: sorted.length ? r2(sorted[sorted.length - 1]) : 0,
    cutsPerMinute: total ? r1((shots.length / total) * 60) : 0,
    sizes: tally('size', SHOT_SIZES),
    categories: tally('category', SHOT_CATEGORIES),
    cameras: tally('camera', CAMERA_MOVES),
    rhythms: (doc?.shots ?? []).some((x) => x.rhythm) ? tally('rhythm', RHYTHM_ROLES) : [],
  };
}

/* ------------------------------------------------------------------ */
/* validate：13 道门，全是代码                                          */
/* ------------------------------------------------------------------ */

/**
 * 门的名字。报告、Markdown、命令行三处共用，跟着 `--lang` 走——
 * 英文报告里印一排中文门名，等于没做英文。
 */
const GATE_LABELS = {
  zh: {
    timeline: '时间轴连续', duration: '时长自洽', numbering: '镜号纪律',
    size: '景别枚举', category: '类别枚举', camera: '运镜枚举', transition: '转场枚举',
    'frame-text': '画面描述可核对', dedup: '画面描述不重复', subjects: '主体对账',
    'category-evidence': '类别要有证据', motion: '运镜实测对账',
    boundary: '边界来自检测', frames: '关键帧齐全', rhythm: '节奏分析可核对',
  },
  en: {
    timeline: 'Timeline is continuous', duration: 'Durations add up', numbering: 'Shot numbering',
    size: 'Shot size vocabulary', category: 'Category vocabulary', camera: 'Camera vocabulary',
    transition: 'Transition vocabulary', 'frame-text': 'Frame description is checkable',
    dedup: 'No duplicate descriptions', subjects: 'Subjects reconcile with cast',
    'category-evidence': 'Categories carry evidence', motion: 'Camera vs. measured motion',
    boundary: 'Boundaries come from detection', frames: 'Keyframes present',
    rhythm: 'Rhythm annotation is checkable',
  },
};

export const gateLabel = (id, lang) => GATE_LABELS[lang === 'en' ? 'en' : 'zh'][id] ?? id;

/**
 * 校验与命令行的每一句话。两种语言各写一遍，不靠拼接——
 * 中文的「第 3 个镜头」和英文的 "shot 3" 语序不一样，拼不出来。
 */
const MSG = {
  recutNote: {
    zh: (from) => `边界变过（原 ${from}），标注已清空，重看画面再填`,
    en: (from) => `boundary changed (was ${from}); annotations cleared — look at the frames again`,
  },
  noShots: { zh: () => '没有任何镜头', en: () => 'there are no shots at all' },
  notNumber: { zh: (id) => `${id}：start/end 不是数字`, en: (id) => `${id}: start/end is not a number` },
  endBeforeStart: {
    zh: (id, end, start) => `${id}：end(${end}) 不大于 start(${start})`,
    en: (id, end, start) => `${id}: end (${end}) is not after start (${start})`,
  },
  notFromZero: {
    zh: (id, start) => `${id}：首镜不是从 0.00 开始（${start}）`,
    en: (id, start) => `${id}: the first shot does not start at 0.00 (it starts at ${start})`,
  },
  gap: {
    zh: (id, d, prev, start) => `${id}：与上一镜之间漏了 ${d} 秒（${prev} → ${start}）`,
    en: (id, d, prev, start) => `${id}: ${d}s is unaccounted for before this shot (${prev} → ${start})`,
  },
  overlap: {
    zh: (id, d, prev, start) => `${id}：与上一镜重叠 ${d} 秒（${prev} → ${start}）`,
    en: (id, d, prev, start) => `${id}: overlaps the previous shot by ${d}s (${prev} → ${start})`,
  },
  tail: {
    zh: (id, end, total, diff) => `${id}：末镜收在 ${end}，片长 ${total}，差 ${diff} 秒`,
    en: (id, end, total, diff) => `${id}: the last shot ends at ${end} but the film runs ${total} — ${diff}s short`,
  },
  secondsMismatch: {
    zh: (id, got, want) => `${id}：seconds=${got}，end−start=${want}`,
    en: (id, got, want) => `${id}: seconds=${got} but end−start=${want}`,
  },
  tooShort: {
    zh: (id, sec, min) => `${id}：只有 ${sec} 秒（短于 ${min}），必须在 note 里说明是闪切还是检测碎片`,
    en: (id, sec, min) => `${id}: only ${sec}s (under ${min}) — note must say whether it is a flash cut or detector noise`,
  },
  wrongId: {
    zh: (i, got, want) => `第 ${i} 个镜头的 id 是 ${got ?? '(空)'}，应为 ${want}`,
    en: (i, got, want) => `shot ${i} has id ${got ?? '(empty)'}, expected ${want}`,
  },
  enumEmpty: {
    zh: (id, label) => `${id}：${label} 还没填`,
    en: (id, label) => `${id}: ${label} is still empty`,
  },
  enumBad: {
    zh: (id, value, keys) => `${id}：${value} 不在词表里（可选：${keys}）`,
    en: (id, value, keys) => `${id}: ${value} is not in the vocabulary (choose from: ${keys})`,
  },
  transitionBad: {
    zh: (id, value) => `${id}：transitionIn=${value} 不在词表里`,
    en: (id, value) => `${id}: transitionIn=${value} is not in the vocabulary`,
  },
  frameEmpty: {
    zh: (id) => `${id}：画面描述是空的`,
    en: (id) => `${id}: the frame description is empty`,
  },
  frameShort: {
    zh: (id, got, min) => `${id}：画面描述只有 ${got} 字（至少 ${min}）`,
    en: (id, got, min) => `${id}: the frame description is only ${got} words (at least ${min})`,
  },
  frameVague: {
    zh: (id, words) => `${id}：画面描述里有空话「${words}」——换成看得见的东西`,
    en: (id, words) => `${id}: the frame description leans on "${words}" — replace it with something visible`,
  },
  frameFiller: {
    zh: (id) => `${id}：画面描述用「这个镜头…」开头——镜头表里每行都是镜头，直接写画面`,
    en: (id) => `${id}: the frame description opens with "this shot…" — every row is a shot, just describe the frame`,
  },
  dedup: {
    zh: (id, other) => `${id}：画面描述与 ${other} 一字不差——两镜真一样也要写出差别（机位、动作进度、景别）`,
    en: (id, other) => `${id}: the frame description is word-for-word identical to ${other} — even a repeat needs its difference written down (angle, how far the action has got, size)`,
  },
  subjectUnknown: {
    zh: (id, sub) => `${id}：主体 ${sub} 不在 cast 里`,
    en: (id, sub) => `${id}: subject ${sub} is not in the cast`,
  },
  needAudio: {
    zh: (id) => `${id}：类别是「对话」却没记台词（audio 空）`,
    en: (id) => `${id}: category is "dialogue" but no line was recorded (audio is empty)`,
  },
  needText: {
    zh: (id) => `${id}：类别是「字卡」却没记画面文字（onscreenText 空）`,
    en: (id) => `${id}: category is "text card" but no on-screen text was recorded (onscreenText is empty)`,
  },
  needSubject: {
    zh: (id) => `${id}：类别是「反应」却没写是谁在反应（subjects 空）`,
    en: (id) => `${id}: category is "reaction" but nobody is reacting (subjects is empty)`,
  },
  needEmpty: {
    zh: (id, who) => `${id}：类别是「空镜」却写了主体 ${[].concat(who).join('、')}`,
    en: (id, who) => `${id}: category is "empty" but subjects lists ${[].concat(who).join(', ')}`,
  },
  motionTooStill: {
    zh: (id, move, m, max) => `${id}：写的是「${move}」，实测帧间变化只有 ${m}（< ${max}）——这一镜画面没动，重看一遍`,
    en: (id, move, m, max) => `${id}: annotated "${move}" but the measured frame change is only ${m} (< ${max}) — nothing moved, look again`,
  },
  motionTooBusy: {
    zh: (id, move, m) => `${id}：写的是「${move}」，实测帧间变化 ${m} 偏高——若是主体在动就对，若是机位在动要改运镜`,
    en: (id, move, m) => `${id}: annotated "${move}" but the measured frame change is ${m} — fine if the subject is moving, but if the camera moved the annotation needs fixing`,
  },
  boundaryUndeclared: {
    zh: (id, start) => `${id}：起点 ${start} 既不在检测切点上，也没写进 manualCuts——自己加的刀要声明`,
    en: (id, start) => `${id}: start ${start} is neither a detected cut nor listed in manualCuts — a cut you added must be declared`,
  },
  frameMissing: {
    zh: (id) => `${id}：缺关键帧 ${id}a.jpg`,
    en: (id) => `${id}: keyframe ${id}a.jpg is missing`,
  },
  rhythmBadRole: {
    zh: (id, value, keys) => `${id}：节奏角色 ${value} 不在词表里（可选：${keys}）`,
    en: (id, value, keys) => `${id}: rhythm role ${value} is not in the vocabulary (choose from: ${keys})`,
  },
  rhythmHalfDone: {
    zh: (done, all) => `只标了 ${done}/${all} 镜的节奏——要么整片都标，要么一镜都不标，半张表汇总不出东西`,
    en: (done, all) => `rhythm is annotated on only ${done}/${all} shots — annotate the whole film or none of it; half a table aggregates to nothing`,
  },
  rhythmNoteEmpty: {
    zh: (id, role) => `${id}：标了「${role}」却没写为什么（rhythmNote 空）`,
    en: (id, role) => `${id}: tagged "${role}" with no reason given (rhythmNote is empty)`,
  },
  rhythmNoteShort: {
    zh: (id, got, min) => `${id}：节奏理由只有 ${got} 字（至少 ${min}）`,
    en: (id, got, min) => `${id}: the rhythm reason is only ${got} words (at least ${min})`,
  },
  rhythmNoteVague: {
    zh: (id, words) => `${id}：节奏理由里有空话「${words}」——写观众在这一刻看到什么、为什么不划走`,
    en: (id, words) => `${id}: the rhythm reason leans on "${words}" — say what the viewer sees here and why they stay`,
  },
  hintNoHook: {
    zh: (sec) => `开篇 ${sec} 秒内没有任何「钩子」——短视频的去留就在这几秒，回头看看第一镜到底给了什么`,
    en: (sec) => `no "hook" in the first ${sec}s — that window decides whether a short video keeps the viewer; look at the opening shots again`,
  },
  hintPayoffNoSetup: {
    zh: (id) => `${id}：标了「兑现」，但它前面没有任何「铺垫」或「递进」——兑现的是什么？`,
    en: (id) => `${id}: tagged "payoff" but nothing before it is "setup" or "build" — what is being paid off?`,
  },
  hintFlat: {
    zh: (from, to, role, n) => `${from}–${to} 连着 ${n} 镜都是「${role}」——节奏在这一段是平的，观众最容易在这里走`,
    en: (from, to, role, n) => `${from}–${to}: ${n} shots in a row are all "${role}" — the rhythm flattens here, and that is where viewers leave`,
  },
  skipNoCast: { zh: () => '没有声明 cast，跳过（视为通过）', en: () => 'no cast declared — skipped (counts as passed)' },
  skipNoTrack: { zh: () => '没有给 --track，跳过（视为通过）', en: () => 'no --track given — skipped (counts as passed)' },
  skipNoSeed: { zh: () => '文档里没有 seedCuts，跳过（视为通过）', en: () => 'no seedCuts in the document — skipped (counts as passed)' },
  skipNoFrameDir: {
    zh: (dir) => (dir ? `${dir}/ 不存在，跳过（视为通过）` : '没有检查关键帧目录，跳过（视为通过）'),
    en: (dir) => (dir ? `${dir}/ does not exist — skipped (counts as passed)` : 'keyframe directory not checked — skipped (counts as passed)'),
  },
  // 命令行
  cliHints: { zh: () => '提示（不拦）：', en: () => 'Hints (not blocking):' },
  cliSummary: {
    zh: (n, total, avg, rate) => `${n} 镜 / ${total} 秒 / 平均 ${avg} 秒 / 每分钟 ${rate} 切`,
    en: (n, total, avg, rate) => `${n} shots / ${total}s / ${avg}s average / ${rate} cuts per minute`,
  },
  cliFailed: {
    zh: (n) => `${n} 道门没过，逐条修完重跑。`,
    en: (n) => `${n} gates failed — fix them one by one and run again.`,
  },
  cliSeed: {
    zh: (d, fps, w, h, cuts, shots) => `[seed] ${d}s / ${fps}fps / ${w}x${h} → 检测 ${cuts} 个切点，合并后 ${shots} 镜`,
    en: (d, fps, w, h, cuts, shots) => `[seed] ${d}s / ${fps}fps / ${w}x${h} → ${cuts} cuts detected, ${shots} shots after merging`,
  },
  cliTrack: {
    zh: (path, n, hz) => `[seed] 运动曲线 → ${path}（${n} 个采样点 @ ${hz}Hz）`,
    en: (path, n, hz) => `[seed] motion curve → ${path} (${n} samples @ ${hz}Hz)`,
  },
  cliRecut: {
    zh: (from, to, splits, merges) => `[recut] ${from} 镜 → ${to} 镜（补 ${splits} 刀 / 并 ${merges} 刀）`,
    en: (from, to, splits, merges) => `[recut] ${from} shots → ${to} shots (${splits} added / ${merges} merged)`,
  },
  cliRecutNoTrack: {
    zh: () => '[recut] 没给 --track，新镜头的实测运动是空的',
    en: () => '[recut] no --track given — measured motion is empty on the new shots',
  },
  cliFrames: { zh: (n, dir) => `[frames] ${n} 张 → ${dir}/`, en: (n, dir) => `[frames] ${n} files → ${dir}/` },
  cliFrameFail: { zh: (id) => `[frames] ${id} 抽帧失败，跳过`, en: (id) => `[frames] could not extract ${id}, skipping` },
  cliSheet: {
    zh: (out, from, to) => `[sheet] ${out}（${from}–${to}，行优先）`,
    en: (out, from, to) => `[sheet] ${out} (${from}–${to}, row-major)`,
  },
  cliSheetFail: { zh: (out) => `[sheet] ${out} 生成失败，跳过`, en: (out) => `[sheet] could not build ${out}, skipping` },
  cliSheetNone: {
    zh: (pick) => `[sheet] 没有可用的 ${pick} 帧，先跑 frames`,
    en: (pick) => `[sheet] no ${pick} frames available — run frames first`,
  },
};

/** 文案的全部键名，供自测逐条对账中英两套都在。 */
export const MESSAGE_KEYS = Object.keys(MSG);

/** 取一套语言的文案：`const M = msgs(lang); M('gap', id, ...)` */
export const msgs = (lang) => (key, ...args) => MSG[key][lang === 'en' ? 'en' : 'zh'](...args);

const gate = (id, lang, issues, skipped = null) => ({
  id,
  label: gateLabel(id, lang),
  ok: skipped ? true : issues.length === 0,
  skipped,
  issues,
});

export function validate(doc, ctx = {}) {
  // 语言优先级和报告一致：--lang > JSON 顶层 lang > 默认中文。
  // 门的名字、违规信息、跳过理由都吃这一个值，否则英文报告里会混一段中文。
  const lang = (ctx.lang ?? doc?.lang) === 'en' ? 'en' : 'zh';
  const M = msgs(lang);
  const p = paramsOf(doc);
  const shots = doc?.shots ?? [];
  const total = Number(doc?.meta?.durationSeconds) || 0;
  const gates = [];
  const hints = [];

  /* 1. 时间轴连续：按时间排序、首尾相接、从 0 开始、到片尾结束 */
  {
    const bad = [];
    if (!shots.length) bad.push(M('noShots'));
    shots.forEach((s, i) => {
      const start = Number(s.start);
      const end = Number(s.end);
      if (!Number.isFinite(start) || !Number.isFinite(end)) { bad.push(M('notNumber', s.id)); return; }
      if (end <= start) bad.push(M('endBeforeStart', s.id, end, start));
      if (i === 0 && Math.abs(start) > p.boundaryTolerance) bad.push(M('notFromZero', s.id, start));
      if (i > 0) {
        const prev = Number(shots[i - 1].end);
        const d = start - prev;
        if (Math.abs(d) > p.boundaryTolerance) {
          bad.push(d > 0 ? M('gap', s.id, r2(d), prev, start) : M('overlap', s.id, r2(-d), prev, start));
        }
      }
      if (i === shots.length - 1 && total && Math.abs(end - total) > p.endTolerance) {
        bad.push(M('tail', s.id, end, total, r2(Math.abs(end - total))));
      }
    });
    gates.push(gate('timeline', lang, bad));
  }

  /* 2. 时长自洽：seconds 必须等于 end − start，短镜必须带 note */
  {
    const bad = [];
    for (const s of shots) {
      const want = r2(Number(s.end) - Number(s.start));
      if (!Number.isFinite(want)) continue;
      if (Math.abs(Number(s.seconds) - want) > 0.011) bad.push(M('secondsMismatch', s.id, s.seconds, want));
      if (want > 0 && want < p.minShotSeconds && !String(s.note ?? '').trim()) {
        bad.push(M('tooShort', s.id, want, p.minShotSeconds));
      }
    }
    gates.push(gate('duration', lang, bad));
  }

  /* 3. 镜号纪律：S01 起、两位数、连号、唯一 */
  {
    const bad = [];
    shots.forEach((s, i) => {
      const want = shotNo(i);
      if (s.id !== want) bad.push(M('wrongId', i + 1, s.id, want));
    });
    gates.push(gate('numbering', lang, bad));
  }

  /* 4–6. 三张词表：景别 / 类别 / 运镜 */
  for (const [id, key, table] of [['size', 'size', SHOT_SIZES], ['category', 'category', SHOT_CATEGORIES], ['camera', 'camera', CAMERA_MOVES]]) {
    const bad = [];
    for (const s of shots) {
      const v = s[key];
      if (!v) { bad.push(M('enumEmpty', s.id, gateLabel(id, lang))); continue; }
      if (!table[v]) bad.push(M('enumBad', s.id, v, Object.keys(table).join(' / ')));
    }
    gates.push(gate(id, lang, bad));
  }

  /* 7. 转场枚举：可省略，写了就得在表里 */
  {
    const bad = [];
    for (const s of shots) {
      if (s.transitionIn == null || s.transitionIn === '') continue;
      if (!TRANSITIONS[s.transitionIn]) bad.push(M('transitionBad', s.id, s.transitionIn));
    }
    gates.push(gate('transition', lang, bad));
  }

  /*
   * 8. 画面描述可核对：非空、够长、没有空话、不用废话开头。
   *
   * **判据跟着描述本身的语言走，不跟着 --lang 走**——中文界面下拉英文片是常事。
   * 中文数字数（12 字是一句话），英文数词数（12 个字符只有两个单词，等于没设门）。
   */
  {
    const bad = [];
    for (const s of shots) {
      const text = String(s.frame ?? '').trim();
      if (!text) { bad.push(M('frameEmpty', s.id)); continue; }
      const cjk = CJK.test(text);
      if (cjk) {
        const chars = text.replace(/\s+/g, '').length;
        if (chars < p.minFrameChars) bad.push(M('frameShort', s.id, `${chars} 字`, `${p.minFrameChars} 字`));
      } else {
        const words = text.split(/\s+/).filter(Boolean).length;
        if (words < p.minFrameWords) bad.push(M('frameShort', s.id, words, p.minFrameWords));
      }
      const lower = text.toLowerCase();
      const hitAll = (cjk ? VAGUE_WORDS : VAGUE_WORDS_EN).filter((w) => lower.includes(w.toLowerCase()));
      // 「stunning」被「visually stunning」包住时只报长的那条，不重复点名
      const vague = hitAll.filter((w) => !hitAll.some((o) => o !== w && o.toLowerCase().includes(w.toLowerCase())));
      if (vague.length) bad.push(M('frameVague', s.id, vague.join(cjk ? '」「' : '", "')));
      if ((cjk ? FILLER_OPENERS : FILLER_OPENERS_EN).some((re) => re.test(text))) bad.push(M('frameFiller', s.id));
    }
    gates.push(gate('frame-text', lang, bad));
  }

  /* 9. 画面描述不重复：整句照抄上一镜 = 没看第二眼 */
  {
    const bad = [];
    const seen = new Map();
    for (const s of shots) {
      const text = String(s.frame ?? '').trim();
      if (!text) continue;
      if (seen.has(text)) bad.push(M('dedup', s.id, seen.get(text)));
      else seen.set(text, s.id);
    }
    gates.push(gate('dedup', lang, bad));
  }

  /* 10. 主体对账：subjects 里的编号必须在顶层 cast 里（没 cast 就明说跳过） */
  {
    const cast = doc?.cast ?? [];
    if (!cast.length) {
      gates.push(gate('subjects', lang, [], M('skipNoCast')));
    } else {
      const ids = new Set(cast.map((c) => c.id));
      const bad = [];
      for (const s of shots) {
        for (const sub of s.subjects ?? []) {
          if (!ids.has(sub)) bad.push(M('subjectUnknown', s.id, sub));
        }
      }
      gates.push(gate('subjects', lang, bad));
    }
  }

  /* 11. 类别要有证据：说是对话就得有台词，说是字卡就得有画面文字 */
  {
    const bad = [];
    for (const s of shots) {
      const need = SHOT_CATEGORIES[s.category]?.evidence;
      if (!need) continue;
      const subs = s.subjects ?? [];
      if (need === 'audio' && !String(s.audio ?? '').trim()) bad.push(M('needAudio', s.id));
      if (need === 'onscreenText' && !String(s.onscreenText ?? '').trim()) bad.push(M('needText', s.id));
      if (need === 'subjects' && !subs.length) bad.push(M('needSubject', s.id));
      if (need === 'no-subjects' && subs.length) bad.push(M('needEmpty', s.id, subs));
    }
    gates.push(gate('category-evidence', lang, bad));
  }

  /*
   * 12. 运镜实测对账（给 --track 才查）。
   *
   * 只拦一个方向：**声称整幅画面在动，实测却几乎不动**——摄影机真动了，
   * 像素不可能不变，这个方向没有误拦。反过来（声称固定、实测很动）不拦：
   * 固定机位前面有人跳舞，帧间差一样会爆。它进提示。
   */
  {
    const track = ctx.track;
    if (!track) {
      gates.push(gate('motion', lang, [], M('skipNoTrack')));
    } else {
      const bad = [];
      for (const s of shots) {
        const move = CAMERA_MOVES[s.camera];
        if (!move) continue;
        const m = medianMotion(track, Number(s.start), Number(s.end));
        if (m == null) continue;
        // 短镜采样点太少，一个尖峰就能翻案——给值不设门。
        if ((Number(s.seconds) || 0) < p.motionGateMinSeconds) continue;
        const name = labelOf(CAMERA_MOVES, s.camera, lang);
        if (move.motion === 'strong' && m < p.staticMaxMotion) bad.push(M('motionTooStill', s.id, name, m, p.staticMaxMotion));
        if (move.motion === 'still' && m > p.busyMinMotion) hints.push(M('motionTooBusy', s.id, name, m));
      }
      gates.push(gate('motion', lang, bad));
    }
  }

  /*
   * 13. 边界来自检测：每个镜头边界要么来自 seedCuts（合并只会减边界，白送），
   *     要么写进 manualCuts 声明「这刀是我加的」。凭空挪切点过不去。
   */
  {
    const seedCuts = doc?.seedCuts;
    if (!Array.isArray(seedCuts) || !seedCuts.length) {
      gates.push(gate('boundary', lang, [], M('skipNoSeed')));
    } else {
      const allowed = [0, total, ...seedCuts, ...(doc.manualCuts ?? [])].filter((t) => Number.isFinite(t));
      const near = (t) => allowed.some((a) => Math.abs(a - t) <= p.cutTolerance);
      const bad = [];
      shots.forEach((s, i) => {
        if (i > 0 && !near(Number(s.start))) bad.push(M('boundaryUndeclared', s.id, s.start));
      });
      gates.push(gate('boundary', lang, bad));
    }
  }

  /* 14. 关键帧齐全：报告要嵌图，缺图就明说缺，不猜不骗 */
  {
    const dir = ctx.frameDir;
    if (!dir || !existsSync(dir)) {
      gates.push(gate('frames', lang, [], M('skipNoFrameDir', dir)));
    } else {
      const bad = [];
      for (const s of shots) {
        if (!existsSync(join(dir, `${s.id}a.jpg`))) bad.push(M('frameMissing', s.id));
      }
      gates.push(gate('frames', lang, bad));
    }
  }

  /*
   * 15. 节奏分析可核对。字段是**可选**的——整片不标也行；但**标了就得标全**，
   *     半张表汇总不出任何东西。标了的镜头必须写清为什么（和画面描述同一条规矩：
   *     写观众看到什么，不写「很有节奏感」）。
   */
  {
    const bad = [];
    const tagged = shots.filter((s) => String(s.rhythm ?? '').trim());
    if (tagged.length && tagged.length < shots.length) bad.push(M('rhythmHalfDone', tagged.length, shots.length));
    for (const s of shots) {
      const role = String(s.rhythm ?? '').trim();
      if (!role) continue;
      if (!RHYTHM_ROLES[role]) {
        bad.push(M('rhythmBadRole', s.id, role, Object.keys(RHYTHM_ROLES).join(' / ')));
        continue;
      }
      const note = String(s.rhythmNote ?? '').trim();
      if (!note) { bad.push(M('rhythmNoteEmpty', s.id, labelOf(RHYTHM_ROLES, role, lang))); continue; }
      const cjk = CJK.test(note);
      if (cjk) {
        const chars = note.replace(/\s+/g, '').length;
        if (chars < p.minRhythmChars) bad.push(M('rhythmNoteShort', s.id, `${chars} 字`, `${p.minRhythmChars} 字`));
      } else {
        const words = note.split(/\s+/).filter(Boolean).length;
        if (words < p.minRhythmWords) bad.push(M('rhythmNoteShort', s.id, words, p.minRhythmWords));
      }
      const lower = note.toLowerCase();
      const hitAll = (cjk ? VAGUE_WORDS : VAGUE_WORDS_EN).filter((w) => lower.includes(w.toLowerCase()));
      const vague = hitAll.filter((w) => !hitAll.some((o) => o !== w && o.toLowerCase().includes(w.toLowerCase())));
      if (vague.length) bad.push(M('rhythmNoteVague', s.id, vague.join(cjk ? '」「' : '", "')));
    }
    gates.push(gate('rhythm', lang, bad));

    /*
     * 节奏的三条提示都**不拦**：它们是导演判断，不是对错。
     * 但它们指的是短视频最常掉人的三个地方，值得回头看一眼。
     */
    if (tagged.length === shots.length && shots.length) {
      const roleOf = (s) => String(s.rhythm ?? '').trim();
      const early = shots.filter((s) => Number(s.start) < p.hookWindowSeconds);
      if (early.length && !early.some((s) => roleOf(s) === 'hook')) hints.push(M('hintNoHook', p.hookWindowSeconds));

      const firstPayoff = shots.findIndex((s) => roleOf(s) === 'payoff');
      if (firstPayoff > -1 && !shots.slice(0, firstPayoff).some((s) => ['setup', 'build'].includes(roleOf(s)))) {
        hints.push(M('hintPayoffNoSetup', shots[firstPayoff].id));
      }

      let run = 1;
      for (let i = 1; i <= shots.length; i += 1) {
        if (i < shots.length && roleOf(shots[i]) === roleOf(shots[i - 1])) { run += 1; continue; }
        if (run >= p.flatRun) {
          hints.push(M('hintFlat', shots[i - run].id, shots[i - 1].id,
            labelOf(RHYTHM_ROLES, roleOf(shots[i - 1]), lang), run));
        }
        run = 1;
      }
    }
  }

  const failed = gates.filter((g) => !g.ok);
  return { gates, hints, ok: failed.length === 0, failed };
}

/* ------------------------------------------------------------------ */
/* 报告                                                                */
/* ------------------------------------------------------------------ */

const I18N = {
  zh: {
    title: '拉片报告', shots: '镜头数', total: '总时长', avg: '平均镜长', median: '中位镜长',
    range: '最短 / 最长', rate: '每分钟切次', pace: '镜头节奏带', dist: '分布',
    table: '镜头表', gates: '质量门', size: '景别', category: '类别', camera: '运镜',
    frame: '画面', subjects: '主体', text: '画面文字', audio: '声音', motion: '实测运动',
    transition: '转场', note: '备注', copy: '复制', copied: '已复制', export: '导出 JSON',
    sec: '秒', shotsUnit: '镜', missing: '未生成', pass: '通过', fail: '未通过', skip: '跳过',
    hints: '提示（不拦）', sound: '有声', mute: '无声', colon: '：', sep: '　', listSep: '、', counts: '片数 · 占时',
    no: '镜号', keyframes: '关键帧', span: '时间', say: '文字 · 声音', motionShort: '实测',
    rhythm: '节奏', rhythmTitle: '节奏角色', rhythmSub: '观众为什么还没划走',
    scrollHint: '窄屏自动拆成逐镜卡片，每格都带字段名',
    paceHint: '宽度表示时长 · 深浅表示景别 · 点击跳到该镜',
    // 报告界面（shell + report.js 共用同一张表）
    indexLabel: '报告内容', pageShots: '镜头明细', pageAnalysis: '统计分布', pageCast: '出场人物', pageQuality: '质量检查',
    gatePassed: '项通过', gateFailed: '项未通过', gateSkipped: '项跳过', hintCount: '条提示',
    playerTitle: '原视频', choose: '选择本地视频', ready: '就绪', playing: '播放中', paused: '已暂停', ended: '播放结束',
    videoMissing: '原视频未加载，请选择本地视频', mismatch: '视频时长与报告不一致，请确认所选文件',
    playError: '视频无法播放，请选择浏览器支持的原视频', shot: '当前镜头', noShot: '此时间无镜头标注',
    help: '播放同步高亮镜头与时长 · 点击镜头跳转 · 点击关键帧放大', progress: '当前镜头播放进度', seek: '跳转镜头',
    searchHint: '搜索镜号、画面、台词…', viewSwitch: '视图切换', viewCards: '卡片视图', viewList: '列表视图',
    sortLabel: '镜头排序', sortTimeline: '按时间顺序', sortLongest: '镜头时长 · 由长到短', sortShortest: '镜头时长 · 由短到长',
    headFrames: '镜号 / 首尾关键帧', headTime: '时间 / 时长', headTags: '景别 / 运镜',
    noMatch: '没有找到匹配的镜头', clearFilter: '清除筛选', showing: '显示',
    unitShot: '镜', unitCut: '次', unitSecond: '秒', frameA: '首帧', frameB: '尾帧',
    filterAll: '全部镜头', filterOther: '其他', exported: 'JSON 导出已开始',
    audioMark: '声', textMark: '字',
    distTitle: '景别、类别与运镜', distCaption: '占比按镜头时长计算',
    sizeTitle: '景别分布', sizeSub: '画面距离，塑造观看的关系',
    catTitle: '镜头类别', catSub: '叙事功能，组织故事的推进',
    camTitle: '运镜方式', camSub: '镜头运动，传递情绪的起伏',
    note: '影片以 {size} 为主要景别，占总时长的 {sizePct}；{cat} 占 {catPct}。最长镜头为 {longest}，最短镜头为 {shortest}。',
    castTitle: '人物与镜头', castCaption: '{n} 位人物 · 点击查看相关分镜',
    castShots: '查看 {n} 个相关镜头', castShotsOne: '查看 1 个相关镜头',
    qualityOk: '{n} 项质量检查全部通过', qualityBad: '{n} 项质量检查未通过',
    qualityNote: '时间轴、关键帧与镜头标注全部由脚本确定性校验',
    hintTitle: '{n} 条待复核提示', hintTitleOne: '1 条待复核提示',
    close: '关闭大图', lightboxHint: '← → 切换首尾帧　·　ESC 关闭',
    noscript: '请启用 JavaScript 以浏览交互式报告。原始数据也可在同目录的 shots.json 和 shots.md 中查看。',

  },
  en: {
    title: 'Shot Breakdown', shots: 'Shots', total: 'Duration', avg: 'Avg shot', median: 'Median shot',
    range: 'Min / Max', rate: 'Cuts per min', pace: 'Pace strip', dist: 'Distribution',
    table: 'Shot list', gates: 'Quality gates', size: 'Size', category: 'Category', camera: 'Camera',
    frame: 'Frame', subjects: 'Subjects', text: 'On-screen text', audio: 'Audio', motion: 'Measured motion',
    transition: 'Transition', note: 'Note', copy: 'Copy', copied: 'Copied', export: 'Export JSON',
    sec: 's', shotsUnit: '', missing: 'not generated', pass: 'pass', fail: 'fail', skip: 'skipped',
    hints: 'Hints (not blocking)', sound: 'with audio', mute: 'silent', colon: ': ', sep: '   ', listSep: ', ', counts: 'shots · share',
    no: 'No.', keyframes: 'Keyframes', span: 'Time', say: 'Text · Audio', motionShort: 'measured',
    rhythm: 'Rhythm', rhythmTitle: 'Rhythm role', rhythmSub: 'why the viewer has not swiped away',
    scrollHint: 'stacks into labelled cards on narrow screens',
    paceHint: 'width = duration · shade = shot size · click to jump',
    indexLabel: 'Contents', pageShots: 'Shots', pageAnalysis: 'Distribution', pageCast: 'Cast', pageQuality: 'Quality',
    gatePassed: 'passed', gateFailed: 'failed', gateSkipped: 'skipped', hintCount: 'hints',
    playerTitle: 'Source video', choose: 'Pick a local file', ready: 'Ready', playing: 'Playing', paused: 'Paused', ended: 'Ended',
    videoMissing: 'Source video not loaded — pick a local file', mismatch: 'Video duration does not match the report',
    playError: 'Cannot play this file — pick a format the browser supports', shot: 'Current shot', noShot: 'No shot at this time',
    help: 'Playback highlights the current shot · click a shot to jump · click a frame to enlarge',
    progress: 'Progress within the current shot', seek: 'Jump to',
    searchHint: 'Search shot no., frame, dialogue…', viewSwitch: 'View', viewCards: 'Cards', viewList: 'List',
    sortLabel: 'Sort shots', sortTimeline: 'By timeline', sortLongest: 'Duration · longest first', sortShortest: 'Duration · shortest first',
    headFrames: 'No. / first & last frame', headTime: 'Time / duration', headTags: 'Size / camera',
    noMatch: 'No shots match', clearFilter: 'Clear filters', showing: 'Showing',
    unitShot: 'shots', unitCut: '', unitSecond: 's', frameA: 'first frame', frameB: 'last frame',
    filterAll: 'All shots', filterOther: 'Other', exported: 'JSON export started',
    audioMark: 'A', textMark: 'T',
    distTitle: 'Size, category and camera', distCaption: 'Share is by runtime, not shot count',
    sizeTitle: 'Shot size', sizeSub: 'Distance — how close the audience stands',
    catTitle: 'Category', catSub: 'Narrative function — what the shot is for',
    camTitle: 'Camera', camSub: 'Movement — how the emotion travels',
    note: 'Mostly {size}, {sizePct} of the runtime; {cat} accounts for {catPct}. Longest shot {longest}, shortest {shortest}.',
    castTitle: 'Cast and shots', castCaption: '{n} in the cast · click to filter',
    castShots: 'See {n} shots', castShotsOne: 'See 1 shot',
    qualityOk: 'All {n} quality gates passed', qualityBad: '{n} quality gates failed',
    qualityNote: 'Timeline, keyframes and annotations are all checked deterministically by the script',
    hintTitle: '{n} hints to review', hintTitleOne: '1 hint to review',
    close: 'Close', lightboxHint: '← → switch first/last frame　·　ESC to close',
    noscript: 'Enable JavaScript for the interactive report. The raw data is in shots.json and shots.md next to this file.',

  },
};

const tOf = (lang) => I18N[lang === 'en' ? 'en' : 'zh'];
const labelOf = (table, key, lang) => (table[key] ? (lang === 'en' ? table[key].en : table[key].zh) : (key || '—'));

const esc = (s) => String(s ?? '')
  .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
  .replace(/"/g, '&quot;').replace(/'/g, '&#39;');

export function renderMd(doc, ctx = {}) {
  const lang = ctx.lang ?? doc.lang ?? 'zh';
  const t = tOf(lang);
  const st = stats(doc);
  const v = validate(doc, ctx);
  const name = doc.title || doc.source || t.title;
  const out = [];
  out.push(`# ${name} · ${t.title}`, '');
  out.push(`- ${t.total}${t.colon}${st.totalSeconds} ${t.sec}${t.sep}${t.shots}${t.colon}${st.count}${t.sep}${t.rate}${t.colon}${st.cutsPerMinute}`);
  out.push(`- ${t.avg}${t.colon}${st.avgSeconds} ${t.sec}${t.sep}${t.median}${t.colon}${st.medianSeconds} ${t.sec}${t.sep}${t.range}${t.colon}${st.minSeconds} / ${st.maxSeconds} ${t.sec}`);
  const paren = (x) => (lang === 'en' ? ` (${x})` : `（${x}）`);
  if (doc.meta) out.push(`- ${doc.meta.width}×${doc.meta.height}${paren(doc.meta.aspect)} · ${doc.meta.fps} fps · ${doc.meta.hasAudio ? t.sound : t.mute}`);
  out.push('');
  const TABLE_OF = {
    [t.size]: SHOT_SIZES, [t.category]: SHOT_CATEGORIES, [t.camera]: CAMERA_MOVES, [t.rhythm]: RHYTHM_ROLES,
  };
  for (const [label, rows] of [[t.size, st.sizes], [t.category, st.categories], [t.camera, st.cameras],
    ...(st.rhythms.length ? [[t.rhythm, st.rhythms]] : [])]) {
    out.push(`**${label}**${paren(t.counts)}${t.colon}${rows.map((r) => `${labelOf(TABLE_OF[label] ?? {}, r.key, lang)} ${r.count}${t.shotsUnit} · ${st.totalSeconds ? Math.round((r.seconds / st.totalSeconds) * 100) : 0}%`).join(t.sep)}`);
  }
  out.push('', `## ${t.table}`, '');
  out.push(`| # | ${t.span} | ${t.sec} | ${t.size} | ${t.category} | ${t.camera} | ${t.frame} | ${t.rhythm} | ${t.subjects} | ${t.text} | ${t.audio} |`);
  out.push('| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |');
  for (const s of doc.shots ?? []) {
    const cell = (x) => String(x ?? '').replace(/\|/g, '\\|').replace(/\n/g, ' ');
    out.push(`| ${s.id} | ${fmtTime(s.start)}—${fmtTime(s.end)} | ${s.seconds} | ${labelOf(SHOT_SIZES, s.size, lang)} | ${labelOf(SHOT_CATEGORIES, s.category, lang)} | ${labelOf(CAMERA_MOVES, s.camera, lang)} | ${cell(s.frame)} | ${cell(s.rhythm ? `${labelOf(RHYTHM_ROLES, s.rhythm, lang)}${t.colon}${s.rhythmNote ?? ''}` : '')} | ${cell((s.subjects ?? []).join(t.listSep))} | ${cell(s.onscreenText)} | ${cell(s.audio)} |`);
  }
  out.push('', `## ${t.gates}`, '');
  for (const g of v.gates) {
    out.push(`- ${g.skipped ? '⊘' : g.ok ? '✅' : '❌'} **${g.label}**${g.skipped ? paren(g.skipped) : ''}`);
    for (const issue of g.issues) out.push(`  - ${issue}`);
  }
  if (v.hints.length) {
    out.push('', `## ${t.hints}`, '');
    for (const h of v.hints) out.push(`- ${h}`);
  }
  return out.join('\n');
}

/** 报告的样式与交互是两份独立资产，改它们就能改报告，不用动生成器。 */
const readAsset = (name) => readFileSync(new URL(`./${name}`, import.meta.url), 'utf8');

/** 人物头像取哪一镜：优先「只有他一个人」的镜头，其次景别越近越好，再次出场越早越好。 */
function portraitOf(doc, castId, frames) {
  const ranked = (doc.shots ?? [])
    .filter((s) => (s.subjects ?? []).includes(castId) && (frames[s.id] ?? '').includes('a'))
    .sort((a, b) => ((a.subjects ?? []).length - (b.subjects ?? []).length)
      || ((SHOT_SIZES[b.size]?.depth ?? 0) - (SHOT_SIZES[a.size]?.depth ?? 0))
      || (a.start - b.start));
  return ranked[0]?.id ?? null;
}

/** 筛选条：按占镜头数排前四的类别，剩下的归「其他」。 */
function filterList(doc, lang, t) {
  const tally = new Map();
  for (const s of doc.shots ?? []) tally.set(s.category, (tally.get(s.category) ?? 0) + 1);
  const top = [...tally.entries()].sort((a, b) => b[1] - a[1]).slice(0, 4).map(([k]) => k);
  const out = [['all', t.filterAll], ...top.map((k) => [k, labelOf(SHOT_CATEGORIES, k, lang)])];
  if (tally.size > top.length) out.push(['other', t.filterOther]);
  return out;
}

/** 时间轴刻度：取整的步长走五格，末尾补片长。「01:21」这种刻度没人看得下去。 */
const TICK_STEPS = [1, 2, 5, 10, 15, 20, 30, 40, 60, 90, 120, 180, 240, 300, 600, 900, 1200, 1800];
function ticksOf(total) {
  const want = total / 5;
  const step = [...TICK_STEPS].reverse().find((x) => x <= want) ?? 1;
  const out = [];
  for (let i = 0; i * step < total && i < 5; i += 1) {
    const at = i * step;
    out.push(`${String(Math.floor(at / 60)).padStart(2, '0')}:${String(Math.round(at % 60)).padStart(2, '0')}`);
  }
  out.push(fmtTime(total));
  return out;
}

export function renderHtml(doc, ctx = {}) {
  const lang = ctx.lang ?? doc.lang ?? 'zh';
  const t = tOf(lang);
  const st = stats(doc);
  const v = validate(doc, ctx);
  const shots = doc.shots ?? [];
  const name = doc.title || doc.source || t.title;
  const m = doc.meta ?? {};
  const frameDir = ctx.frameRel ?? paramsOf(doc).frameDir;
  const has = ctx.frameExists ?? {};

  // 每个镜头有哪几张关键帧——缺图报告里明说缺，不摆一个会 404 的 <img>
  const frames = {};
  for (const s of shots) {
    frames[s.id] = `${has[`${s.id}a`] ? 'a' : ''}${has[`${s.id}b`] ? 'b' : ''}`;
  }
  const frameCount = Object.values(frames).reduce((a, f) => a + f.length, 0);

  const portraits = {};
  for (const c of doc.cast ?? []) {
    const shot = portraitOf(doc, c.id, frames);
    if (shot) portraits[c.id] = shot;
  }

  const failed = v.gates.filter((g) => !g.ok);
  const skipped = v.gates.filter((g) => g.skipped);
  const status = [
    `${v.gates.length - failed.length - skipped.length} ${t.gatePassed}`,
    failed.length ? `${failed.length} ${t.gateFailed}` : '',
    skipped.length ? `${skipped.length} ${t.gateSkipped}` : '',
    v.hints.length ? `${v.hints.length} ${t.hintCount}` : '',
  ].filter(Boolean).join(' · ');

  const gateItems = v.gates.map((g) => `<li class="${g.skipped ? 'skip' : g.ok ? 'ok' : 'bad'}">`
    + `<b>${esc(g.label)}</b><span class="tag">${g.skipped ? t.skip : g.ok ? t.pass : t.fail}</span>`
    + (g.skipped ? `<div class="gate-note">${esc(g.skipped)}</div>` : '')
    + (g.issues.length ? `<ul>${g.issues.map((i) => `<li>${esc(i)}</li>`).join('')}</ul>` : '')
    + '</li>').join('');

  // 提示里点名的镜号做成按钮，点一下跳到那一镜
  const hintItems = v.hints.map((h) => {
    const hit = /^(S\d+)[：:]\s*(.*)$/.exec(h);
    return hit
      ? `<p><button data-shot="${esc(hit[1])}">${esc(hit[1])} ↗</button> ${esc(hit[2])}</p>`
      : `<p>${esc(h)}</p>`;
  }).join('');

  const cfg = {
    frameDir,
    frames,
    portraits,
    exportName: `${String(doc.source || 'shots').replace(/\.[^.]+$/, '')}-shots.json`,
    labels: {
      sizes: Object.fromEntries(Object.entries(SHOT_SIZES).map(([k, x]) => [k, lang === 'en' ? x.en : x.zh])),
      cats: Object.fromEntries(Object.entries(SHOT_CATEGORIES).map(([k, x]) => [k, lang === 'en' ? x.en : x.zh])),
      cams: Object.fromEntries(Object.entries(CAMERA_MOVES).map(([k, x]) => [k, lang === 'en' ? x.en : x.zh])),
      trans: Object.fromEntries(Object.entries(TRANSITIONS).map(([k, x]) => [k, lang === 'en' ? x.en : x.zh])),
      rhythms: Object.fromEntries(Object.entries(RHYTHM_ROLES).map(([k, x]) => [k, lang === 'en' ? x.en : x.zh])),
    },
    rhythmColors: Object.fromEntries(Object.entries(RHYTHM_ROLES).map(([k, x]) => [k, x.color])),
    hasRhythm: (doc.shots ?? []).some((x) => x.rhythm),
    colors: Object.fromEntries(Object.entries(SHOT_SIZES).map(([k, x]) => [k, x.color])),
    filters: filterList(doc, lang, t),
    words: t,
  };

  const video = ctx.video ?? doc.source ?? '';
  const poster = frames[shots[0]?.id]?.includes('a') ? `${frameDir}/${shots[0].id}a.jpg` : '';
  const payload = JSON.stringify(doc).replace(/</g, '\\u003c');
  const cfgJson = JSON.stringify(cfg).replace(/</g, '\\u003c');

  return `<!doctype html>
<html lang="${lang === 'en' ? 'en' : 'zh-CN'}">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="theme-color" content="#f6f7f3"><title>${esc(name)} · ${esc(t.title)}</title>
<style>
${readAsset('report.css')}
</style>
</head>
<body>
<svg style="display:none" aria-hidden="true"><defs>
<symbol id="i-download" viewBox="0 0 24 24"><path d="M12 3v12m-4-4 4 4 4-4M4 15v5h16v-5"/></symbol>
<symbol id="i-search" viewBox="0 0 24 24"><circle cx="10" cy="10" r="6"/><path d="m15 15 5 5"/></symbol>
<symbol id="i-grid" viewBox="0 0 24 24"><rect x="3" y="3" width="7" height="7" rx="1"/><rect x="14" y="3" width="7" height="7" rx="1"/><rect x="3" y="14" width="7" height="7" rx="1"/><rect x="14" y="14" width="7" height="7" rx="1"/></symbol>
<symbol id="i-list" viewBox="0 0 24 24"><path d="M9 5h12M9 12h12M9 19h12M3 5h1M3 12h1M3 19h1"/></symbol>
<symbol id="i-right" viewBox="0 0 24 24"><path d="m10 6 6 6-6 6"/></symbol>
</defs></svg>
<div class="report">
<main class="content">

<header class="report-header">
  <div class="report-heading">
    <h1>${esc(name)} <small>${esc(t.title)}</small></h1>
    <div class="metadata">
      <span>${esc(doc.source ?? '')}</span><i></i>
      <span class="mono">${esc(m.width ?? '?')} × ${esc(m.height ?? '?')}</span><i></i>
      <span class="mono">${esc(m.fps ?? '?')} fps</span><i></i>
      <span>${esc(m.aspect ?? '')} · ${m.hasAudio ? t.sound : t.mute}</span>
    </div>
  </div>
  <div class="report-actions">
    <span class="report-status">${esc(status)}</span>
    <button class="export" id="export"><svg class="icon"><use href="#i-download"/></svg>${esc(t.export)}</button>
  </div>
</header>

<nav class="report-index" aria-label="${esc(t.indexLabel)}">
  <span class="muted">${esc(t.indexLabel)}</span>
  <button data-page="overview">${esc(t.pageShots)} <span>${st.count}</span></button>
  <button data-page="analysis">${esc(t.pageAnalysis)}</button>
  ${(doc.cast ?? []).length ? `<button data-page="cast">${esc(t.pageCast)} <span>${(doc.cast ?? []).length}</span></button>` : ''}
  <button data-page="quality">${esc(t.pageQuality)} <span>${v.gates.length}</span></button>
</nav>

<div class="stats" id="stats"></div>

<div class="page" id="overview">
  <section class="report-player" aria-label="${esc(t.playerTitle)}">
    <div class="player-media">
      <video id="report-video" controls playsinline preload="metadata" aria-label="${esc(t.playerTitle)}"${video ? ` src="${esc(video)}"` : ''}${poster ? ` poster="${esc(poster)}"` : ''}></video>
    </div>
    <div class="player-info">
      <div class="player-heading">
        <b>${esc(t.playerTitle)}</b>
        <span id="player-state" role="status">${esc(t.ready)}</span>
        <label class="player-file">${esc(t.choose)}<input id="player-file" type="file" accept="video/*" aria-label="${esc(t.choose)}"></label>
      </div>
      <div class="player-current"><span>${esc(t.shot)} <strong id="player-shot">—</strong></span><output id="player-clock">00:00.00</output></div>
      <div class="player-range"><span id="player-range">—</span><b id="player-duration">—</b></div>
      <progress id="player-progress" value="0" max="1" aria-label="${esc(t.progress)}"></progress>
      <p id="player-description"></p>
      <p id="player-beat"></p>
      <p id="player-dialogue"></p>
      <div class="player-help">${esc(t.help)}</div>
      <p id="player-error" role="status" hidden></p>
    </div>
  </section>

  <div class="timeline-panel">
    <div class="timeline-top">
      <h3>${esc(t.pace)} <span class="mono" style="margin-left:9px">SHOT TIMELINE</span></h3>
      <span>${esc(t.paceHint)}</span>
    </div>
    <div class="timeline" id="timeline" aria-label="${esc(t.pace)}"></div>
    <div class="timeline-ticks">${ticksOf(st.totalSeconds).map((x) => `<span>${esc(x)}</span>`).join('')}</div>
    <div class="legend" id="legend"></div>
  </div>

  <section class="library" id="library">
    <div class="library-header">
      <h2>${esc(t.table)} <small>${st.count}</small></h2>
      <div class="library-tools">
        <label class="search"><svg class="icon"><use href="#i-search"/></svg>
          <input id="search" type="search" placeholder="${esc(t.searchHint)}" aria-label="${esc(t.searchHint)}"></label>
        <div class="views" aria-label="${esc(t.viewSwitch)}">
          <button id="grid-view" aria-label="${esc(t.viewCards)}" aria-pressed="false"><svg class="icon"><use href="#i-grid"/></svg></button>
          <button class="active" id="list-view" aria-label="${esc(t.viewList)}" aria-pressed="true"><svg class="icon"><use href="#i-list"/></svg></button>
        </div>
      </div>
    </div>
    <div class="filterbar">
      <div class="filters" id="filters"></div>
      <div class="filter-right">
        <span id="result-count" aria-live="polite"></span>
        <select id="sort" aria-label="${esc(t.sortLabel)}">
          <option value="timeline">${esc(t.sortTimeline)}</option>
          <option value="longest">${esc(t.sortLongest)}</option>
          <option value="shortest">${esc(t.sortShortest)}</option>
        </select>
      </div>
    </div>
    <div class="list-head" id="list-head">
      <span>${esc(t.headFrames)}</span><span>${esc(t.headTime)}</span><span>${esc(t.headTags)}</span>
      <span>${esc(t.frame)}</span><span>${esc(t.say)}</span>
    </div>
    <div class="cards list" id="cards"></div>
    <div class="empty" id="empty" hidden>${esc(t.noMatch)}<br><button id="reset">${esc(t.clearFilter)}</button></div>
  </section>
</div>

<details class="report-section" id="analysis">
  <summary>${esc(t.pageAnalysis)}</summary>
  <div class="report-section-body">
    <div class="section-title"><h2>${esc(t.distTitle)}</h2><span class="section-caption">${esc(t.distCaption)}</span></div>
    <div class="analysis-note" id="analysis-note"></div>
    <div class="analysis-grid" id="distributions"></div>
  </div>
</details>

${(doc.cast ?? []).length ? `<details class="report-section" id="cast">
  <summary>${esc(t.pageCast)}</summary>
  <div class="report-section-body">
    <div class="section-title"><h2>${esc(t.castTitle)}</h2><span class="section-caption">${esc(t.castCaption.replace('{n}', (doc.cast ?? []).length))}</span></div>
    <div class="cast-grid" id="cast-grid"></div>
  </div>
</details>` : ''}

<details class="report-section" id="quality">
  <summary>${esc(t.pageQuality)}</summary>
  <div class="report-section-body">
    <div class="quality-banner${failed.length ? ' bad' : ''}">
      <div class="quality-check">${failed.length ? '!' : '✓'}</div>
      <div>
        <h3>${esc(failed.length ? t.qualityBad.replace('{n}', failed.length) : t.qualityOk.replace('{n}', v.gates.length))}</h3>
        <p>${esc(t.qualityNote)}</p>
      </div>
    </div>
    <ul class="gates">${gateItems}</ul>
    ${v.hints.length ? `<div class="hint"><b>${esc(v.hints.length === 1 ? t.hintTitleOne : t.hintTitle.replace('{n}', v.hints.length))}</b>${hintItems}</div>` : ''}
  </div>
</details>

<footer class="footer">
  <span>video-shots · ${esc(t.title)}</span>
  <span>${st.count} ${esc(t.unitShot)} · ${frameCount} ${esc(t.keyframes)} · ${esc(new Date().toISOString().slice(0, 10))}</span>
</footer>
</main>
</div>

<dialog id="lightbox" aria-label="${esc(t.keyframes)}">
  <div class="lightbox-head"><span id="lightbox-title"></span><button id="close-lightbox" aria-label="${esc(t.close)}">✕</button></div>
  <img id="lightbox-img" alt="">
  <div class="dialog-hint">${esc(t.lightboxHint)}</div>
</dialog>
<div id="toast" role="status" hidden></div>
<noscript>${esc(t.noscript)}</noscript>
<script>
const DOC=${payload};
const CFG=${cfgJson};
${readAsset('report.js')}
</script>
</body>
</html>`;
}

/* ------------------------------------------------------------------ */
/* CLI                                                                 */
/* ------------------------------------------------------------------ */

const USAGE = `video-shots.mjs — video-shots skill 的确定性工具（拉片）

  所有命令都认 --lang zh|en（默认中文）：门的名字、违规信息、命令行输出跟着切

  seed <video> [--threshold 0.3] [--min 0.3] [--track track.json] [--title 片名]
      场景检测 + 运动曲线 → 工作底稿 shots.json（stdout）。切点和时长在这一步定死。
      --track 另存运动曲线（validate 的 motion 门要它）；--no-motion 跳过运动测量

  frames <shots.json> --video <video> [--dir frames] [--width 480] [--single]
      每镜抽关键帧：<dir>/S01a.jpg（起手 15%）+ S01b.jpg（收尾 85%，--single 不抽）

  sheet <shots.json> [--dir frames] [--cols 5] [--rows 5] [--out sheets] [--pick a|b]
      把关键帧拼成联系表（行优先，S01 在左上），一张图看 25 个镜头。
      a 表和 b 表对照着看，一眼能看出哪些镜头的取景变了 = 运镜

  recut <shots.json> [--split <秒>]... [--merge <秒>]... [--track track.json]
      补刀 / 并刀 → 新的 shots.json（stdout）。自动重编号、重算时长与实测运动；
      边界没动的镜头标注原样保留，被拆被并的清空标注并在 note 里写明出身

  validate <shots.json> [--track track.json] [--frames <dir>]
      14 道质量门，全是代码。有违规退出码 1

  render <shots.json> --md|--html [--track track.json] [--frames <dir>] [--lang zh|en] [--video <路径>]
      Markdown 镜头表 / 单页报告（stdout）。--video 给报告里的播放器指原片
      （相对报告文件的路径，默认用 JSON 里的 source；播放器也能现场选本地文件）
`;

function readJson(path) {
  return JSON.parse(readFileSync(path, 'utf8'));
}

function flag(rest, name, fallback = null) {
  const i = rest.indexOf(name);
  if (i === -1) return fallback;
  const v = rest[i + 1];
  return v == null || v.startsWith('--') ? true : v;
}

function flags(rest, name) {
  const out = [];
  rest.forEach((a, i) => { if (a === name && rest[i + 1] != null && !rest[i + 1].startsWith('--')) out.push(rest[i + 1]); });
  return out;
}

function loadCtx(rest, doc) {
  const ctx = { lang: flag(rest, '--lang', null) };
  const video = flag(rest, '--video');
  if (typeof video === 'string') ctx.video = video; // 报告里的播放器去哪儿找原片
  const trackPath = flag(rest, '--track');
  if (typeof trackPath === 'string') ctx.track = readJson(trackPath);
  const frames = flag(rest, '--frames', true); // 默认就按 params.frameDir 查，缺图报告里明说
  const dir = typeof frames === 'string' ? frames : paramsOf(doc).frameDir;
  {
    ctx.frameDir = dir;
    ctx.frameRel = dir;
    ctx.frameExists = {};
    if (existsSync(dir)) {
      for (const f of readdirSync(dir)) {
        const mm = /^(S\d+[ab])\.jpg$/.exec(f);
        if (mm) ctx.frameExists[mm[1]] = true;
      }
    }
  }
  return ctx;
}

function cmdSeed(rest) {
  const M = msgs(flag(rest, '--lang'));
  const video = rest[0];
  if (!video) throw new Error('seed 要一个视频文件');
  const threshold = Number(flag(rest, '--threshold', DEFAULT_PARAMS.sceneThreshold));
  const minShotSeconds = Number(flag(rest, '--min', DEFAULT_PARAMS.minShotSeconds));
  const meta = probe(video);
  const cuts = detectCuts(video, threshold);
  const track = flag(rest, '--no-motion') ? null : motionTrack(video, DEFAULT_PARAMS.trackHz);
  const doc = buildSeed(meta, cuts, track, {
    source: basename(video),
    title: typeof flag(rest, '--title') === 'string' ? flag(rest, '--title') : '',
    lang: flag(rest, '--lang') === 'en' ? 'en' : 'zh',
    params: { sceneThreshold: threshold, minShotSeconds },
  });
  const trackOut = flag(rest, '--track');
  if (typeof trackOut === 'string' && track) writeFileSync(trackOut, JSON.stringify(track));
  process.stderr.write(`${M('cliSeed', meta.durationSeconds, meta.fps, meta.width, meta.height, cuts.length, doc.shots.length)}\n`);
  if (typeof trackOut === 'string' && track) process.stderr.write(`${M('cliTrack', trackOut, track.values.length, track.hz)}\n`);
  process.stdout.write(`${JSON.stringify(doc, null, 2)}\n`);
}

function cmdFrames(rest) {
  const M = msgs(flag(rest, '--lang'));
  const doc = readJson(rest[0]);
  const video = flag(rest, '--video');
  if (typeof video !== 'string') throw new Error('frames 要 --video <视频文件>');
  const dir = typeof flag(rest, '--dir') === 'string' ? flag(rest, '--dir') : paramsOf(doc).frameDir;
  const width = Number(flag(rest, '--width', 480));
  const single = flag(rest, '--single') === true;
  mkdirSync(dir, { recursive: true });
  let n = 0;
  for (const s of doc.shots ?? []) {
    const start = Number(s.start);
    const end = Number(s.end);
    const span = end - start;
    const picks = single ? [['a', start + span * 0.15]] : [['a', start + span * 0.15], ['b', start + span * 0.85]];
    for (const [suffix, at] of picks) {
      const out = join(dir, `${s.id}${suffix}.jpg`);
      try {
        execFileSync('ffmpeg', ['-v', 'error', '-y', '-ss', String(r2(at)), '-i', video,
          '-frames:v', '1', '-vf', `scale=${width}:-2`, '-q:v', '3', out], { stdio: 'ignore' });
        n += 1;
      } catch {
        process.stderr.write(`${M('cliFrameFail', `${s.id}${suffix}`)}\n`);
      }
    }
  }
  process.stderr.write(`${M('cliFrames', n, dir)}\n`);
}

function cmdSheet(rest) {
  const M = msgs(flag(rest, '--lang'));
  const doc = readJson(rest[0]);
  const dir = typeof flag(rest, '--dir') === 'string' ? flag(rest, '--dir') : paramsOf(doc).frameDir;
  const cols = Number(flag(rest, '--cols', 5));
  const rows = Number(flag(rest, '--rows', 5));
  const outDir = typeof flag(rest, '--out') === 'string' ? flag(rest, '--out') : 'sheets';
  const pick = flag(rest, '--pick') === 'b' ? 'b' : 'a'; // a = 起手帧联系表，b = 收尾帧（两张对照着看运镜）
  mkdirSync(outDir, { recursive: true });
  const per = cols * rows;
  const ids = (doc.shots ?? []).map((s) => s.id).filter((id) => existsSync(join(dir, `${id}${pick}.jpg`)));
  const made = [];
  for (let i = 0; i < ids.length; i += per) {
    const batch = ids.slice(i, i + per);
    const listFile = join(outDir, `.sheet-${i}.txt`);
    writeFileSync(listFile, batch.map((id) => `file '${resolve(dir, `${id}${pick}.jpg`)}'`).join('\n'));
    const out = join(outDir, `sheet-${pick}${String(Math.floor(i / per) + 1).padStart(2, '0')}.jpg`);
    try {
      execFileSync('ffmpeg', ['-v', 'error', '-y', '-f', 'concat', '-safe', '0', '-i', listFile,
        '-vf', `scale=320:-2,tile=${cols}x${rows}:padding=4:margin=4:color=white`,
        '-frames:v', '1', '-q:v', '3', out], { stdio: 'ignore' });
      made.push(M('cliSheet', out, batch[0], batch[batch.length - 1]));
    } catch {
      process.stderr.write(`${M('cliSheetFail', out)}\n`);
    }
    rmSync(listFile, { force: true });
  }
  process.stderr.write(made.length ? `${made.join('\n')}\n` : `${M('cliSheetNone', pick)}\n`);
}

function cmdRecut(rest) {
  const M = msgs(flag(rest, '--lang'));
  const doc = readJson(rest[0]);
  const trackPath = flag(rest, '--track');
  const track = typeof trackPath === 'string' ? readJson(trackPath) : null;
  const splits = flags(rest, '--split').map(Number);
  const merges = flags(rest, '--merge').map(Number);
  if (!splits.length && !merges.length) throw new Error('recut 至少要一个 --split 或 --merge');
  const next = recut(doc, { splits, merges, track });
  process.stderr.write(`${M('cliRecut', doc.shots.length, next.shots.length, splits.length, merges.length)}\n`);
  if (!track) process.stderr.write(`${M('cliRecutNoTrack')}\n`);
  process.stdout.write(`${JSON.stringify(next, null, 2)}\n`);
}

function cmdValidate(rest) {
  const doc = readJson(rest[0]);
  const ctx = loadCtx(rest, doc);
  const v = validate(doc, ctx);
  for (const g of v.gates) {
    const mark = g.skipped ? '⊘' : g.ok ? '✅' : '❌';
    process.stdout.write(`${mark} ${g.label}${g.skipped ? `　（${g.skipped}）` : ''}\n`);
    for (const issue of g.issues) process.stdout.write(`   · ${issue}\n`);
  }
  const M = msgs(ctx.lang);
  if (v.hints.length) {
    process.stdout.write(`\n${M('cliHints')}\n`);
    for (const h of v.hints) process.stdout.write(`   · ${h}\n`);
  }
  const st = stats(doc);
  process.stdout.write(`\n${M('cliSummary', st.count, st.totalSeconds, st.avgSeconds, st.cutsPerMinute)}\n`);
  if (!v.ok) {
    process.stdout.write(`\n${M('cliFailed', v.failed.length)}\n`);
    process.exitCode = 1;
  }
}

function cmdRender(rest) {
  const doc = readJson(rest[0]);
  const ctx = loadCtx(rest, doc);
  const html = flag(rest, '--html') === true;
  process.stdout.write(html ? renderHtml(doc, ctx) : renderMd(doc, ctx));
  process.stdout.write('\n');
}

export function main(argv) {
  const [cmd, ...rest] = argv;
  switch (cmd) {
    case 'seed': return cmdSeed(rest);
    case 'frames': return cmdFrames(rest);
    case 'sheet': return cmdSheet(rest);
    case 'recut': return cmdRecut(rest);
    case 'validate': return cmdValidate(rest);
    case 'render': return cmdRender(rest);
    default:
      process.stdout.write(USAGE);
      if (cmd && cmd !== '--help' && cmd !== '-h') process.exitCode = 1;
      return undefined;
  }
}

function isMainModule() {
  if (!process.argv[1]) return false;
  try {
    return realpathSync(process.argv[1]) === realpathSync(fileURLToPath(import.meta.url));
  } catch {
    return false;
  }
}

if (isMainModule()) {
  // 下游把管道关了（`| head`）就安静退出，不要吐一屏 EPIPE 栈
  process.stdout.on('error', (err) => { if (err.code === 'EPIPE') process.exit(0); });
  process.stderr.on('error', () => {});
  try {
    main(process.argv.slice(2));
  } catch (err) {
    process.stderr.write(`${err.message}\n`);
    process.exitCode = 1;
  }
}
