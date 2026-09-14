#!/usr/bin/env node
// video-sync — 把拉片数据和原片合成一条视频：画面一边，分镜信息一边，随播放切换。
// 零 npm 依赖。需要 node >= 18、ffmpeg/ffprobe，以及一个无头浏览器（面板是 HTML 渲的）。

import { execFileSync } from 'node:child_process';
import { existsSync, mkdirSync, readFileSync, readdirSync, realpathSync, rmSync, writeFileSync } from 'node:fs';
import { basename, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

/* ------------------------------------------------------------------ */
/* 常量                                                                */
/* ------------------------------------------------------------------ */
/*
 * 这个 skill 只干一件事：把 video-shots 产出的 shots.json 和原片拼成一条视频。
 *
 *   横版 / 方版原片 → 画面在上，分镜信息在下
 *   竖版原片       → 画面在左，分镜信息在右
 *
 * 信息面板是一张 HTML 页面。**只截三张图**——量一次行位置、截一张底板、
 * 截一张把镜头表铺开的长图——剩下的交给 ffmpeg：滚动是按时间裁窗，
 * 高亮是按时间画框，都是 t 的分段线性函数。逐帧截图要几千张，不干。
 * 面板的长相全在 panel.css 里，改它就能改布局，不用碰这个脚本。
 */

export const DEFAULT_PARAMS = {
  maxVideoWidth: 1920,   // 画面区的宽上限（横版按它缩）
  maxVideoHeight: 1080,  // 画面区的高上限（竖版按它缩）
  panelRatioWide: 0.8,   // 横版：面板高 ÷ 画面高
  panelRatioTall: 1.6,   // 竖版：面板宽 ÷ 画面宽
  minPanelPx: 260,       // 面板最短边，再小就放不下字
  videoScale: 1,         // 画面区相对原片的倍数。默认 1 = 只缩不放；
                         // 640x360 这种小素材放大到 2 倍才够面板排四列
  fps: 30,               // 输出帧率
  crf: 20,               // x264 质量，越小越清晰
};

/** 四张词表：与 video-shots 同名同义。**本 skill 自包含，不跨目录 import。** */
export const SHOT_SIZES = {
  none: { zh: '无景别', en: 'n/a', color: '#e4e8d9' },
  'extreme-wide': { zh: '大远景', en: 'extreme wide', color: '#dae0c8' },
  wide: { zh: '全景', en: 'wide', color: '#c4cfaa' },
  'medium-wide': { zh: '中远景', en: 'medium wide', color: '#b0c091' },
  medium: { zh: '中景', en: 'medium', color: '#94aa74' },
  'medium-close': { zh: '中近景', en: 'medium close', color: '#788f58' },
  close: { zh: '特写', en: 'close-up', color: '#526f45' },
  'extreme-close': { zh: '大特写', en: 'extreme close-up', color: '#345136' },
};

export const SHOT_CATEGORIES = {
  establishing: { zh: '定场', en: 'establishing' },
  subject: { zh: '主体', en: 'subject' },
  dialogue: { zh: '对话', en: 'dialogue' },
  reaction: { zh: '反应', en: 'reaction' },
  insert: { zh: '插入特写', en: 'insert' },
  pov: { zh: '主观', en: 'POV' },
  empty: { zh: '空镜', en: 'empty' },
  product: { zh: '产品展示', en: 'product' },
  'text-card': { zh: '字卡', en: 'text card' },
  transition: { zh: '转场镜头', en: 'transition' },
  archive: { zh: '引用素材', en: 'archive' },
};

export const CAMERA_MOVES = {
  static: { zh: '固定', en: 'static' },
  'push-in': { zh: '推', en: 'push in' },
  'pull-out': { zh: '拉', en: 'pull out' },
  'zoom-in': { zh: '变焦推', en: 'zoom in' },
  'zoom-out': { zh: '变焦拉', en: 'zoom out' },
  'pan-left': { zh: '左摇', en: 'pan left' },
  'pan-right': { zh: '右摇', en: 'pan right' },
  'tilt-up': { zh: '上摇', en: 'tilt up' },
  'tilt-down': { zh: '下摇', en: 'tilt down' },
  'truck-left': { zh: '左移', en: 'truck left' },
  'truck-right': { zh: '右移', en: 'truck right' },
  'pedestal-up': { zh: '升', en: 'pedestal up' },
  'pedestal-down': { zh: '降', en: 'pedestal down' },
  tracking: { zh: '跟拍', en: 'tracking' },
  arc: { zh: '环绕', en: 'arc' },
  'whip-pan': { zh: '甩镜', en: 'whip pan' },
  handheld: { zh: '手持微晃', en: 'handheld' },
  shake: { zh: '剧烈晃动', en: 'shake' },
  'rack-focus': { zh: '变焦点', en: 'rack focus' },
  'micro-push': { zh: '微推', en: 'micro push' },
  roll: { zh: '旋转', en: 'roll' },
  drone: { zh: '航拍移动', en: 'drone' },
};

/** 节奏角色：与 video-shots 的 RHYTHM_ROLES 同名同义（本 skill 自带一份，不跨目录 import）。 */
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

const I18N = {
  zh: {
    shots: '镜头', shot: '镜号', of: '共', duration: '时长', size: '景别', category: '类别',
    camera: '运镜', transition: '转场', frame: '画面', subjects: '主体', text: '画面文字',
    audio: '声音', motion: '实测运动', now: '当前镜头', sec: '秒',
    rhythm: '节奏分析', headShot: '镜号 · 时间 · 景别 · 运镜', headFrame: '画面', headDesc: '画面描述',
  },
  en: {
    shots: 'Shots', shot: 'Shot', of: 'of', duration: 'Duration', size: 'Size', category: 'Category',
    camera: 'Camera', transition: 'Transition', frame: 'Frame', subjects: 'Subjects', text: 'On-screen text',
    audio: 'Audio', motion: 'Measured motion', now: 'Now playing', sec: 's',
    rhythm: 'Rhythm', headShot: 'No. · time · size', headFrame: 'Frame', headDesc: 'Description',
  },
};

const tOf = (lang) => I18N[lang === 'en' ? 'en' : 'zh'];
const labelOf = (table, key, lang) => (table[key] ? (lang === 'en' ? table[key].en : table[key].zh) : (key || '—'));
const even = (n) => Math.max(2, Math.round(n / 2) * 2); // h264 要偶数边长
const r2 = (n) => Math.round(n * 100) / 100;

export function paramsOf(doc) {
  return { ...DEFAULT_PARAMS, ...(doc?.syncParams ?? {}) };
}

export function fmtTime(sec) {
  const s = Math.max(0, Number(sec) || 0);
  const m = Math.floor(s / 60);
  return `${String(m).padStart(2, '0')}:${(s - m * 60).toFixed(2).padStart(5, '0')}`;
}

const esc = (s) => String(s ?? '')
  .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
  .replace(/"/g, '&quot;').replace(/'/g, '&#39;');

/* ------------------------------------------------------------------ */
/* plan：先把几何算清楚，再动 ffmpeg                                    */
/* ------------------------------------------------------------------ */
/*
 * 版式只看原片的宽高比，不看别的：
 *   宽 ≥ 高（横版、方版）→ 上下叠（vstack），画面在上
 *   宽 <  高（竖版）     → 左右并（hstack），画面在左
 * 两个方向都保证画面**原样缩放、不裁不拉**，面板补足剩下的地方。
 */
export function plan(meta, opts = {}) {
  const p = { ...DEFAULT_PARAMS, ...opts };
  const w = Number(meta.width) || 0;
  const h = Number(meta.height) || 0;
  if (!(w > 0 && h > 0)) throw new Error('plan：原片宽高读不出来');
  const portrait = h > w;

  // 先按 videoScale 放大（默认 1，也就是老行为：只缩不放），再受上限夹一次。
  // 放大只动画面区的像素尺寸，宽高比一分不动。
  const scale = Number(p.videoScale) > 0 ? Number(p.videoScale) : 1;
  let videoW;
  let videoH;
  if (portrait) {
    videoH = Math.min(h * scale, p.maxVideoHeight);
    videoW = (w / h) * videoH;
  } else {
    videoW = Math.min(w * scale, p.maxVideoWidth);
    videoH = (h / w) * videoW;
  }
  videoW = even(videoW);
  videoH = even(videoH);

  const panel = portrait
    ? { width: even(Math.max(p.minPanelPx, videoW * p.panelRatioTall)), height: videoH }
    : { width: videoW, height: even(Math.max(p.minPanelPx, videoH * p.panelRatioWide)) };

  return {
    portrait,
    stack: portrait ? 'hstack' : 'vstack',
    video: { width: videoW, height: videoH },
    panel,
    output: portrait
      ? { width: even(videoW + panel.width), height: videoH }
      : { width: videoW, height: even(videoH + panel.height) },
    fps: p.fps,
    crf: p.crf,
  };
}

/* ------------------------------------------------------------------ */
/* 动画：滚动与高亮都是时间的函数，交给 ffmpeg 逐帧算                    */
/* ------------------------------------------------------------------ */
/*
 * 原来是「一个镜头截一张图」，面板在切点上跳变。要连续滚动只有两条路：
 * 逐帧截图（几千张，慢到不能用），或者**截一张长图、让 ffmpeg 按时间裁窗**。
 * 这里走后者：
 *
 *   list.png    整张镜头表铺开的长图（一次截图）
 *   static.png  表头与进度条的底板（一次截图）
 *   位置        由浏览器量出来（行高是 CSS 排的，脚本推不出来）
 *
 * 然后把「滚到哪」「哪一行亮」写成 t 的分段线性函数，交给 crop / drawbox 的
 * 表达式逐帧求值。截图从 N 次降到 3 次，还换来了真正的连续滚动。
 */

/**
 * 一段缓动的表达式：从 a 走到 b，用 d 秒，之后夹住不动。
 * **必须短**——ffmpeg 的表达式解析器在一百来项上就崩（53 镜拼成一条 8000 字符的
 * 表达式会让 crop 直接配置失败，踩过）。所以每个镜头只发自己这一段。
 */
export function ramp(a, b, start, dur) {
  if (a === b || !(dur > 0)) return `${r2(a)}`;
  const lo = Math.min(a, b);
  const hi = Math.max(a, b);
  return `clip(${r2(a)}+${r2(b - a)}*(t-${r2(start)})/${r2(dur)},${r2(lo)},${r2(hi)})`;
}

/** sendcmd 的命令文件：一行一条，`时刻 目标 参数 '表达式';` */
export function commandFile(commands) {
  return commands.map((c) => `${r2(c.time)} ${c.target} ${c.command} '${c.arg}';`).join('\n');
}

/**
 * 滚动与高亮的时间函数——**每个镜头一条命令**，在切点处下发。
 *
 * **当前镜头钉在第 `anchorRow + 1` 行的位置**（默认钉在第二行）：
 *   第 1 镜   列表在顶，它就是第一行，不滚
 *   第 2 镜   还在顶，高亮往下挪一行到第二行，仍然不滚
 *   第 3 镜起 每切一次正好往上滚**一行**，当前镜头始终停在第二行
 *
 * 锚点必须取**整行的位置**（`rows[1].top`），不能用「视窗高的百分之多少」——
 * 那样滚完当前行卡在两行中间，整张表看着是歪的（踩过）。
 *
 * 滚动只发生在切点：滚 `easeSeconds` 秒到位就**停住**，镜头再长也不动。
 * 高亮条和视窗用同一段缓动，是「一起滑过去」，不是一个跳一个滑。
 */
export function motionPlan({
  shots, rows, view, content, easeSeconds = 0.45, anchorRow = 1, panelHeight = 0,
}) {
  const byId = new Map(rows.map((r) => [r.id, r]));
  const row = (s) => byId.get(s.id) ?? { top: 0, height: 0 };
  // 锚点 = 第 anchorRow 行的行首，也就是「当前镜头该停在第几行」
  const slot = Math.min(Math.max(0, Math.round(anchorRow)), Math.max(0, rows.length - 1));
  const anchor = rows[slot] ? rows[slot].top - (rows[0]?.top ?? 0) : 0;
  const maxOffset = Math.max(0, content - view.height);
  const target = (s) => Math.min(maxOffset, Math.max(0, row(s).top - anchor));

  /*
   * 行高由内容决定（描述长的行更高），而 **crop 的高度改不动**——它只认 x/y 的运行时命令。
   * 所以按行高分层：出现过几种行高就建几层高亮条，当前镜头用哪种高度就点亮哪一层，
   * 其余各层挪到画面外「停车」。绝大多数片子只有一两种行高，也就一两层。
   */
  const bands = [...new Set(rows.map((r) => r.height))].sort((a, b) => a - b);
  const bandOf = (s) => Math.max(0, bands.indexOf(row(s).height));
  /*
   * 停车位要在**整块面板之外**，不只是列表视窗之外——视窗底下通常还留着进度条那一条，
   * 停在「视窗底 + 10」的话，没轮到的那层高亮条会有一截露在进度条上（踩过）。
   * 面板高度拿不到时退回「视窗底 + 最高的一行」，那也保证整条在视窗外。
   */
  const tallestRow = Math.max(0, ...rows.map((r) => r.height));
  const parkY = Math.max(panelHeight, view.y + view.height + tallestRow) + 10;
  const screenY = (s, top, offset) => {
    const hi = view.y + view.height - row(s).height;
    return `clip(${view.y}+(${top})-(${offset}),${r2(view.y)},${r2(Math.max(view.y, hi))})`;
  };

  const commands = [];
  shots.forEach((s, i) => {
    const prev = shots[i - 1];
    const start = Number(s.start);
    const span = Math.max(0.001, Number(s.end) - start);
    const ease = Math.min(easeSeconds, span);
    const band = bandOf(s);
    const prevBand = prev ? bandOf(prev) : band;
    const offset = ramp(prev ? target(prev) : target(s), target(s), start, ease);
    const top = ramp(prev ? row(prev).top : row(s).top, row(s).top, start, ease);

    commands.push({ time: start, target: 'crop@win', command: 'y', arg: offset });
    commands.push({ time: start, target: `crop@band${band}`, command: 'y', arg: top });
    commands.push({ time: start, target: `overlay@band${band}`, command: 'y', arg: screenY(s, top, offset) });

    // 别的层停到画面外。上一镜那层晚 ease 秒再停——让它陪着滑完这一程，不要在切点上凭空消失
    bands.forEach((_, j) => {
      if (j === band) return;
      commands.push({ time: j === prevBand ? start + ease : start, target: `overlay@band${j}`, command: 'y', arg: `${r2(parkY)}` });
    });
  });

  const first = shots[0];
  return {
    commands,
    bands,
    parkY,
    initial: {
      offset: first ? target(first) : 0,
      top: first ? row(first).top : 0,
      screenY: view.y + (first ? row(first).top - target(first) : 0),
      band: first ? bandOf(first) : 0,
    },
    rowHeight: Math.max(...rows.map((r) => r.height), 1),
    maxOffset,
    anchor,
    anchorRow: slot,
    easeSeconds,
  };
}

/* ------------------------------------------------------------------ */
/* 面板页面：一张 HTML，靠 #S07 这样的 hash 决定高亮哪一镜               */
/* ------------------------------------------------------------------ */

const readAsset = (name) => readFileSync(new URL(`./${name}`, import.meta.url), 'utf8');

/** 面板页面要用的数据。抽出来是为了让调参台和真产物用同一份构造逻辑。 */
export function panelData(doc, ctx = {}) {
  const lang = ctx.lang ?? doc.lang ?? 'zh';
  const t = tOf(lang);
  const shots = doc.shots ?? [];
  const total = Number(doc?.meta?.durationSeconds) || shots.reduce((a, s) => a + (Number(s.seconds) || 0), 0);
  const frameDir = ctx.frameDir ?? null;
  const has = ctx.frameExists ?? {};

  return {
    title: doc.title || doc.source || '',
    source: doc.source ?? '',
    total,
    lang,
    words: t,
    frameDir,
    cast: Object.fromEntries((doc.cast ?? []).map((c) => [c.id, c.name])),
    colors: Object.fromEntries(Object.entries(SHOT_SIZES).map(([k, x]) => [k, x.color])),
    rhythmColors: Object.fromEntries(Object.entries(RHYTHM_ROLES).map(([k, x]) => [k, x.color])),
    shots: shots.map((s) => ({
      id: s.id,
      start: Number(s.start),
      end: Number(s.end),
      seconds: Number(s.seconds),
      size: labelOf(SHOT_SIZES, s.size, lang),
      sizeKey: s.size,
      category: labelOf(SHOT_CATEGORIES, s.category, lang),
      camera: labelOf(CAMERA_MOVES, s.camera, lang),
      transition: s.transitionIn && s.transitionIn !== 'cut' ? labelOf(TRANSITIONS, s.transitionIn, lang) : '',
      frame: s.frame ?? '',
      subjects: (s.subjects ?? []).map((id) => (doc.cast ?? []).find((c) => c.id === id)?.name ?? id),
      onscreenText: s.onscreenText ?? '',
      audio: s.audio ?? '',
      motion: s.motion ?? null,
      rhythm: s.rhythm ? labelOf(RHYTHM_ROLES, s.rhythm, lang) : '',
      rhythmKey: s.rhythm ?? '',
      rhythmNote: s.rhythmNote ?? '',
      thumb: frameDir && has[`${s.id}a`] ? `${frameDir}/${s.id}a.jpg` : '',
      startText: fmtTime(s.start),
      endText: fmtTime(s.end),
    })),
  };
}

export function panelHtml(doc, layout, ctx = {}) {
  const data = panelData(doc, ctx);

  return readAsset('panel.html')
    .replace('/*__CSS__*/', readAsset('panel.css'))
    .replace('"__DATA__"', JSON.stringify(data).replace(/</g, '\\u003c'))
    .replace('"__LAYOUT__"', JSON.stringify(layout))
    .replace('__TITLE__', esc(data.title || 'panel'))
    .replace('__LANG__', data.lang === 'en' ? 'en' : 'zh-CN');
}

/* ------------------------------------------------------------------ */
/* ffmpeg / 浏览器                                                     */
/* ------------------------------------------------------------------ */

const CHROME_CANDIDATES = [
  '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
  '/Applications/Chromium.app/Contents/MacOS/Chromium',
  '/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge',
  '/usr/bin/google-chrome',
  '/usr/bin/chromium',
  '/usr/bin/chromium-browser',
  '/usr/bin/microsoft-edge',
];

export function findChrome(explicit) {
  if (explicit) {
    if (!existsSync(explicit)) throw new Error(`--chrome ${explicit} 不存在`);
    return explicit;
  }
  const found = CHROME_CANDIDATES.find((p) => existsSync(p));
  if (found) return found;
  throw new Error('没找到无头浏览器（面板是 HTML 渲的）。装 Chrome/Chromium，或用 --chrome 指路径');
}

export function probe(video) {
  const out = execFileSync('ffprobe', [
    '-v', 'error', '-print_format', 'json',
    '-show_entries', 'format=duration',
    '-show_entries', 'stream=codec_type,width,height,r_frame_rate',
    video,
  ], { encoding: 'utf8' });
  const j = JSON.parse(out);
  const v = (j.streams ?? []).find((s) => s.codec_type === 'video');
  if (!v) throw new Error(`${video} 里没有视频流`);
  const [num, den] = String(v.r_frame_rate ?? '0/1').split('/').map(Number);
  return {
    durationSeconds: r2(Number(j.format?.duration ?? 0)),
    width: v.width ?? 0,
    height: v.height ?? 0,
    fps: den ? r2(num / den) : 0,
    hasAudio: (j.streams ?? []).some((s) => s.codec_type === 'audio'),
  };
}

/* ------------------------------------------------------------------ */
/* CLI                                                                 */
/* ------------------------------------------------------------------ */

const USAGE = `video-sync.mjs — 把拉片数据和原片合成一条视频

  plan <shots.json> [--video <片>] [--panel <比例>] [--width N] [--height N] [--scale 倍数]
      只算几何：版式、画面区、面板区、输出尺寸（JSON 到 stdout），不碰视频。
      默认只缩不放；原片太小（640x360 这种）面板排不下四列，用 --scale 2 放大画面区

  panels <shots.json> --video <片> [--out panels] [--frames <关键帧目录>]
         [--lang zh|en] [--chrome <路径>] [--panel <比例>]
      量一次行位置 + 截两张图：static.png（表头与进度条的底板）、list.png（整张
      镜头表铺开的长图），外加 panel.html 供预览调样式

  compose <shots.json> --video <片> [--panels panels] [-o out.mp4] [--crf 20] [--ease 0.28]
      把画面与面板合成一条视频：横版上下叠、竖版左右并，音轨照搬原片。
      当前镜头钉在第二行：前两镜不滚，第三镜起每切一次往上滚一行，滚完就停住。
      --anchor 改钉第几行（0 = 第一行），--ease 调缓动秒数

  export <shots.json> --video <片> [-o out.mp4] [--panels panels] [其余同上]
      panels + compose 一条龙（中间产物放 --panels 指的目录，成片用 -o）

  面板的长相全在 scripts/panel.css 里，改它就能改布局，不用碰这个脚本。
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

function geometryOf(rest, doc, video) {
  const meta = video ? probe(video) : doc.meta;
  const opts = {};
  const panelRatio = flag(rest, '--panel');
  const width = flag(rest, '--width');
  const height = flag(rest, '--height');
  if (typeof panelRatio === 'string') {
    const portrait = (Number(meta.height) || 0) > (Number(meta.width) || 0);
    opts[portrait ? 'panelRatioTall' : 'panelRatioWide'] = Number(panelRatio);
  }
  if (typeof width === 'string') opts.maxVideoWidth = Number(width);
  if (typeof height === 'string') opts.maxVideoHeight = Number(height);
  const scale = flag(rest, '--scale');
  if (typeof scale === 'string') opts.videoScale = Number(scale);
  const crf = flag(rest, '--crf');
  if (typeof crf === 'string') opts.crf = Number(crf);
  return { meta, layout: plan(meta, { ...paramsOf(doc), ...opts }) };
}

function frameCtx(rest, doc) {
  const dir = flag(rest, '--frames');
  const frameDir = typeof dir === 'string' ? dir : 'frames';
  const frameExists = {};
  if (existsSync(frameDir)) {
    for (const f of readdirSync(frameDir)) {
      const m = /^(S\d+[ab])\.jpg$/.exec(f);
      if (m) frameExists[m[1]] = true;
    }
  }
  return { frameDir: existsSync(frameDir) ? resolve(frameDir) : null, frameExists };
}

/**
 * ffmpeg 的完整参数。抽出来是为了能在不跑 ffmpeg 的情况下断言它。
 *
 * 四路输入：原片、底板（表头+进度条）、暗底长图、亮条长图。
 *   滚动 = 从暗底长图按 `offset(t)` 裁一个视窗大小的窗口，盖到列表位置
 *   高亮 = 从亮条长图按 `rowTop(t)` 裁一行高，盖到这一行此刻在屏幕上的位置
 * 两个都用 crop + overlay——**只有这两个滤镜的表达式是逐帧求值的**；
 * drawbox 的表达式在初始化时算一次就定死，拿它做动画得到的是一张不动的框（踩过）。
 */
export function composeArgs({
  video, still, listDim, listLit, out, layout, motion, view, commands, hasAudio,
}) {
  const { width: vw, height: vh } = layout.video;
  const { width: pw, height: ph } = layout.panel;
  const { bands, parkY, initial } = motion;
  const chain = [
    `[0:v]scale=${vw}:${vh}:flags=lanczos,setsar=1,fps=${layout.fps}[v]`,
    `[1:v]scale=${pw}:${ph},setsar=1,fps=${layout.fps},sendcmd=f='${commands}'[base]`,
    // x 必须取 view.x：长图是整块面板宽的，列表在里面是缩进的。
    // 从 x=0 裁再盖回 x=view.x，整张表会往右挪一个缩进，右边同样宽度的字被切掉（踩过）
    `[2:v]fps=${layout.fps},crop@win=w=${view.width}:h=${view.height}:x=${view.x}:y=${initial.offset}[win]`,
    // 一种行高一层高亮条：crop 的高度是配置期定死的，改不动，只能分层
    `[3:v]fps=${layout.fps}${bands.length > 1 ? `,split=${bands.length}${bands.map((_, i) => `[lit${i}]`).join('')}` : '[lit0]'}`,
    ...bands.map((h, i) => `[lit${i}]crop@band${i}=w=${view.width}:h=${h}:x=${view.x}:y=${initial.top}[band${i}]`),
    `[base][win]overlay@win=x=${view.x}:y=${view.y}[p0]`,
    ...bands.map((h, i) => `[p${i}][band${i}]overlay@band${i}=x=${view.x}:y=${i === initial.band ? initial.screenY : parkY}[p${i + 1}]`),
    `[v][p${bands.length}]${layout.stack}=inputs=2[out]`,
  ];
  const filter = chain.join(';');

  const args = [
    '-v', 'error', '-y',
    '-i', video,
    '-loop', '1', '-i', still,
    '-loop', '1', '-i', listDim,
    '-loop', '1', '-i', listLit,
    '-filter_complex', filter,
    '-map', '[out]',
  ];
  if (hasAudio) args.push('-map', '0:a', '-c:a', 'aac', '-b:a', '160k');
  args.push('-c:v', 'libx264', '-preset', 'medium', '-crf', String(layout.crf), '-pix_fmt', 'yuv420p',
    '-movflags', '+faststart', '-shortest', out);
  return args;
}

function cmdPlan(rest) {
  const doc = readJson(rest[0]);
  const video = flag(rest, '--video');
  const { meta, layout } = geometryOf(rest, doc, typeof video === 'string' ? video : null);
  process.stdout.write(`${JSON.stringify({ source: { width: meta.width, height: meta.height }, ...layout }, null, 2)}\n`);
}

function cmdPanels(rest) {
  const doc = readJson(rest[0]);
  const video = flag(rest, '--video');
  if (typeof video !== 'string') throw new Error('panels 要 --video <片>');
  const { layout } = geometryOf(rest, doc, video);
  // --out 与 --panels 同义：export 一条龙时 -o 给成片，--panels 给中间产物
  const outDir = typeof flag(rest, '--out') === 'string' ? flag(rest, '--out')
    : typeof flag(rest, '--panels') === 'string' ? flag(rest, '--panels') : 'panels';
  const lang = flag(rest, '--lang');
  const chrome = findChrome(typeof flag(rest, '--chrome') === 'string' ? flag(rest, '--chrome') : null);
  mkdirSync(outDir, { recursive: true });

  const html = panelHtml(doc, layout, { lang: typeof lang === 'string' ? lang : undefined, ...frameCtx(rest, doc) });
  const page = join(outDir, 'panel.html');
  writeFileSync(page, html);

  const shot = (hash, w, h, file) => execFileSync(chrome, [
    '--headless', '--disable-gpu', '--hide-scrollbars', '--force-device-scale-factor=1',
    `--window-size=${w},${h}`, '--virtual-time-budget=3000',
    `--screenshot=${resolve(file)}`, `file://${resolve(page)}#${hash}`,
  ], { stdio: 'ignore' });

  // 一次 dump-dom 把每行的位置量回来：行高是 CSS 排出来的，脚本推不出来
  const dom = execFileSync(chrome, [
    '--headless', '--disable-gpu', '--hide-scrollbars', '--force-device-scale-factor=1',
    `--window-size=${layout.panel.width},${layout.panel.height}`, '--virtual-time-budget=3000',
    '--dump-dom', `file://${resolve(page)}#measure`,
  ], { encoding: 'utf8', maxBuffer: 1 << 28 });
  const hit = /<pre id="measure">([\s\S]*?)<\/pre>/.exec(dom);
  if (!hit) throw new Error('量不到行的位置——面板页面没渲出来，先用浏览器打开 panel.html 看看');
  const measure = JSON.parse(hit[1].replace(/&quot;/g, '"').replace(/&amp;/g, '&').replace(/&lt;/g, '<').replace(/&gt;/g, '>'));

  const tallHeight = Math.max(layout.panel.height, measure.content);
  if (tallHeight > 16000) {
    throw new Error(`镜头表铺开有 ${tallHeight}px，超过浏览器的截图上限——把 --panel 调大一点，或者分段拉片`);
  }

  shot('static', layout.panel.width, layout.panel.height, join(outDir, 'static.png'));
  shot('tall', layout.panel.width, tallHeight, join(outDir, 'list-dim.png'));
  shot('tall-lit', layout.panel.width, tallHeight, join(outDir, 'list-lit.png'));
  writeFileSync(join(outDir, 'layout.json'), `${JSON.stringify({ ...measure, tallHeight }, null, 2)}\n`);

  process.stderr.write(`[panels] 底板 ${layout.panel.width}×${layout.panel.height} + 长图 ×2（暗底/亮条）${layout.panel.width}×${tallHeight}，共 ${measure.rows.length} 行 → ${outDir}/\n`);
  process.stderr.write(`[panels] 面板页面 → ${page}（浏览器打开它调样式，改 panel.css 重跑即可）\n`);
}

function cmdCompose(rest) {
  const doc = readJson(rest[0]);
  const video = flag(rest, '--video');
  if (typeof video !== 'string') throw new Error('compose 要 --video <片>');
  const { meta, layout } = geometryOf(rest, doc, video);
  const panelDir = typeof flag(rest, '--panels') === 'string' ? flag(rest, '--panels') : 'panels';
  const out = typeof flag(rest, '-o') === 'string' ? flag(rest, '-o')
    : typeof flag(rest, '--out') === 'string' ? flag(rest, '--out')
      : `${basename(video).replace(/\.[^.]+$/, '')}-sync.mp4`;
  const still = join(panelDir, 'static.png');
  const listDim = join(panelDir, 'list-dim.png');
  const listLit = join(panelDir, 'list-lit.png');
  const layoutFile = join(panelDir, 'layout.json');
  for (const f of [still, listDim, listLit, layoutFile]) {
    if (!existsSync(f)) throw new Error(`${f} 不在，先跑 panels`);
  }
  const measure = readJson(layoutFile);
  const ease = flag(rest, '--ease');
  const anchor = flag(rest, '--anchor');
  const motion = motionPlan({
    shots: doc.shots ?? [],
    rows: measure.rows,
    view: measure.view,
    content: measure.content,
    easeSeconds: typeof ease === 'string' ? Number(ease) : undefined,
    anchorRow: typeof anchor === 'string' ? Number(anchor) : undefined,
    panelHeight: Number(measure.panel?.height) || layout.panel.height,
  });
  const cmdFile = join(panelDir, 'motion.cmd');
  writeFileSync(cmdFile, `${commandFile(motion.commands)}\n`);
  const args = composeArgs({
    video, still, listDim, listLit, out, layout, motion,
    view: measure.view, commands: resolve(cmdFile), hasAudio: meta.hasAudio,
  });

  execFileSync('ffmpeg', args, { stdio: ['ignore', 'ignore', 'inherit'] });
  const done = probe(out);
  process.stderr.write(`[compose] ${out} — ${done.width}×${done.height} / ${done.durationSeconds}s / ${layout.stack === 'vstack' ? '画面在上' : '画面在左'}\n`);
  process.stderr.write(`[compose] 当前镜头钉在第 ${motion.anchorRow + 1} 行，切点处滚 ${motion.easeSeconds}s 到位后停住`
    + `（全表 ${measure.content}px，行高 ${motion.bands.join('/')}px）\n`);
}

export function main(argv) {
  const [cmd, ...rest] = argv;
  switch (cmd) {
    case 'plan': return cmdPlan(rest);
    case 'panels': return cmdPanels(rest);
    case 'compose': return cmdCompose(rest);
    case 'export': { cmdPanels(rest); return cmdCompose(rest); }
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
  process.stdout.on('error', (err) => { if (err.code === 'EPIPE') process.exit(0); });
  try {
    main(process.argv.slice(2));
  } catch (err) {
    process.stderr.write(`${err.message}\n`);
    process.exitCode = 1;
  }
}
