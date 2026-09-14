#!/usr/bin/env node
// video-shots 自测：不调模型、不花额度、不碰 ffmpeg。
// 14 道质量门每一道都有**击穿用例**——证明它真的会拦，不是摆设。

import { mkdtempSync, mkdirSync, readFileSync, writeFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import {
  DEFAULT_PARAMS, SHOT_SIZES, SHOT_CATEGORIES, CAMERA_MOVES, TRANSITIONS,
  VAGUE_WORDS, VAGUE_WORDS_EN, MESSAGE_KEYS, msgs, gateLabel, RHYTHM_ROLES,
  buildSeed, recut, validate, stats, medianMotion, fmtTime, shotNo, renderMd, renderHtml, paramsOf,
} from './video-shots.mjs';

let passed = 0;
const failures = [];

function ok(cond, label) {
  if (cond) passed += 1;
  else failures.push(label);
}
const eq = (got, want, label) => ok(Object.is(got, want), `${label}（得到 ${JSON.stringify(got)}，应为 ${JSON.stringify(want)}）`);

/** 某道门是否红了。 */
function gateOf(doc, id, ctx = {}) {
  return validate(doc, ctx).gates.find((g) => g.id === id);
}
const fails = (doc, id, label, ctx = {}) => {
  const g = gateOf(doc, id, ctx);
  ok(g && !g.ok && !g.skipped, `${label} —— ${id} 门应该拦下`);
};
const holds = (doc, id, label, ctx = {}) => {
  const g = gateOf(doc, id, ctx);
  ok(g && g.ok, `${label} —— ${id} 门不该拦（${g?.issues?.join('；')}）`);
};

/* ------------------------------------------------------------------ */
/* 夹具                                                                */
/* ------------------------------------------------------------------ */

const baseDoc = () => ({
  source: 't.mp4',
  title: '测试片',
  lang: 'zh',
  meta: { durationSeconds: 10, fps: 25, width: 1920, height: 1080, aspect: '16:9', hasAudio: true },
  params: {},
  seedCuts: [4, 7],
  manualCuts: [],
  cast: [{ id: 'P1', name: '老太太' }],
  shots: [
    {
      id: 'S01', start: 0, end: 4, seconds: 4, size: 'wide', category: 'establishing', camera: 'static',
      transitionIn: 'cut', subjects: [], frame: '雪地上一排木架挂着冻肉，远处山坡发灰', onscreenText: '', audio: '', motion: 0.5,
    },
    {
      id: 'S02', start: 4, end: 7, seconds: 3, size: 'close', category: 'dialogue', camera: 'handheld',
      transitionIn: 'cut', subjects: ['P1'], frame: '老太太侧脸贴着柜台，嘴张开还在往下说', onscreenText: '', audio: '老太太：给我拿药', motion: 2,
    },
    {
      id: 'S03', start: 7, end: 10, seconds: 3, size: 'medium', category: 'empty', camera: 'push-in',
      transitionIn: 'cut', subjects: [], frame: '空院子里晾着一条毛巾，被风吹得贴在绳上', onscreenText: '', audio: '', motion: 5,
    },
  ],
});

/** 10 秒 / 5Hz：S01 区间 0.5，S02 区间 2，S03 区间 5。 */
const baseTrack = () => {
  const values = new Array(50).fill(0.5);
  for (let i = 22; i <= 33; i += 1) values[i] = 2;
  for (let i = 37; i <= 48; i += 1) values[i] = 5;
  return { hz: 5, values };
};
const CTX = () => ({ track: baseTrack() });

/* ------------------------------------------------------------------ */
/* 0. 基线：干净的稿子一道门都不响                                       */
/* ------------------------------------------------------------------ */
{
  const v = validate(baseDoc(), CTX());
  ok(v.ok, `基线稿应全绿（红的是：${v.failed.map((g) => g.id).join('、')}）`);
  eq(v.gates.length, 15, '一共 15 道门');
  eq(v.hints.length, 0, '基线没有提示');
  eq(new Set(v.gates.map((g) => g.id)).size, 15, '门的 id 不重复');
}

/* ------------------------------------------------------------------ */
/* 1. 时间轴连续                                                        */
/* ------------------------------------------------------------------ */
{
  const gap = baseDoc();
  gap.shots[1].start = 4.5;
  gap.shots[1].seconds = 2.5;
  fails(gap, 'timeline', '两镜之间漏了半秒');

  const overlap = baseDoc();
  overlap.shots[1].start = 3.5;
  overlap.shots[1].seconds = 3.5;
  fails(overlap, 'timeline', '两镜重叠');

  const late = baseDoc();
  late.shots[0].start = 0.5;
  late.shots[0].seconds = 3.5;
  fails(late, 'timeline', '首镜不从 0 开始');

  const short = baseDoc();
  short.shots[2].end = 9;
  short.shots[2].seconds = 2;
  fails(short, 'timeline', '末镜没收到片尾');

  const tiny = baseDoc();
  tiny.shots[2].end = 9.9;
  tiny.shots[2].seconds = 2.9;
  holds(tiny, 'timeline', '末镜差 0.1 秒在容差内');

  const empty = baseDoc();
  empty.shots = [];
  fails(empty, 'timeline', '一个镜头都没有');
}

/* ------------------------------------------------------------------ */
/* 2. 时长自洽                                                          */
/* ------------------------------------------------------------------ */
{
  const wrong = baseDoc();
  wrong.shots[1].seconds = 5;
  fails(wrong, 'duration', 'seconds 与 end−start 对不上');

  const flash = baseDoc();
  flash.shots[1].end = 4.2;
  flash.shots[1].seconds = 0.2;
  flash.shots[2].start = 4.2;
  flash.shots[2].seconds = 5.8;
  fails(flash, 'duration', '0.2 秒的闪切没写 note');

  const flashNoted = JSON.parse(JSON.stringify(flash));
  flashNoted.shots[1].note = '闪切，一帧插入';
  holds(flashNoted, 'duration', '闪切写了 note 就放行');
}

/* ------------------------------------------------------------------ */
/* 3. 镜号纪律                                                          */
/* ------------------------------------------------------------------ */
{
  const jump = baseDoc();
  jump.shots[1].id = 'S05';
  fails(jump, 'numbering', '镜号跳号');

  const low = baseDoc();
  low.shots[0].id = 'S1';
  fails(low, 'numbering', '镜号不补零');

  eq(shotNo(0), 'S01', 'shotNo 从 S01 起');
  eq(shotNo(11), 'S12', 'shotNo 两位补零');
}

/* ------------------------------------------------------------------ */
/* 4–6. 三张词表                                                        */
/* ------------------------------------------------------------------ */
{
  for (const [field, id, bad] of [['size', 'size', 'closeup'], ['category', 'category', 'b-roll'], ['camera', 'camera', 'zoom']]) {
    const doc = baseDoc();
    doc.shots[0][field] = bad;
    fails(doc, id, `${field}=${bad} 不在词表里`);
    const blank = baseDoc();
    blank.shots[0][field] = '';
    fails(blank, id, `${field} 空着没填`);
  }
  ok(SHOT_SIZES.none && SHOT_SIZES['extreme-close'], '景别表含黑场与大特写');
  ok(SHOT_CATEGORIES['text-card'].evidence === 'onscreenText', '字卡的证据字段是画面文字');
  eq(CAMERA_MOVES.static.motion, 'still', '固定机位属于 still 档');
  eq(CAMERA_MOVES['push-in'].motion, 'strong', '推属于 strong 档');
  eq(CAMERA_MOVES.handheld.motion, 'subtle', '手持属于 subtle 档');
}

/* ------------------------------------------------------------------ */
/* 7. 转场枚举                                                          */
/* ------------------------------------------------------------------ */
{
  const bad = baseDoc();
  bad.shots[1].transitionIn = 'flash';
  fails(bad, 'transition', '转场词不在表里');

  const omitted = baseDoc();
  delete omitted.shots[1].transitionIn;
  holds(omitted, 'transition', '转场字段可省略');
  ok(TRANSITIONS.dissolve && TRANSITIONS['match-cut'], '转场表含叠化与匹配剪辑');
}

/* ------------------------------------------------------------------ */
/* 8. 画面描述可核对                                                    */
/* ------------------------------------------------------------------ */
{
  const blank = baseDoc();
  blank.shots[0].frame = '';
  fails(blank, 'frame-text', '画面描述是空的');

  const tooShort = baseDoc();
  tooShort.shots[0].frame = '老太太说话';
  fails(tooShort, 'frame-text', '画面描述太短');

  for (const word of ['氛围感', '视觉冲击', '令人']) {
    const vague = baseDoc();
    vague.shots[0].frame = `雪地上的木架很有${word}，挂着几条冻肉在那里`;
    fails(vague, 'frame-text', `画面描述含空话「${word}」`);
  }

  const filler = baseDoc();
  filler.shots[0].frame = '这个镜头拍的是雪地里的木架，上面挂着冻肉';
  fails(filler, 'frame-text', '画面描述用「这个镜头」开头');

  ok(VAGUE_WORDS.length >= 10, '空话词表至少 10 个词');
}

/* ------------------------------------------------------------------ */
/* 9. 画面描述不重复                                                    */
/* ------------------------------------------------------------------ */
{
  const copy = baseDoc();
  copy.shots[1].frame = copy.shots[0].frame;
  fails(copy, 'dedup', '第二镜照抄第一镜的画面描述');

  const nearly = baseDoc();
  nearly.shots[1].frame = `${nearly.shots[0].frame}，机位比上一镜更近`;
  holds(nearly, 'dedup', '写出差别就放行');
}

/* ------------------------------------------------------------------ */
/* 10. 主体对账                                                         */
/* ------------------------------------------------------------------ */
{
  const ghost = baseDoc();
  ghost.shots[1].subjects = ['P9'];
  fails(ghost, 'subjects', 'P9 不在 cast 里');

  const noCast = baseDoc();
  noCast.cast = [];
  const g = gateOf(noCast, 'subjects');
  ok(g.skipped && g.ok, '没有 cast 时明说跳过并视为通过');
}

/* ------------------------------------------------------------------ */
/* 11. 类别要有证据                                                     */
/* ------------------------------------------------------------------ */
{
  const noLine = baseDoc();
  noLine.shots[1].audio = '';
  fails(noLine, 'category-evidence', '对话镜头没记台词');

  const card = baseDoc();
  card.shots[0].category = 'text-card';
  card.shots[0].size = 'none';
  fails(card, 'category-evidence', '字卡没记画面文字');

  const cardOk = JSON.parse(JSON.stringify(card));
  cardOk.shots[0].onscreenText = '十分钟前';
  holds(cardOk, 'category-evidence', '字卡记了画面文字就放行');

  const reaction = baseDoc();
  reaction.shots[0].category = 'reaction';
  fails(reaction, 'category-evidence', '反应镜头没写是谁在反应');

  const crowdedEmpty = baseDoc();
  crowdedEmpty.shots[2].subjects = ['P1'];
  fails(crowdedEmpty, 'category-evidence', '空镜里写了人');
}

/* ------------------------------------------------------------------ */
/* 12. 运镜实测对账                                                     */
/* ------------------------------------------------------------------ */
{
  const still = baseDoc();
  const ctx = CTX();
  // S01 区间实测 0.5，把它说成「推」——摄影机动了像素不可能不动
  still.shots[0].camera = 'push-in';
  fails(still, 'motion', '声称推镜但实测几乎不动', ctx);

  const tracking = baseDoc();
  tracking.shots[0].camera = 'tracking';
  fails(tracking, 'motion', '声称跟拍但实测几乎不动', ctx);

  holds(baseDoc(), 'motion', 'S03 实测 5 撑得起推镜', ctx);

  const noTrack = baseDoc();
  noTrack.shots[0].camera = 'push-in';
  const skipped = gateOf(noTrack, 'motion');
  ok(skipped.skipped && skipped.ok, '没给 --track 时明说跳过');

  // 短镜采样点太少，不设门
  const shortShot = baseDoc();
  shortShot.shots[0].end = 0.8;
  shortShot.shots[0].seconds = 0.8;
  shortShot.shots[0].camera = 'push-in';
  shortShot.shots[0].note = '闪切';
  shortShot.shots[1].start = 0.8;
  shortShot.shots[1].seconds = 6.2;
  shortShot.seedCuts = [0.8, 7];
  holds(shortShot, 'motion', '0.8 秒的镜头不查运镜', ctx);

  // 反方向：说固定、实测很动 —— 只提示不拦（主体在动也会这样）
  const busy = baseDoc();
  busy.shots[2].camera = 'static';
  const v = validate(busy, ctx);
  const mg = v.gates.find((g) => g.id === 'motion');
  ok(mg.ok, '固定机位实测偏高不拦');
  const busyTrack = baseTrack();
  for (let i = 37; i <= 48; i += 1) busyTrack.values[i] = 30;
  const v2 = validate(busy, { track: busyTrack });
  ok(v2.gates.find((g) => g.id === 'motion').ok, '主体狂动也不拦固定机位');
  ok(v2.hints.some((h) => h.includes('S03')), '但要出一条提示');
}

/* ------------------------------------------------------------------ */
/* 13. 边界来自检测                                                     */
/* ------------------------------------------------------------------ */
{
  const moved = baseDoc();
  moved.shots[0].end = 5;
  moved.shots[0].seconds = 5;
  moved.shots[1].start = 5;
  moved.shots[1].seconds = 2;
  fails(moved, 'boundary', '把刀挪到检测切点以外');

  const declared = JSON.parse(JSON.stringify(moved));
  declared.manualCuts = [5];
  holds(declared, 'boundary', '写进 manualCuts 就认');

  const merged = baseDoc();
  merged.shots = [
    { ...merged.shots[0], end: 7, seconds: 7, frame: '雪地木架上挂着冻肉，镜头一直没动过' },
    { ...merged.shots[2], id: 'S02' },
  ];
  holds(merged, 'boundary', '合并只会少边界，白送通过');

  const noSeed = baseDoc();
  delete noSeed.seedCuts;
  const g = gateOf(noSeed, 'boundary');
  ok(g.skipped && g.ok, '没有 seedCuts 时明说跳过');
}

/* ------------------------------------------------------------------ */
/* 14. 关键帧齐全                                                       */
/* ------------------------------------------------------------------ */
{
  const dir = mkdtempSync(join(tmpdir(), 'vshots-'));
  const frames = join(dir, 'frames');
  mkdirSync(frames);
  writeFileSync(join(frames, 'S01a.jpg'), 'x');
  writeFileSync(join(frames, 'S02a.jpg'), 'x');
  fails(baseDoc(), 'frames', '缺 S03 的关键帧', { frameDir: frames });
  writeFileSync(join(frames, 'S03a.jpg'), 'x');
  holds(baseDoc(), 'frames', '三张都在就通过', { frameDir: frames });
  const g = gateOf(baseDoc(), 'frames', { frameDir: join(dir, '不存在') });
  ok(g.skipped && g.ok, '关键帧目录不存在时明说跳过');
  rmSync(dir, { recursive: true, force: true });
}

/* ------------------------------------------------------------------ */
/* buildSeed：切点 → 工作底稿                                           */
/* ------------------------------------------------------------------ */
{
  const meta = { durationSeconds: 10, fps: 25, width: 640, height: 360, aspect: '16:9', hasAudio: false };
  const doc = buildSeed(meta, [3, 3.1, 6], null, { source: 'a.mp4' });
  eq(doc.shots.length, 3, '3.1 秒那个碎片被并掉');
  eq(doc.shots[0].id, 'S01', 'seed 从 S01 编号');
  eq(doc.shots[0].end, 3, '第一刀落在 3 秒');
  eq(doc.shots[2].end, 10, '末镜收在片尾');
  eq(doc.seedCuts.length, 3, 'seedCuts 保留全部原始切点（含被并掉的）');
  eq(doc.shots[1].frame, '', 'seed 不替模型写画面');
  eq(doc.shots[0].size, '', 'seed 不替模型定景别');
  ok(validate(doc).failed.some((g) => g.id === 'size'), '刚 seed 出来的底稿过不了枚举门——本来就没填');

  const clamped = buildSeed(meta, [0.05, 9.98], null, {});
  eq(clamped.shots.length, 1, '贴着首尾的碎片不单独成镜');

  const withTrack = buildSeed(meta, [5], baseTrack(), {});
  ok(withTrack.shots[0].motion != null, '给了曲线就写实测运动');

  eq(doc.lang, 'zh', '不给语言就是中文底稿');
  eq(buildSeed(meta, [5], null, { lang: 'en' }).lang, 'en', 'seed --lang en 要把语言写进底稿（不写的话英文片默认按中文判画面描述长度）');
  eq(buildSeed(meta, [5], null, { lang: 'fr' }).lang, 'zh', '只认 en，别的语言退回中文');

  eq(paramsOf({ params: { minShotSeconds: 1 } }).minShotSeconds, 1, 'params 能覆盖默认值');
  eq(paramsOf({}).trackHz, DEFAULT_PARAMS.trackHz, '没覆盖就用默认值');
}

/* ------------------------------------------------------------------ */
/* recut：补刀与并刀                                                    */
/* ------------------------------------------------------------------ */
{
  const split = recut(baseDoc(), { splits: [2], track: baseTrack() });
  eq(split.shots.length, 4, '补一刀多一镜');
  eq(split.shots.map((s) => s.id).join(','), 'S01,S02,S03,S04', '补完重新连号');
  eq(split.shots[0].end, 2, '新边界落在 2 秒');
  eq(split.shots[1].seconds, 2, '拆出来的第二半 2 秒');
  eq(split.shots[0].frame, '', '被拆的镜头标注清空');
  ok(split.shots[0].note.includes('S01'), 'note 里写明出身');
  eq(split.shots[2].frame, baseDoc().shots[1].frame, '没动过的镜头标注原样保留');
  eq(split.manualCuts.join(','), '2', '补的刀记进 manualCuts');
  ok(validate(split, CTX()).gates.find((g) => g.id === 'boundary').ok, '补完刀 boundary 门仍然通过');
  ok(split.shots[0].motion != null, '新镜头重算了实测运动');

  ok(/重看画面/.test(split.shots[0].note), '中文底稿补刀，note 是中文');
  const splitEn = recut({ ...baseDoc(), lang: 'en' }, { splits: [2] });
  ok(/look at the frames again/.test(splitEn.shots[0].note), '英文底稿补刀，note 也要是英文（不然英文报告里混一句中文）');
  ok(!/[\u4e00-\u9fa5]/.test(splitEn.shots[0].note), '英文 note 里一个汉字都不能有');

  const merged = recut(baseDoc(), { merges: [4], track: baseTrack() });
  eq(merged.shots.length, 2, '并一刀少一镜');
  eq(merged.shots[0].seconds, 7, '并出来的镜头 7 秒');
  eq(merged.shots[0].frame, '', '被并的镜头标注清空');
  eq(merged.shots[1].frame, baseDoc().shots[2].frame, '后面没动的镜头不受影响');

  const both = recut(baseDoc(), { splits: [8.5], merges: [4] });
  eq(both.shots.length, 3, '一次又并又补');

  const dropped = recut({ ...baseDoc(), manualCuts: [4] }, { merges: [4] });
  eq(dropped.manualCuts.length, 0, '并掉的刀从 manualCuts 里移除');

  let threw = 0;
  try { recut(baseDoc(), { splits: [4.02] }); } catch { threw += 1; }
  try { recut(baseDoc(), { splits: [12] }); } catch { threw += 1; }
  try { recut(baseDoc(), { merges: [5.5] }); } catch { threw += 1; }
  eq(threw, 3, '重复补刀、越界补刀、空处并刀都报错');
}

/* ------------------------------------------------------------------ */
/* 统计与格式化                                                         */
/* ------------------------------------------------------------------ */
{
  const st = stats(baseDoc());
  eq(st.count, 3, '3 个镜头');
  eq(st.totalSeconds, 10, '总时长 10 秒');
  eq(st.avgSeconds, 3.33, '平均镜长四舍五入到两位');
  eq(st.medianSeconds, 3, '中位镜长 3 秒');
  eq(st.minSeconds, 3, '最短 3 秒');
  eq(st.maxSeconds, 4, '最长 4 秒');
  eq(st.cutsPerMinute, 18, '每分钟 18 切');
  eq(st.sizes[0].key, 'wide', '景别按占时排序，全景最长');
  eq(st.sizes[0].seconds, 4, '全景占 4 秒');
  eq(st.categories.length, 3, '三种类别');

  eq(fmtTime(0), '00:00.00', '零点');
  eq(fmtTime(9.5), '00:09.50', '不到一分钟');
  eq(fmtTime(75.25), '01:15.25', '过一分钟');
  eq(fmtTime(-1), '00:00.00', '负数按零算');

  const track = baseTrack();
  eq(medianMotion(track, 0, 4), 0.5, '区间中位数');
  eq(medianMotion(track, 7, 10), 5, '另一段区间');
  eq(medianMotion(null, 0, 4), null, '没有曲线就返回 null');
  eq(medianMotion(track, 5, 5), null, '零长区间返回 null');
  const spike = baseTrack();
  spike.values[20] = 200; // 切点那一帧
  spike.values[21] = 200;
  eq(medianMotion(spike, 4, 7), 2, '切点尖峰被剔除在区间之外');
}

/* ------------------------------------------------------------------ */
/* 报告                                                                */
/* ------------------------------------------------------------------ */
const cfgOf = (html) => JSON.parse(html.split('\n').find((l) => l.startsWith('const CFG=')).slice('const CFG='.length, -1));
{
  const md = renderMd(baseDoc(), CTX());
  ok(md.includes('S01') && md.includes('S03'), 'Markdown 列出每个镜头');
  ok(md.includes('00:04.00'), 'Markdown 用时间码');
  ok(md.includes('全景') && md.includes('手持微晃'), 'Markdown 用中文标签');
  ok(renderMd(baseDoc(), { lang: 'en' }).includes('Shot list'), '--lang en 切英文界面');

  // 英文报告里不能混中文标点：主体分隔和节奏冒号都跟着语言走
  const enMd = renderMd({
    ...baseDoc(),
    lang: 'en',
    title: 'a test clip',
    cast: [{ id: 'P1', name: 'Granny' }, { id: 'P2', name: 'Doctor' }],
    shots: baseDoc().shots.map((x, i) => ({
      ...x,
      subjects: i === 1 ? ['P1', 'P2'] : x.subjects,
      frame: `a plain english frame description number ${i} for the gate`,
      audio: x.audio ? 'Granny: get me the medicine' : '',
      rhythm: 'hook',
      rhythmNote: 'the viewer stays because the frame withholds the face',
    })),
  }, {});
  ok(enMd.includes('P1, P2'), '英文报告里主体用半角逗号分隔');
  ok(enMd.includes('hook: the viewer stays'), '英文报告里节奏用半角冒号');
  ok(!/[、：（）「」]/.test(enMd), '英文报告里一个中文标点都不能有');

  // 界面语言的优先级：--lang > JSON 顶层 lang > 默认中文
  const enDoc = { ...baseDoc(), lang: 'en' };
  ok(cfgOf(renderHtml(enDoc, CTX())).words.pageShots === 'Shots', 'JSON 里写了 lang:en 就出英文');
  ok(cfgOf(renderHtml(enDoc, { ...CTX(), lang: 'zh' })).words.pageShots === '镜头明细', '--lang 压过 JSON 里的 lang');
  ok(cfgOf(renderHtml(baseDoc(), CTX())).words.pageShots === '镜头明细', '两个都没有就默认中文');
  // 切的只是标签：模型写的正文一个字都不动
  ok(renderHtml(enDoc, CTX()).includes(baseDoc().shots[0].frame), '英文界面下画面描述原样保留');
  ok(cfgOf(renderHtml(enDoc, CTX())).labels.sizes.wide === 'wide', '英文界面下词表也跟着切');
}

/*
 * HTML 报告是「壳 + 资产 + 数据」三层：壳由 renderHtml 生成，样式与交互是
 * scripts/report.css 与 report.js 两份可直接编辑的资产，页面内容由 report.js
 * 从 DOC 与 CFG 算出来。所以这里查的是**契约**，不是某一段 HTML 长什么样。
 */
{
  const doc = baseDoc();
  doc.shots[0].frame = '窗台上放着 <script>alert(1)</script> 的纸盒，边角磨白了';
  const html = renderHtml(doc, CTX());

  ok(!html.includes('<script>alert(1)</script>'), '画面描述里的标签不会原样落进页面');
  ok(html.includes('\\u003cscript'), '内嵌 JSON 的 < 被转义，截不断脚本块');
  ok(html.includes('createReportPlayer'), 'report.js 被内联进来');
  ok(html.includes('.shot-card'), 'report.css 被内联进来');
  ok(!/\$\{/.test(html.slice(0, html.indexOf('const DOC='))), '壳里没有漏替换的模板占位');
  ok(html.includes('id="report-video"'), '有播放器');
  ok(html.includes('id="timeline"') && html.includes('id="cards"') && html.includes('id="distributions"'), '四块容器齐全');
  eq((html.match(/<li class="(ok|bad|skip)"/g) ?? []).length, 15, '15 道门逐条列在页面上');
  ok(html.includes('00:00') && html.includes('00:10.00'), '时间轴刻度取整、末尾补片长');

  const cfg = cfgOf(html);
  eq(Object.keys(cfg.labels.sizes).length, Object.keys(SHOT_SIZES).length, '景别词表全量下发给页面');
  eq(Object.keys(cfg.labels.cams).length, Object.keys(CAMERA_MOVES).length, '运镜词表全量下发（不是只给用到的那几个）');
  eq(Object.keys(cfg.colors).length, Object.keys(SHOT_SIZES).length, '每个景别都有色阶');
  eq(cfg.filters[0][0], 'all', '筛选第一项是全部');
  ok(cfg.filters.some(([k]) => k === 'dialogue'), '出现过的类别进筛选条');
  eq(cfg.exportName, 't-shots.json', '导出文件名跟着片名走');

  // report.js 里引用的每一个文案键都必须存在——写漏一个，页面上就是 undefined
  const source = readFileSync(new URL('./report.js', import.meta.url), 'utf8');
  const keys = [...new Set([...source.matchAll(/\bW\.([A-Za-z]+)/g)].map((m) => m[1]))];
  ok(keys.length > 20, `report.js 至少引用 20 个文案键（实际 ${keys.length}）`);
  for (const lang of ['zh', 'en']) {
    const words = cfgOf(renderHtml(baseDoc(), { ...CTX(), lang })).words;
    const missing = keys.filter((k) => words[k] === undefined);
    ok(missing.length === 0, `${lang} 文案表缺少：${missing.join('、')}`);
  }
}

{
  // 关键帧：有就嵌，没有就明说，绝不摆一个会 404 的 img
  const cfg = cfgOf(renderHtml(baseDoc(), { ...CTX(), frameRel: 'frames', frameExists: { S01a: true, S01b: true, S02a: true } }));
  eq(cfg.frames.S01, 'ab', '两张都在');
  eq(cfg.frames.S02, 'a', '只有起手帧');
  eq(cfg.frames.S03, '', '一张都没有');
  eq(cfg.frameDir, 'frames', '关键帧目录下发给页面');

  const bare = cfgOf(renderHtml(baseDoc(), CTX()));
  eq(Object.values(bare.frames).join(''), '', '没抽帧时一张都不声称有');
}

{
  // 人物头像取「只有他一个人 + 景别最近 + 出场最早」的那一镜
  const doc = baseDoc();
  doc.cast = [{ id: 'P1', name: '老太太' }];
  doc.shots[0].subjects = ['P1'];          // S01 全景，同框只有他
  doc.shots[1].subjects = ['P1'];          // S02 特写，同框只有他
  doc.shots[2].subjects = ['P1'];
  doc.shots[2].category = 'subject';
  const all = { S01a: true, S02a: true, S03a: true };
  eq(cfgOf(renderHtml(doc, { ...CTX(), frameExists: all })).portraits.P1, 'S02', '挑最近的那一镜当头像');
  eq(cfgOf(renderHtml(doc, { ...CTX(), frameExists: { S01a: true } })).portraits.P1, 'S01', '只有一张能用就用它');
  eq(cfgOf(renderHtml(doc, CTX())).portraits.P1, undefined, '没帧就不给头像，不硬凑');

  const noCast = baseDoc();
  noCast.cast = [];
  ok(!renderHtml(noCast, CTX()).includes('id="cast-grid"'), '没有 cast 就不出人物那一节');
  ok(renderHtml(baseDoc(), CTX()).includes('id="cast-grid"'), '有 cast 就出');
}

{
  // 提示里点名的镜号做成按钮，点一下能跳过去
  const doc = baseDoc();
  doc.shots[2].camera = 'static';
  const busy = baseTrack();
  for (let i = 37; i <= 48; i += 1) busy.values[i] = 30;
  const html = renderHtml(doc, { track: busy });
  ok(html.includes('data-shot="S03"'), '提示里的镜号是可点的按钮');
  ok(html.includes('class="hint"'), '提示单独成块');

  const broken = baseDoc();
  broken.shots[1].size = 'closeup';
  const bad = renderHtml(broken, CTX());
  ok(bad.includes('quality-banner bad'), '有门没过时横幅变红');
  ok(bad.includes('<li class="bad"'), '没过的门单独标红');
}

{
  // 播放器指哪条片子
  ok(renderHtml(baseDoc(), { ...CTX(), video: '../demo.mp4' }).includes('src="../demo.mp4"'), '--video 指定原片路径');
  ok(renderHtml(baseDoc(), CTX()).includes('src="t.mp4"'), '不给就用 JSON 里的 source');
  const posterless = renderHtml(baseDoc(), CTX());
  ok(!posterless.includes('poster='), '没抽帧就不设封面');
  ok(renderHtml(baseDoc(), { ...CTX(), frameExists: { S01a: true } }).includes('poster="frames/S01a.jpg"'), '有首帧就当封面');
}

/* ------------------------------------------------------------------ */
/* 15. 节奏分析：可选字段，但标了就得标全、标了就得说清为什么             */
/* ------------------------------------------------------------------ */
{
  const beat = (patch = () => {}) => {
    const doc = baseDoc();
    doc.shots.forEach((s, i) => {
      s.rhythm = ['hook', 'build', 'payoff'][i];
      s.rhythmNote = ['开篇就把冻肉怼到脸上，反常画面先钉住人',
        '手部特写把压力再往上推一层', '老太太终于说出那句话，前面的憋屈在这儿放掉'][i];
    });
    patch(doc);
    return doc;
  };

  holds(beat(), 'rhythm', '整片标满、每条都有理由');
  holds(baseDoc(), 'rhythm', '一镜都不标是允许的——这个字段可选');

  fails(beat((d) => { delete d.shots[1].rhythm; delete d.shots[1].rhythmNote; }), 'rhythm',
    '只标了一半——半张表汇总不出东西');
  fails(beat((d) => { d.shots[0].rhythm = 'climax'; }), 'rhythm', '节奏角色不在词表里');
  fails(beat((d) => { d.shots[0].rhythmNote = ''; }), 'rhythm', '标了角色却没写为什么');
  fails(beat((d) => { d.shots[0].rhythmNote = '很带感'; }), 'rhythm', '理由太短');
  fails(beat((d) => { d.shots[0].rhythmNote = '开篇氛围感很强，一下子就抓住人了'; }), 'rhythm',
    '理由里是空话，不是观众看到了什么');
  fails(beat((d) => { d.shots[0].rhythmNote = 'Visually stunning opening that hooks you'; }), 'rhythm',
    '英文理由里的空话照样拦');
  fails(beat((d) => { d.shots[0].rhythmNote = 'Opens cold'; }), 'rhythm', '英文理由太短（按词数）');

  ok(Object.keys(RHYTHM_ROLES).length === 8, '节奏词表八个角色');
  ok(RHYTHM_ROLES.hook && RHYTHM_ROLES.payoff && RHYTHM_ROLES.turn, '钩子/兑现/转折都在');
  for (const [k, v] of Object.entries(RHYTHM_ROLES)) ok(v.color && v.zh && v.en, `${k} 有中英名与颜色`);

  // 三条提示都不拦，但要指出短视频最常掉人的地方
  const hintsOf = (doc, ctx = {}) => validate(doc, ctx).hints.join(' / ');
  ok(hintsOf(beat((d) => { d.shots[0].rhythm = 'setup'; d.shots[0].rhythmNote = '先把地点和人交代清楚'; }))
    .includes('钩子'), '开篇没钩子会提示');
  ok(validate(beat((d) => { d.shots[0].rhythm = 'setup'; d.shots[0].rhythmNote = '先把地点和人交代清楚'; }))
    .gates.find((g) => g.id === 'rhythm').ok, '但它只是提示，不拦');
  ok(hintsOf(beat((d) => {
    d.shots[0].rhythm = 'payoff'; d.shots[0].rhythmNote = '一上来就给结果，前面什么都没埋';
  })).includes('兑现'), '兑现前面没有铺垫会提示');

  const flat = baseDoc();
  flat.meta.durationSeconds = 60;
  flat.seedCuts = [];
  flat.shots = Array.from({ length: 8 }, (_, i) => ({
    id: shotNo(i), start: i * 7.5, end: (i + 1) * 7.5, seconds: 7.5,
    size: 'medium', category: 'subject', camera: 'static', subjects: [],
    frame: `第 ${i + 1} 镜：老太太在院子里来回走动，动作没有变化`,
    onscreenText: '', audio: '', motion: 0.5,
    rhythm: 'setup', rhythmNote: '继续交代环境，没有新信息进来',
  }));
  flat.shots[0].rhythm = 'hook';
  flat.shots[0].rhythmNote = '开篇一记摔门声把人钉住，画面先给的是背影';
  ok(validate(flat, {}).hints.some((h) => h.includes('节奏在这一段是平的')), '连着七镜同一个角色会提示节奏平');
}

/* ------------------------------------------------------------------ */
/* 中英文案：门的名字、违规信息、命令行输出都跟着 --lang 走               */
/* ------------------------------------------------------------------ */
{
  const zh = validate(baseDoc(), CTX());
  const en = validate(baseDoc(), { ...CTX(), lang: 'en' });
  eq(zh.gates[0].label, '时间轴连续', '中文门名');
  eq(en.gates[0].label, 'Timeline is continuous', '英文门名');
  for (const g of en.gates) ok(!/[一-鿿]/.test(g.label), `英文门名里不该有中文：${g.label}`);
  for (const g of en.gates) ok(!g.skipped || !/[一-鿿]/.test(g.skipped), `英文跳过原因里不该有中文：${g.skipped}`);
  eq(gateLabel('motion', 'en'), 'Camera vs. measured motion', 'gateLabel 直接取名字');
  eq(gateLabel('motion', undefined), '运镜实测对账', '不给语言就中文');

  // 违规信息本身也得跟着切
  const broken = baseDoc();
  broken.shots[1].size = 'closeup';
  broken.shots[1].seconds = 99;
  const issues = validate(broken, { ...CTX(), lang: 'en' }).failed.flatMap((g) => g.issues);
  ok(issues.length >= 2, '英文模式下照样报出违规');
  for (const i of issues) ok(!/[一-鿿]/.test(i), `英文违规信息里不该有中文：${i}`);

  // 每一条文案，中英两套都得在，且不能是复制粘贴
  ok(MESSAGE_KEYS.length > 30, `文案键至少 30 条（实际 ${MESSAGE_KEYS.length}）`);
  const args = ['S01', 2, 3, 4, 5, 6];
  for (const key of MESSAGE_KEYS) {
    const z = msgs('zh')(key, ...args);
    const e = msgs('en')(key, ...args);
    ok(typeof z === 'string' && z.length > 0, `zh 文案缺失：${key}`);
    ok(typeof e === 'string' && e.length > 0, `en 文案缺失：${key}`);
    ok(z !== e, `${key} 的中英文案一模一样，八成是漏翻了`);
    ok(!/[一-鿿]/.test(e), `en 文案里混了中文：${key} → ${e}`);
  }
}

/* ------------------------------------------------------------------ */
/* 英文片：画面描述的判据跟着**描述本身的语言**走，不跟着界面语言走        */
/* ------------------------------------------------------------------ */
{
  const en = (frame) => {
    const doc = baseDoc();
    doc.cast = [];
    doc.shots = [{
      id: 'S01', start: 0, end: 10, seconds: 10, size: 'wide', category: 'subject', camera: 'static',
      transitionIn: 'cut', subjects: [], frame, onscreenText: '', audio: '', motion: 0.5,
    }];
    doc.seedCuts = [];
    return validate(doc, { lang: 'en' }).gates.find((g) => g.id === 'frame-text');
  };

  ok(en('A snow-covered meat rack fills the yard, carcasses swaying on iron hooks').ok,
    '像样的英文描述放行');
  ok(!en('An old woman talks').ok, '五个单词的英文描述太短——中文的 12 字判据在这里等于没门');
  eq(DEFAULT_PARAMS.minFrameWords, 8, '英文按词数，不按字符数');
  for (const word of ['visually stunning', 'atmospheric', 'breathtaking']) {
    ok(!en(`The yard is ${word} in the cold morning light, hooks and carcasses everywhere`).ok,
      `英文空话「${word}」照拦`);
  }
  ok(!en('This shot shows an old woman crossing the frozen yard toward the counter').ok,
    '英文废话开头「This shot…」照拦');
  ok(!en('We see an old woman crossing the frozen yard toward the wooden counter').ok,
    '「We see…」也是废话开头');
  ok(VAGUE_WORDS_EN.length >= 10, '英文空话词表至少 10 个');

  // 一条违规只报一次：stunning 被 visually stunning 包住
  const hits = en('The yard looks visually stunning under the cold grey sky of winter').issues
    .filter((i) => i.includes('leans on'));
  eq(hits.length, 1, '短词被长词包住时不重复点名');

  // 中文描述在英文界面下仍按中文判据
  const zhText = baseDoc();
  zhText.shots[0].frame = '雪地木架上挂着冻肉，氛围感很强';
  const g = validate(zhText, { lang: 'en' }).gates.find((x) => x.id === 'frame-text');
  ok(!g.ok && g.issues.some((i) => i.includes('氛围感')), '中文描述照中文词表查，只是信息用英文说');
}

/* ------------------------------------------------------------------ */
/* 自带样例：既是质量基准，也是夹具                                      */
/* ------------------------------------------------------------------ */
{
  const url = new URL('../examples/demo-shots.json', import.meta.url);
  const trackUrl = new URL('../examples/demo-track.json', import.meta.url);
  const doc = JSON.parse(readFileSync(url, 'utf8'));
  const track = JSON.parse(readFileSync(trackUrl, 'utf8'));
  const v = validate(doc, { track });
  ok(v.ok, `样例必须全绿（红的是：${v.failed.map((g) => g.id).join('、')}）`);
  eq(doc.shots.length, 53, '样例 53 镜');
  eq(stats(doc).totalSeconds, doc.meta.durationSeconds, '样例镜头时长之和 = 片长');
  ok(v.hints.length >= 1, '样例里留着一条运动提示当范例');
  ok(doc.shots.every((s) => s.frame.length >= 12), '样例每镜都有像样的画面描述');
}

/* ------------------------------------------------------------------ */

if (failures.length) {
  process.stderr.write(`\n${failures.length} 项没过：\n`);
  for (const f of failures) process.stderr.write(`  ✗ ${f}\n`);
  process.stderr.write(`\n通过 ${passed}／${passed + failures.length}\n`);
  process.exitCode = 1;
} else {
  process.stdout.write(`✅ ${passed} 项断言全部通过（15 道门每道都有击穿用例）\n`);
}
