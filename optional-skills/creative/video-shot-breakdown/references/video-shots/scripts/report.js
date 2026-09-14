/* video-shots 拉片报告的交互层。
 *
 * 数据只有两个入口，都由 render 内联在本文件之前：
 *   DOC — shots.json 原样（报告里能看到的每个数字都从它算，不预先写死）
 *   CFG — 词表、色阶、筛选项、人物头像取哪一镜、关键帧有没有、界面文案
 *
 * 所以这份文件是**通用的**：换一条片子、换一种语言，它不用改。
 */

const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const W = CFG.words;
const { sizes, cats, cams, rhythms } = CFG.labels;
const colors = CFG.colors;
const icon = (n) => `<svg class="icon" aria-hidden="true"><use href="#i-${n}"/></svg>`;
const fmt = (t) => {
  const v = Math.max(0, Math.round((Number(t) || 0) * 100));
  return `${String(Math.floor(v / 6000)).padStart(2, '0')}:${String(Math.floor(v / 100) % 60).padStart(2, '0')}.${String(v % 100).padStart(2, '0')}`;
};
const castName = (id) => DOC.cast?.find((c) => c.id === id)?.name || id;
/** 缺图就摆占位，不摆一个会 404 的 <img>——报告不装有。 */
const fimg = (id, suffix, cls = '') => (CFG.frames[id] ?? '').includes(suffix)
  ? `<img${cls ? ` class="${cls}"` : ''} src="${CFG.frameDir}/${id}${suffix}.jpg" alt="${id} ${suffix === 'a' ? W.frameA : W.frameB}" loading="lazy">`
  : `<div class="frame-missing${cls ? ` ${cls}` : ''}">${id}${suffix}<br>${W.missing}</div>`;

const SHOTS = DOC.shots ?? [];
const N = SHOTS.length;
const total = Number(DOC.meta?.durationSeconds) || SHOTS.reduce((a, s) => a + s.seconds, 0);
const durations = SHOTS.map((s) => s.seconds).sort((a, b) => a - b);
const median = N ? (N % 2 ? durations[(N - 1) / 2] : (durations[N / 2 - 1] + durations[N / 2]) / 2) : 0;
const longest = SHOTS.reduce((a, s) => (!a || s.seconds > a.seconds ? s : a), null);
const shortest = SHOTS.reduce((a, s) => (!a || s.seconds < a.seconds ? s : a), null);

let playback = null;
let selected = SHOTS[0];
let galleryShot = SHOTS[0];
let frame = 'a';
let category = 'all';
let castFilter = '';
let layout = 'list';
let toastTimer;

/* 统计条 ---------------------------------------------------------------- */
$('stats').innerHTML = [
  [W.shots, N, W.unitShot],
  [W.total, fmt(total), ''],
  [W.avg, N ? (total / N).toFixed(2) : '0', 's'],
  [W.median, median.toFixed(2), 's'],
  [W.range, `${durations[0] ?? 0} / ${durations.at(-1) ?? 0}`, 's'],
  [W.rate, total ? (N / total * 60).toFixed(1) : '0', W.unitCut],
].map((v, i) => `<div class="stat ${i === 4 ? 'range' : ''}"><span class="stat-label">${esc(v[0])}</span><b>${esc(v[1])}<small>${esc(v[2])}</small></b></div>`).join('');

/* 时间轴 ---------------------------------------------------------------- */
$('timeline').innerHTML = SHOTS.map((s) => `<button class="segment" style="width:${s.seconds / total * 100}%;--color:${colors[s.size] || '#c4cfaa'}" data-player-segment="${s.id}" title="${esc(`${s.id} · ${fmt(s.start)} · ${s.seconds}s · ${sizes[s.size] || s.size}`)}" aria-label="${esc(`${W.seek} ${s.id}，${sizes[s.size] || s.size}，${s.seconds}${W.unitSecond}`)}"></button>`).join('');
$('legend').innerHTML = Object.entries(sizes).filter(([k]) => SHOTS.some((s) => s.size === k))
  .map(([k, v]) => `<span><i style="--color:${colors[k] || '#c4cfaa'}"></i>${esc(v)}</span>`).join('');

/* 筛选 ------------------------------------------------------------------ */
$('filters').innerHTML = CFG.filters.map(([k, v]) => `<button data-category="${k}" class="${k === 'all' ? 'active' : ''}" aria-pressed="${k === 'all'}">${esc(v)}</button>`).join('');
const named = CFG.filters.map(([k]) => k).filter((k) => k !== 'all' && k !== 'other');

function renderCards() {
  const query = $('search').value.trim().toLowerCase();
  let shots = SHOTS.filter((s) => (category === 'all' || s.category === category || (category === 'other' && !named.includes(s.category)))
    && (!castFilter || (s.subjects ?? []).includes(castFilter))
    && [s.id, s.frame, s.audio, s.onscreenText, sizes[s.size], cats[s.category], cams[s.camera], ...(s.subjects ?? []).map(castName)]
      .join(' ').toLowerCase().includes(query));
  if ($('sort').value !== 'timeline') shots = [...shots].sort((a, b) => ($('sort').value === 'longest' ? b.seconds - a.seconds : a.seconds - b.seconds));

  $('cards').innerHTML = shots.map((s) => {
    const who = (s.subjects ?? []).map((id) => esc(castName(id))).join(' / ') || '—';
    const beat = s.rhythm
      ? `<span class="beat"><i style="background:${CFG.rhythmColors[s.rhythm] || '#c4cfaa'}"></i>`
        + `${esc(rhythms[s.rhythm] || s.rhythm)}</span>${s.rhythmNote ? ` <span class="beat-note">${esc(s.rhythmNote)}</span>` : ''}`
      : '';
    const say = [
      s.audio ? `<div class="audio-text"><span>${W.audioMark} · </span>${esc(s.audio)}</div>` : '',
      s.onscreenText ? `<div class="audio-text"><span>${W.textMark} · </span>${esc(s.onscreenText)}</div>` : '',
    ].join('') || '—';
    return `<button class="shot-card" id="card-${s.id}" data-player-shot="${s.id}" aria-label="${esc(`${s.id} ${s.frame}`)}">`
      + `<div class="card-image">${fimg(s.id, 'a')}${fimg(s.id, 'b', 'frame-b')}`
      + `<span class="number">${s.id}</span><span class="seconds">${s.seconds.toFixed(2)}s</span></div>`
      + `<div class="card-body"><div class="card-meta"><span class="mono">${fmt(s.start)}</span>`
      + `<span class="tag size">${esc(sizes[s.size] || s.size)}</span></div>`
      + `<p class="card-desc">${esc(s.frame)}</p>`
      + `<div class="card-bottom"><span>${esc(cats[s.category] || s.category)}<span style="margin:0 5px">·</span>${esc(cams[s.camera] || s.camera)}</span><span>${who}</span></div>`
      + (beat ? `<div class="card-beat">${beat}</div>` : '') + '</div>'
      + `<div class="row-content"><div class="row-time">${fmt(s.start)}<br>${fmt(s.end)}<b>${s.seconds.toFixed(2)} s</b></div>`
      + `<div class="row-tags"><span class="tag size">${esc(sizes[s.size] || s.size)}</span><span class="tag">${esc(cats[s.category] || s.category)}</span>`
      + `<span class="tag cam">${esc(cams[s.camera] || s.camera)}</span><span class="row-subject">${who}</span></div>`
      + `<div class="row-desc">${esc(s.frame)}${beat ? `<div class="row-beat">${beat}</div>` : ''}</div>`
      + `<div class="row-audio">${say}</div></div></button>`;
  }).join('');

  playback?.refresh();
  $('empty').hidden = shots.length > 0;
  $('result-count').textContent = `${castFilter ? `${castName(castFilter)} · ` : ''}${W.showing} ${shots.length} / ${N} ${W.unitShot}`;
}

/* 关键帧大图 ------------------------------------------------------------- */
function setFrame(value) {
  frame = value;
  const has = (CFG.frames[galleryShot.id] ?? '').includes(value);
  $('lightbox-img').src = has ? `${CFG.frameDir}/${galleryShot.id}${value}.jpg` : '';
  $('lightbox-img').alt = `${galleryShot.id} ${value === 'a' ? W.frameA : W.frameB}`;
  $('lightbox-title').textContent = `${galleryShot.id} / FRAME ${value.toUpperCase()} · ${sizes[galleryShot.size] || galleryShot.size}`;
}

function selectShot(id) {
  if (!SHOTS.some((s) => s.id === id)) return;
  playback?.seek(id);
  history.replaceState(null, '', `#${id}`);
}

function showPage(page) {
  const section = $(page);
  if (!section) return;
  if (section.tagName === 'DETAILS') section.open = true;
  section.scrollIntoView({ block: 'start', behavior: 'instant' });
}
document.querySelectorAll('[data-page]').forEach((b) => { b.onclick = () => showPage(b.dataset.page); });

document.addEventListener('click', (e) => {
  const image = e.target.closest('.shot-card img');
  if (image) {
    const id = image.closest('[data-player-shot]').dataset.playerShot;
    galleryShot = SHOTS.find((s) => s.id === id);
    setFrame(image.classList.contains('frame-b') ? 'b' : 'a');
    $('lightbox').showModal();
    return;
  }
  const filter = e.target.closest('[data-category]');
  if (filter) {
    category = filter.dataset.category;
    castFilter = '';
    document.querySelectorAll('[data-category]').forEach((b) => {
      b.classList.toggle('active', b === filter);
      b.setAttribute('aria-pressed', b === filter);
    });
    renderCards();
  }
  const f = e.target.closest('[data-frame]');
  if (f) setFrame(f.dataset.frame);
  // 质量门的提示里指名的镜头，点一下跳过去
  const jump = e.target.closest('[data-shot]');
  if (jump) { showPage('overview'); selectShot(jump.dataset.shot); }
});

$('search').oninput = renderCards;
$('sort').onchange = renderCards;

function resetFilters() {
  category = 'all';
  castFilter = '';
  $('search').value = '';
  $('sort').value = 'timeline';
  document.querySelectorAll('[data-category]').forEach((b) => {
    b.classList.toggle('active', b.dataset.category === 'all');
    b.setAttribute('aria-pressed', b.dataset.category === 'all');
  });
  renderCards();
}
$('reset').onclick = resetFilters;

['grid', 'list'].forEach((v) => {
  $(`${v}-view`).onclick = () => {
    layout = v;
    $('list-head').hidden = v !== 'list';
    $('cards').classList.toggle('list', v === 'list');
    ['grid', 'list'].forEach((k) => {
      $(`${k}-view`).classList.toggle('active', k === v);
      $(`${k}-view`).setAttribute('aria-pressed', k === v);
    });
  };
});

function notify(message) {
  clearTimeout(toastTimer);
  $('toast').textContent = message;
  $('toast').hidden = false;
  toastTimer = setTimeout(() => { $('toast').hidden = true; }, 3200);
}

$('export').onclick = () => {
  const url = URL.createObjectURL(new Blob([JSON.stringify(DOC, null, 2)], { type: 'application/json' }));
  const a = document.createElement('a');
  a.href = url;
  a.download = CFG.exportName;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 10000);
  notify(W.exported);
};

$('close-lightbox').onclick = () => $('lightbox').close();
$('lightbox').addEventListener('click', (e) => { if (e.target === $('lightbox')) $('lightbox').close(); });

document.addEventListener('keydown', (e) => {
  if (['INPUT', 'SELECT', 'TEXTAREA', 'VIDEO'].includes(e.target.tagName) || e.target.closest('.report-player')) return;
  if ($('lightbox').open) {
    if (e.key === 'ArrowLeft' || e.key === 'ArrowRight') { e.preventDefault(); setFrame(frame === 'a' ? 'b' : 'a'); }
    return;
  }
  if (e.key === 'ArrowLeft' || e.key === 'ArrowRight') {
    e.preventDefault();
    selectShot(SHOTS[SHOTS.indexOf(selected) + (e.key === 'ArrowLeft' ? -1 : 1)]?.id);
  }
});

/* 分布 ------------------------------------------------------------------ */
$('distributions').innerHTML = [
  ['size', W.sizeTitle, W.sizeSub, sizes],
  ['category', W.catTitle, W.catSub, cats],
  ['camera', W.camTitle, W.camSub, cams],
  ...(CFG.hasRhythm ? [['rhythm', W.rhythmTitle, W.rhythmSub, rhythms]] : []),
].map(([field, title, subtitle, labels]) => {
  const values = Object.entries(labels).map(([k, label]) => {
    const items = SHOTS.filter((s) => s[field] === k);
    return { label, count: items.length, seconds: items.reduce((a, s) => a + s.seconds, 0) };
  }).filter((v) => v.count).sort((a, b) => b.seconds - a.seconds);
  return `<article class="analysis-card"><h3>${esc(title)}</h3><p>${esc(subtitle)}</p>`
    + values.map((v) => `<div class="dist-row"><div class="dist-label"><span>${esc(v.label)}</span>`
      + `<small>${v.count} ${W.unitShot} · ${(v.seconds / total * 100).toFixed(1)}%</small></div>`
      + `<div class="bar-bg"><i style="width:${v.seconds / total * 100}%"></i></div></div>`).join('')
    + '</article>';
}).join('');

/** 概述句同样是算出来的，不是写死的。 */
const share = (field, key) => SHOTS.filter((s) => s[field] === key).reduce((a, s) => a + s.seconds, 0) / total * 100;
// 注意：不能叫 top——window.top 是不可配置的全局，顶层 const top 会直接 SyntaxError
const pickTop = (field) => [...new Set(SHOTS.map((s) => s[field]))].sort((a, b) => share(field, b) - share(field, a))[0];
const topSize = pickTop('size');
const topCat = pickTop('category');
$('analysis-note').innerHTML = W.note
  .replace('{size}', `<b>${esc(sizes[topSize] || topSize)}</b>`)
  .replace('{sizePct}', `<b>${share('size', topSize).toFixed(1)}%</b>`)
  .replace('{cat}', `<b>${esc(cats[topCat] || topCat)}</b>`)
  .replace('{catPct}', `<b>${share('category', topCat).toFixed(1)}%</b>`)
  .replace('{longest}', `<b>${longest.id} · ${longest.seconds} ${W.unitSecond}</b>`)
  .replace('{shortest}', `<b>${shortest.id} · ${shortest.seconds} ${W.unitSecond}</b>`);

/* 人物 ------------------------------------------------------------------ */
if ($('cast-grid')) {
  $('cast-grid').innerHTML = (DOC.cast ?? []).map((c) => {
    const shot = CFG.portraits[c.id];
    const n = SHOTS.filter((s) => (s.subjects ?? []).includes(c.id)).length;
    return `<article class="cast-card">${shot ? fimg(shot, 'a') : `<div class="frame-missing" style="aspect-ratio:7/3">${W.missing}</div>`}`
      + `<div><h3>${esc(c.name)} <span class="mono muted" style="font-size:10px;margin-left:7px">${esc(c.id)}</span></h3>`
      + `<p>${esc(c.note ?? '')}</p>`
      + `<button data-cast="${esc(c.id)}">${esc(n === 1 ? W.castShotsOne : W.castShots.replace('{n}', n))} ${icon('right')}</button></div></article>`;
  }).join('');
  document.querySelectorAll('[data-cast]').forEach((b) => {
    b.onclick = () => {
      resetFilters();
      castFilter = b.dataset.cast;
      renderCards();
      showPage('overview');
      $('library').scrollIntoView({ block: 'start', behavior: 'instant' });
    };
  });
}

renderCards();

/* 播放器：时间轴、镜头行、当前镜头信息三处跟着视频走 ------------------------ */
function shotAtTime(shots, time) {
  if (!Number.isFinite(time) || !shots.length) return null;
  let lo = 0;
  let hi = shots.length - 1;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (Number(shots[mid].start) <= time) lo = mid + 1;
    else hi = mid - 1;
  }
  if (hi < 0) return null;
  const shot = shots[hi];
  return time < Number(shot.end) || (hi === shots.length - 1 && time === Number(shot.end)) ? shot : null;
}

function createReportPlayer(shots, options = {}) {
  const t = options.words;
  const video = document.getElementById('report-video');
  const el = (id) => document.getElementById(id);
  const duration = Number(options.duration) || Number(shots.at(-1)?.end) || 0;
  let activeId;
  let activeShot = null;
  let pendingTime = null;
  let localUrl = null;
  let frameHandle = null;
  let rowNodes = [];
  let segmentNodes = [];
  let activeSegments = [];

  function paint(time) {
    const shot = shotAtTime(shots, time);
    const changed = shot?.id !== activeId;
    activeId = shot?.id;
    activeShot = shot;
    if (changed) {
      el('player-shot').textContent = shot?.id || '—';
      el('player-range').textContent = shot ? `${fmt(shot.start)} → ${fmt(shot.end)}` : t.noShot;
      el('player-duration').textContent = shot ? `${Number(shot.seconds ?? (shot.end - shot.start)).toFixed(2)} s` : '—';
      el('player-description').textContent = shot?.frame || '';
      el('player-beat').textContent = shot?.rhythm
        ? `${CFG.labels.rhythms[shot.rhythm] || shot.rhythm}｜${shot.rhythmNote || ''}` : '';
      el('player-dialogue').textContent = [shot?.audio ? `${t.audio}: ${shot.audio}` : '', shot?.onscreenText ? `${t.text}: ${shot.onscreenText}` : ''].filter(Boolean).join(' / ');
      options.onShotChange?.(shot);
    }
    // 播放过程中只改 class 和进度值，绝不重建行——重建会把滚动位置和焦点抖掉。
    for (const row of rowNodes) {
      const current = !!shot && row.dataset.playerShot === shot.id;
      row.classList.toggle('is-current', current);
      if (current) row.setAttribute('aria-current', 'true'); else row.removeAttribute('aria-current');
    }
    activeSegments = [];
    for (const segment of segmentNodes) {
      const current = !!shot && segment.dataset.playerSegment === shot.id;
      segment.classList.toggle('is-current', current);
      if (current) { segment.setAttribute('aria-current', 'true'); activeSegments.push(segment); }
      else { segment.removeAttribute('aria-current'); segment.style.removeProperty('--played'); }
    }
    const fraction = shot ? Math.max(0, Math.min(1, (time - shot.start) / (shot.end - shot.start))) : 0;
    for (const segment of activeSegments) segment.style.setProperty('--played', `${fraction * 100}%`);
    el('player-progress').value = fraction;
    el('player-clock').textContent = `${fmt(time)} / ${fmt(duration)}`;
  }

  function refresh() {
    rowNodes = [...document.querySelectorAll('[data-player-shot]')];
    segmentNodes = [...document.querySelectorAll('[data-player-segment]')];
    paint(pendingTime ?? video.currentTime);
  }
  function update() { paint(pendingTime ?? video.currentTime); }
  function stopFrames() {
    if (frameHandle == null) return;
    if (video.cancelVideoFrameCallback) video.cancelVideoFrameCallback(frameHandle);
    else cancelAnimationFrame(frameHandle);
    frameHandle = null;
  }
  function scheduleFrame() {
    stopFrames();
    if (video.paused || video.ended) return;
    if (video.requestVideoFrameCallback) {
      frameHandle = video.requestVideoFrameCallback((now, metadata) => {
        frameHandle = null;
        paint(pendingTime ?? metadata.mediaTime);
        scheduleFrame();
      });
    } else frameHandle = requestAnimationFrame(() => { frameHandle = null; update(); scheduleFrame(); });
  }
  function applyPending() {
    if (pendingTime == null || video.readyState < 1 || !video.seekable.length) return;
    video.currentTime = pendingTime;
    pendingTime = null;
  }
  function seek(id) {
    const shot = shots.find((s) => s.id === id);
    if (!shot) return;
    pendingTime = Number(shot.start);
    history.replaceState(null, '', `#${encodeURIComponent(id)}`);
    paint(pendingTime);
    applyPending();
  }
  function status() {
    el('player-state').textContent = video.ended ? t.ended : video.paused ? t.paused : t.playing;
  }
  function checkDuration() {
    const mismatch = Number.isFinite(video.duration) && duration > 0 && Math.abs(video.duration - duration) > 0.5;
    el('player-error').hidden = !mismatch;
    el('player-error').textContent = mismatch ? t.mismatch : '';
  }
  function onClick(event) {
    if (event.target.closest('img, [data-no-seek]')) return;
    const target = event.target.closest('[data-player-shot], [data-player-segment]');
    if (!target) return;
    event.preventDefault();
    seek(target.dataset.playerShot || target.dataset.playerSegment);
  }
  function onKey(event) {
    if (!['Enter', ' '].includes(event.key) || !event.target.matches('[data-player-shot]')) return;
    if (event.target.tagName === 'BUTTON') return; // 按钮自己会合成 click
    event.preventDefault();
    seek(event.target.dataset.playerShot);
  }
  document.addEventListener('click', onClick);
  document.addEventListener('keydown', onKey);
  video.addEventListener('timeupdate', update);
  video.addEventListener('seeking', () => { stopFrames(); update(); });
  video.addEventListener('seeked', () => { update(); scheduleFrame(); });
  video.addEventListener('play', () => { applyPending(); status(); scheduleFrame(); });
  video.addEventListener('pause', () => { stopFrames(); update(); status(); });
  video.addEventListener('ended', () => { stopFrames(); update(); status(); });
  for (const name of ['loadedmetadata', 'loadeddata', 'canplay', 'progress']) video.addEventListener(name, applyPending);
  video.addEventListener('loadedmetadata', checkDuration);
  video.addEventListener('error', () => {
    stopFrames();
    el('player-error').textContent = t.videoMissing;
    el('player-error').hidden = false;
    el('player-state').textContent = t.playError;
  });
  el('player-file').addEventListener('change', (event) => {
    const file = event.target.files?.[0];
    if (!file) return;
    const resumeAt = pendingTime ?? video.currentTime;
    video.pause();
    if (localUrl) URL.revokeObjectURL(localUrl);
    localUrl = URL.createObjectURL(file);
    pendingTime = resumeAt;
    el('player-error').hidden = true;
    el('player-state').textContent = t.ready;
    video.src = localUrl;
    video.load();
  });
  window.addEventListener('pagehide', stopFrames);
  window.addEventListener('pageshow', scheduleFrame);
  if (!video.getAttribute('src')) { el('player-error').textContent = t.videoMissing; el('player-error').hidden = false; }
  refresh();
  return { seek, refresh, currentShot: () => activeShot };
}

playback = createReportPlayer(SHOTS, {
  words: W,
  duration: total,
  onShotChange: (shot) => { if (shot) selected = shot; },
});

const initial = location.hash.slice(1);
if (SHOTS.some((s) => s.id === initial)) selectShot(initial);
window.addEventListener('hashchange', () => {
  const id = location.hash.slice(1);
  if (SHOTS.some((s) => s.id === id)) { showPage('overview'); selectShot(id); }
});
