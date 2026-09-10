import {
  atom, Button, createBudgetedLoop, GlyphSpinner, icons, Popover, PopoverContent, PopoverTrigger,
  RowButton, SearchField, STATUSBAR_AREAS, Tip, usePluginI18n,
  useQuery, useValue
} from '@hermes/plugin-sdk'
import { useEffect, useRef, useState } from 'react'
import { jsx, jsxs } from 'react/jsx-runtime'

const ID = 'radio'
const NIGHTRIDE = 'https://stream.nightride.fm/'
const PRESETS = [
  { id: 'chillsynth', name: 'Chillsynth', description: 'Soft focus · warm synths', provider: 'Nightride FM', url: `${NIGHTRIDE}chillsynth.mp3`, homepage: 'https://nightride.fm/?station=chillsynth' },
  { id: 'nightride', name: 'Nightride', description: 'Synthwave · after hours', provider: 'Nightride FM', url: `${NIGHTRIDE}nightride.mp3`, homepage: 'https://nightride.fm/' },
  { id: 'darksynth', name: 'Darksynth', description: 'Dark electronics · high energy', provider: 'Nightride FM', url: `${NIGHTRIDE}darksynth.mp3`, homepage: 'https://nightride.fm/?station=darksynth' },
  { id: 'spacesynth', name: 'Spacesynth', description: 'Cosmic synths · retro futures', provider: 'Nightride FM', url: `${NIGHTRIDE}spacesynth.mp3`, homepage: 'https://nightride.fm/?station=spacesynth' },
  { id: 'paradise-main', name: 'Main Mix', description: 'Eclectic · human selected', provider: 'Radio Paradise', url: 'https://stream.radioparadise.com/aac-128', homepage: 'https://radioparadise.com/' },
  { id: 'paradise-mellow', name: 'Mellow Mix', description: 'A gentler pace', provider: 'Radio Paradise', url: 'https://stream.radioparadise.com/mellow-flac', homepage: 'https://radioparadise.com/' },
  { id: 'eve-radio', name: 'EVE Radio', description: 'GamingNow · EVE community radio', provider: 'GamingNow', url: 'https://media01.gamingnow.net:8010/erweb.mp3', homepage: 'https://gamingnow.net/eve-radio/' }
]

const EN = {
  radio: 'Radio', browse: 'Choose a station', play: 'Play radio', pause: 'Pause radio', next: 'Next station',
  live: 'Live', paused: 'Paused', connecting: 'Connecting', error: 'Stream unavailable', retry: 'Try again',
  search: 'Find a station…', volume: 'Volume', audioOnly: 'Audio playing · visualizer unavailable',
  mute: 'Mute radio', unmute: 'Unmute radio', save: 'Pin station', unsave: 'Unpin station',
  visit: 'Visit station website', artistProfile: 'Visit artist profile',
  noResults: 'No stations found', searchHint: 'Search by station name. Try jazz, ambient, or house.',
  searchError: 'Station search is unavailable. Your presets still work.', searching: 'Searching stations',
  directory: 'Search powered by Radio Browser', note: 'Live radio · next switches stations',
  streamError: 'This stream could not connect. Try again or choose another station.',
  close: 'Close radio', nowPlaying: 'Now playing', elsewhere: 'Paused — playing in another window',
  'chillsynthDescription': 'Soft focus · warm synths', 'nightrideDescription': 'Synthwave · after hours',
  'darksynthDescription': 'Dark electronics · high energy', 'spacesynthDescription': 'Cosmic synths · retro futures',
  'paradise-mainDescription': 'Eclectic · human selected', 'paradise-mellowDescription': 'A gentler pace'
}
const LOCALES = {
  en: EN,
  ja: { ...EN, artistProfile: 'アーティストのプロフィールを開く', audioOnly: '再生中 · 波形を表示できません', radio: 'ラジオ', browse: 'ステーションを選択', play: 'ラジオを再生', pause: 'ラジオを一時停止', next: '次のステーション', live: 'ライブ', paused: '一時停止', connecting: '接続中', error: '再生できません', retry: '再試行', search: 'ステーションを検索…', volume: '音量', mute: 'ミュート', unmute: 'ミュート解除', save: 'ステーションを固定', unsave: '固定を解除', visit: '公式サイトを開く', noResults: '見つかりませんでした', searchHint: '名前で検索。jazz、ambient、house など。', searchError: '検索できません。おすすめは再生できます。', searching: '検索中', directory: 'Radio Browser による検索', note: 'ライブ放送 · 次へでステーションを切り替え', streamError: '接続できませんでした。再試行するか別のステーションを選んでください。', close: 'ラジオを閉じる', nowPlaying: '再生中', elsewhere: '別のウィンドウで再生中', 'chillsynthDescription': 'やさしい集中 · 暖かなシンセ', 'nightrideDescription': 'シンセウェーブ · 深夜', 'darksynthDescription': 'ダークな電子音 · 高揚感', 'spacesynthDescription': '宇宙的シンセ · レトロな未来', 'paradise-mainDescription': '多彩な選曲 · 人がキュレーション', 'paradise-mellowDescription': 'ゆったりした時間' },
  zh: { ...EN, artistProfile: '访问艺人主页', audioOnly: '正在播放 · 无法显示波形', radio: '电台', browse: '选择电台', play: '播放电台', pause: '暂停电台', next: '下一个电台', live: '直播', paused: '已暂停', connecting: '正在连接', error: '无法播放', retry: '重试', search: '搜索电台…', volume: '音量', mute: '静音', unmute: '取消静音', save: '置顶电台', unsave: '取消置顶', visit: '访问电台网站', noResults: '没有找到电台', searchHint: '按名称搜索，例如 jazz、ambient 或 house。', searchError: '暂时无法搜索，精选电台仍可使用。', searching: '正在搜索', directory: '搜索由 Radio Browser 提供', note: '直播电台 · 下一首将切换电台', streamError: '无法连接，请重试或选择其他电台。', close: '关闭电台', nowPlaying: '正在播放', elsewhere: '已暂停，正在其他窗口播放', 'chillsynthDescription': '轻松专注 · 温暖合成器', 'nightrideDescription': '合成器浪潮 · 深夜', 'darksynthDescription': '暗黑电子 · 充满能量', 'spacesynthDescription': '宇宙合成器 · 复古未来', 'paradise-mainDescription': '多元风格 · 人工精选', 'paradise-mellowDescription': '放慢节奏' },
  'zh-hant': { ...EN, artistProfile: '造訪藝人主頁', audioOnly: '播放中 · 無法顯示波形', radio: '電台', browse: '選擇電台', play: '播放電台', pause: '暫停電台', next: '下一個電台', live: '直播', paused: '已暫停', connecting: '正在連線', error: '無法播放', retry: '重試', search: '搜尋電台…', volume: '音量', mute: '靜音', unmute: '取消靜音', save: '釘選電台', unsave: '取消釘選', visit: '造訪電台網站', noResults: '找不到電台', searchHint: '按名稱搜尋，例如 jazz、ambient 或 house。', searchError: '暫時無法搜尋，精選電台仍可使用。', searching: '正在搜尋', directory: '搜尋由 Radio Browser 提供', note: '直播電台 · 下一首會切換電台', streamError: '無法連線，請重試或選擇其他電台。', close: '關閉電台', nowPlaying: '正在播放', elsewhere: '已暫停，正在其他視窗播放', 'chillsynthDescription': '輕鬆專注 · 溫暖合成器', 'nightrideDescription': '合成器浪潮 · 深夜', 'darksynthDescription': '暗黑電子 · 充滿能量', 'spacesynthDescription': '宇宙合成器 · 復古未來', 'paradise-mainDescription': '多元風格 · 人工精選', 'paradise-mellowDescription': '放慢節奏' }
}

// Disk plugins are not scanned by Tailwind. Only plugin layout lives here;
// Button, SearchField and Popover own their own chrome.
const CSS = `
.hermes-radio-bar{display:flex;align-items:center;gap:2px;height:100%;color:var(--ui-text-tertiary)}
.hermes-radio-waveform{position:relative;height:24px;width:100%;font-family:var(--font-mono,monospace);font-size:12px;line-height:12px;color:var(--ui-text-tertiary);opacity:.45;overflow:hidden;flex-shrink:0}
.hermes-radio-waveform-layer{position:absolute;inset:0;display:grid;grid-template-columns:repeat(var(--radio-columns),minmax(0,1fr));grid-auto-rows:12px}
.hermes-radio-waveform-layer>span{text-align:center;min-width:0}
.hermes-radio-waveform-layer:nth-child(1){opacity:.12}
.hermes-radio-waveform-layer:nth-child(2){opacity:.28}
.hermes-radio-waveform[data-compact=true]{width:40px;height:12px;margin-inline:2px}
.hermes-radio-next{display:flex;align-items:center;width:12px;height:12px}
.hermes-radio-waveform[data-active=true]{color:var(--ui-accent);opacity:1}
.hermes-radio-signal{height:48px;padding:5px 6px 0;min-width:0}
.hermes-radio-track{display:flex;align-items:center;gap:5px;height:19px;min-width:0;font-size:11px;line-height:16px;color:var(--ui-text-secondary)}
.hermes-radio-track-text{overflow:hidden;white-space:nowrap;text-overflow:ellipsis;min-width:0}
.hermes-radio-track[data-track=true]{color:var(--ui-text-primary)}
.hermes-radio-artist{min-width:0;max-width:55%;flex-shrink:1;font-size:inherit;font-weight:inherit}
.hermes-radio-artist>span{overflow:hidden;text-overflow:ellipsis}
.hermes-radio-track-separator{flex-shrink:0;color:var(--ui-text-quaternary)}
.hermes-radio-bar .radio-name{max-width:112px;min-width:0;flex:1;overflow:hidden;text-overflow:ellipsis;text-align:left}
.hermes-radio-bar .hermes-radio-action{width:16px;height:16px;padding:0;flex-shrink:0}
.hermes-radio-action-icon{display:flex;align-items:center;justify-content:center;width:12px;height:12px;flex-shrink:0;overflow:hidden}
.hermes-radio-website{display:flex;width:24px;height:24px;flex-shrink:0}
.hermes-radio-panel{width:292px;max-width:calc(100vw - 24px)}
.hermes-radio-search{display:flex;align-items:center;height:32px;padding:0 4px 4px}
.hermes-radio-search-field{flex:1;min-width:0}
.hermes-radio-search-field input{flex:1;width:0;min-width:0}
.hermes-radio-search .hermes-radio-search-field{border:0}
.hermes-radio-search-loader{display:flex;align-items:center;justify-content:center;width:16px;flex-shrink:0;font-size:12px;color:var(--ui-text-tertiary)}
.hermes-radio-list{height:168px;max-height:calc(100dvh - 180px);min-height:80px;overflow-y:auto;overscroll-behavior:contain;scrollbar-gutter:stable}
.hermes-radio-row{display:flex;align-items:center;gap:4px;border-radius:4px}
.hermes-radio-row[data-current=true]{background:var(--chrome-action-hover)}
.hermes-radio-row[data-current=true] .hermes-radio-row-name{color:var(--ui-accent)}
.hermes-radio-row:hover{background:var(--chrome-action-hover)}
.hermes-radio-row [aria-pressed=true]{color:var(--ui-accent)}
.hermes-radio-pin{opacity:0;pointer-events:none}
.hermes-radio-row:hover .hermes-radio-pin,.hermes-radio-row:focus-within .hermes-radio-pin{opacity:1;pointer-events:auto}
@media(hover:none){.hermes-radio-pin{opacity:1;pointer-events:auto}}
.hermes-radio-row-main{display:flex;align-items:center;gap:8px;flex:1;min-width:0;text-align:left;height:24px;padding:2px 6px;cursor:pointer;border-radius:4px}
.hermes-radio-row-main:focus-visible{outline:1px solid var(--ui-accent);outline-offset:-1px}
.hermes-radio-row-icon{display:flex;align-items:center;justify-content:center;width:14px;flex-shrink:0;color:var(--ui-text-tertiary);opacity:0}
.hermes-radio-row:hover .hermes-radio-row-icon,.hermes-radio-row:focus-within .hermes-radio-row-icon,.hermes-radio-row[data-current=true] .hermes-radio-row-icon{opacity:1}
.hermes-radio-row[data-current=true] .hermes-radio-row-icon{color:var(--ui-accent)}
.hermes-radio-row-copy{min-width:0;flex:1}
.hermes-radio-row-name{display:block;font-size:12px;line-height:17px;color:var(--ui-text-primary);overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.hermes-radio-volume{display:flex;align-items:center;gap:4px;height:28px;padding:2px 2px 0}
.hermes-radio-volume-space{flex:1;min-width:6px}
.hermes-radio-volume input{width:76px;min-width:0;height:3px;accent-color:var(--ui-accent);cursor:pointer}
.hermes-radio-error{font-size:11px;line-height:17px;color:var(--ui-text-secondary);padding:4px 4px 12px}
`

function httpsUrl(value) {
  try {
    const url = new URL(value)
    return url.protocol === 'https:' && !url.username && !url.password ? url.href : null
  } catch { return null }
}

function validStation(value) {
  return value && typeof value.id === 'string' && typeof value.name === 'string' && httpsUrl(value.url)
}

async function fetchJson(url, signal) {
  const response = await fetch(url, { signal: AbortSignal.any([signal, AbortSignal.timeout(10000)]), credentials: 'omit' })
  if (!response.ok) throw new Error(`HTTP ${response.status}`)
  return response.json()
}

let directoryServers = ['https://de1.api.radio-browser.info']
let discovered = false
async function searchStations(text, signal) {
  if (!discovered) {
    try {
      const servers = await fetchJson(`${directoryServers[0]}/json/servers`, signal)
      const hosts = [...new Set(servers.map(server => server.name).filter(name => /^[a-z0-9-]+\.api\.radio-browser\.info$/.test(name)))]
      if (hosts.length) directoryServers = hosts.map(name => `https://${name}`).sort(() => Math.random() - 0.5)
      discovered = true
    } catch (error) {
      if (signal.aborted) throw error
    }
  }
  const params = new URLSearchParams({ name: text, limit: '60', hidebroken: 'true', order: 'votes', reverse: 'true' })
  for (const server of directoryServers.slice(0, 3)) {
    try {
      const rows = await fetchJson(`${server}/json/stations/search?${params}`, signal)
      const seen = new Set()
      return rows.flatMap(row => {
        const url = httpsUrl(row.url_resolved)
        if (!url || !row.stationuuid || !row.name || row.hls || seen.has(url)) return []
        seen.add(url)
        return [{ id: row.stationuuid, name: row.name.trim(), url, provider: 'Radio Browser', description: row.tags?.split(',').slice(0, 2).join(' · ') || '', homepage: httpsUrl(row.homepage) }]
      })
    } catch (error) {
      if (signal.aborted) throw error
    }
  }
  throw new Error('Directory unavailable')
}

function sameStation(a, b) {
  return a.id === b.id || a.url === b.url
}

function uniqueStations(items) {
  const ids = new Set()
  const urls = new Set()
  return items.filter(item => {
    if (ids.has(item.id) || urls.has(item.url)) return false
    ids.add(item.id)
    urls.add(item.url)
    return true
  })
}

function createPlayer(ctx) {
  const saved = ctx.storage.get('local.favorites', [])
  const favorites = atom(uniqueStations(Array.isArray(saved) ? saved.filter(validStation) : []))
  const previous = ctx.storage.get('local.station', null)
  const station = atom(validStation(previous) ? previous : PRESETS[0])
  const storedVolume = ctx.storage.get('local.volume', 25)
  const volume = atom(Number.isFinite(storedVolume) ? Math.max(0, Math.min(100, storedVolume)) : 25)
  const status = atom('paused')
  const meterMode = atom('levels')
  const plainStreams = new Set()
  const open = atom(false)
  let audio = null
  let audioContext = null
  let source = null
  let analyser = null
  let output = null
  const waveformData = new Float32Array(2048)
  let generation = 0
  let timeout = null
  let disposed = false
  let lastVolume = volume.get() || 25
  const bus = new BroadcastChannel('hermes:radio:playback')
  const windowId = crypto.randomUUID()

  function stop(nextStatus = 'paused') {
    generation++
    clearTimeout(timeout)
    source?.disconnect()
    analyser?.disconnect()
    output?.disconnect()
    output = null
    source = null
    analyser = null
    if (audioContext?.state === 'running') void audioContext.suspend()
    if (audio) {
      audio.pause()
      audio.removeAttribute('src')
      audio.load()
      audio.remove()
      audio = null
    }
    status.set(nextStatus)
  }

  async function play(next = station.get(), analyse = true) {
    if (disposed || !validStation(next)) return
    stop()
    const token = generation
    station.set(next)
    ctx.storage.set('local.station', next)
    status.set('connecting')
    bus.postMessage({ type: 'play', sender: windowId })
    const element = new Audio()
    audio = element
    element.dataset.hermesRadioAudio = 'true'
    element.hidden = true
    element.preload = 'none'
    // Try real analysis for every station, not a host allowlist. If CORS or
    // the audio graph fails, retry once with ordinary media playback.
    const metered = analyse && !plainStreams.has(next.url)
    meterMode.set(metered ? 'levels' : 'activity')
    if (metered) element.crossOrigin = 'anonymous'
    element.volume = metered ? 1 : volume.get() / 100
    document.body.append(element)
    const current = () => !disposed && token === generation
    const fail = () => {
      if (!current()) return
      if (metered) {
        plainStreams.add(next.url)
        void play(next, false)
      } else stop('error')
    }
    element.addEventListener('playing', () => {
      if (current()) { clearTimeout(timeout); status.set('live') }
    })
    // Browser/media controls can pause the element outside our buttons.
    // Reflect that state instead of showing a frozen trace as live playback.
    element.addEventListener('pause', () => {
      if (current() && element.paused && !element.ended) stop()
    })
    element.addEventListener('waiting', () => {
      if (current()) { status.set('connecting'); clearTimeout(timeout); timeout = setTimeout(fail, 15000) }
    })
    element.addEventListener('error', fail)
    element.addEventListener('ended', fail)
    timeout = setTimeout(fail, 15000)
    element.src = next.url
    try {
      if (metered) {
        audioContext ??= new AudioContext()
        analyser = audioContext.createAnalyser()
        analyser.fftSize = waveformData.length
        source = audioContext.createMediaElementSource(element)
        output = audioContext.createGain()
        output.gain.value = volume.get() / 100
        source.connect(analyser)
        analyser.connect(output)
        output.connect(audioContext.destination)
        await audioContext.resume()
        if (!current()) return
      }
      await element.play()
    } catch { fail() }
  }

  function toggle() {
    if (status.get() === 'live' || status.get() === 'connecting') stop()
    else void play()
  }

  function next() {
    const queue = uniqueStations([...favorites.get(), ...PRESETS, station.get()])
    const index = queue.findIndex(item => sameStation(item, station.get()))
    void play(queue[(index + 1) % queue.length])
  }

  function setVolume(value) {
    const nextVolume = Math.max(0, Math.min(100, value))
    volume.set(nextVolume)
    if (output) output.gain.value = nextVolume / 100
    else if (audio) audio.volume = nextVolume / 100
    ctx.storage.set('local.volume', nextVolume)
  }

  function mute() {
    if (volume.get()) { lastVolume = volume.get(); setVolume(0) }
    else setVolume(lastVolume)
  }

  function favorite(item) {
    const current = favorites.get()
    const nextFavorites = current.some(value => sameStation(value, item)) ? current.filter(value => !sameStation(value, item)) : [...current, item]
    favorites.set(nextFavorites)
    ctx.storage.set('local.favorites', nextFavorites)
  }

  bus.onmessage = event => {
    if (event.data?.type === 'play' && event.data.sender !== windowId) stop('elsewhere')
  }
  function waveform(columns) {
    if (!analyser || status.get() !== 'live') return Array(columns).fill(0)
    analyser.getFloatTimeDomainData(waveformData)
    // A short, rising-edge-triggered trace reads like an oscilloscope rather
    // than unrelated frequency bars. Fixed gain leaves silence genuinely flat.
    const span = Math.floor(audioContext.sampleRate * .004)
    let start = 0
    for (let i = 1; i < waveformData.length - span; i++) {
      if (waveformData[i - 1] < 0 && waveformData[i] >= 0) { start = i; break }
    }
    return Array.from({ length: columns }, (_, x) => {
      const from = start + Math.floor(x * span / columns)
      const to = start + Math.floor((x + 1) * span / columns)
      let sum = 0
      for (let i = from; i < to; i++) sum += waveformData[i]
      return Math.max(-1, Math.min(1, sum / Math.max(1, to - from) * 2))
    })
  }

  ctx.onDispose(() => {
    disposed = true
    stop()
    bus.close()
    if (audioContext) void audioContext.close()
  })
  return { station, favorites, volume, status, meterMode, open, play, stop, toggle, next, setVolume, mute, favorite, waveform }
}

function brailleTrace(samples, rows = 2) {
  const width = samples.length / 2
  const cells = Array(width * rows).fill(0)
  const dots = [[1, 2, 4, 64], [8, 16, 32, 128]]
  const center = (rows * 4 - 1) / 2
  const row = value => Math.max(0, Math.min(rows * 4 - 1, Math.round(center - value * center)))
  let previous = row(samples[0])
  samples.forEach((value, x) => {
    const y = row(value)
    for (let point = Math.min(previous, y); point <= Math.max(previous, y); point++) {
      cells[Math.floor(point / 4) * width + Math.floor(x / 2)] |= dots[x % 2][point % 4]
    }
    previous = y
  })
  return cells.map(cell => String.fromCharCode(0x2800 | cell))
}

function Waveform({ player, compact = false }) {
  const width = compact ? 6 : 36
  const rows = compact ? 1 : 2
  const ref = useRef(null)
  const status = useValue(player.status)
  const mode = useValue(player.meterMode)
  const station = useValue(player.station)
  const active = status === 'live'
  const baseline = brailleTrace(Array(width * 2).fill(0), rows)
  useEffect(() => {
    const node = ref.current
    if (!node) return
    const blank = Array(width * rows).fill('⠀')
    const flat = brailleTrace(Array(width * 2).fill(0), rows)
    const layers = [...node.children]
    const paint = (layer, cells) => cells.forEach((cell, i) => {
      if (layers[layer].children[i].textContent !== cell) layers[layer].children[i].textContent = cell
    })
    let history = [blank, blank]
    const reset = () => { history = [blank, blank]; paint(0, blank); paint(1, blank); paint(2, flat) }
    reset()
    const motion = matchMedia('(prefers-reduced-motion: reduce)')
    let step = 0
    // Phosphor-style persistence: two dim previous traces, no glow or fake signal.
    // Inspired by hubertlim/oscilloscope_playground (MIT); see README.
    const loop = createBudgetedLoop(() => {
      if (!active || motion.matches) return
      if (mode === 'activity') {
        const cells = [...blank]
        cells[Math.floor(step++ / 2) % width] = '⠤'
        paint(0, blank); paint(1, blank); paint(2, cells)
      } else {
        const cells = brailleTrace(player.waveform(width * 2), rows)
        paint(0, history[0]); paint(1, history[1]); paint(2, cells)
        history = [history[1], cells]
      }
    }, { fps: 12, pauseWhenUnfocused: false, idleWhen: () => !active || motion.matches })
    const updateMotion = () => { reset(); loop.wake() }
    motion.addEventListener('change', updateMotion)
    return () => { loop.dispose(); motion.removeEventListener('change', updateMotion) }
  }, [active, mode, player, rows, width, station.url])
  return jsx('div', { ref, className: 'hermes-radio-waveform', style: { '--radio-columns': width }, 'data-radio-meter': width, 'data-compact': compact, 'data-meter-mode': mode, 'data-active': active, 'aria-hidden': true,
    children: [0, 1, 2].map(layer => jsx('span', { className: 'hermes-radio-waveform-layer', children: baseline.map((cell, i) => jsx('span', { children: layer === 2 ? cell : '⠀' }, i)) }, layer)) })
}

function NextArrow() {
  return jsxs('span', { className: 'hermes-radio-next', 'aria-hidden': true, children: [
    jsx(icons.Play, { style: { width: 7, height: 12, flexShrink: 0 }, fill: 'currentColor' }),
    jsx(icons.Play, { style: { width: 7, height: 12, flexShrink: 0, marginLeft: -2 }, fill: 'currentColor' })
  ] })
}

function SmallAction({ label, icon, onClick, size = 'micro', pressed, className, busy = false }) {
  return jsx(Tip, { label, children: jsx(Button, {
    variant: 'ghost', size, className: `hermes-radio-action ${className ?? ''}`, 'aria-label': label, 'aria-pressed': pressed, onClick,
    children: jsx('span', { className: 'hermes-radio-action-icon', children: busy ? jsx(GlyphSpinner, { ariaLabel: label }) : jsx(icon, {}) })
  }) })
}

function trackCredit(source) {
  const raw = typeof source?.title === 'string' ? source.title.trim() : ''
  const explicit = source?.artist || source?.metadata?.artist
  const parts = raw.match(/^(.+?)\s[-–—]\s(.+)$/)
  const artist = typeof explicit === 'string' ? explicit.trim() : parts?.[1].trim() || ''
  // Only separate an explicit artist prefix when it actually matches.
  const title = parts && parts[1].trim() === artist ? parts[2].trim() : raw
  return { artist: raw ? artist : '', title }
}

async function artistRequest(path, signal, ctx) {
  // The lock and persisted timestamp pace lookups across desktop windows.
  return navigator.locks.request('hermes:radio:artist-lookup', { signal }, async () => {
    const delay = Math.max(0, 1100 - (Date.now() - ctx.storage.get('local.artistRequestAt', 0)))
    if (delay) await new Promise(resolve => setTimeout(resolve, delay))
    signal.throwIfAborted()
    ctx.storage.set('local.artistRequestAt', Date.now())
    const response = await fetch(`https://musicbrainz.org/ws/2/${path}`, {
      signal: AbortSignal.any([signal, AbortSignal.timeout(15000)]), credentials: 'omit',
      headers: { 'User-Agent': 'HermesRadio/1.0 (https://github.com/NousResearch/hermes-agent)' }
    })
    if (!response.ok) throw new Error(`Artist lookup: HTTP ${response.status}`)
    return response.json()
  })
}

function artistProfile(relations) {
  const links = (relations ?? []).filter(item => !item.ended).flatMap(item => {
    const href = httpsUrl(item.url?.resource)
    if (!href) return []
    const url = new URL(href)
    if (url.hostname === 'open.spotify.com' && /^\/artist\/[a-zA-Z0-9]{22}\/?$/.test(url.pathname)) return [{ href, rank: 0 }]
    if (item.type === 'bandcamp' && /^[^.]+\.bandcamp\.com$/.test(url.hostname) && url.pathname === '/') return [{ href, rank: 1 }]
    if (item.type === 'official homepage' && !['open.spotify.com', 'bandcamp.com'].includes(url.hostname)) return [{ href, rank: 2 }]
    return []
  })
  return links.sort((a, b) => a.rank - b.rank)[0]?.href ?? null
}

function normalizedArtist(name) {
  return name.normalize('NFKC').trim().toLowerCase()
}

async function resolveArtist(artist, signal, ctx) {
  // Apple's public artist search supplies direct profile URLs without API keys.
  // Prefer it over a slow/unavailable MusicBrainz lookup; never use song results.
  const params = new URLSearchParams({ term: artist, entity: 'musicArtist', limit: '10' })
  try {
    const result = await fetchJson(`https://itunes.apple.com/search?${params}`, signal)
    const matches = (result.results ?? []).filter(item => item.wrapperType === 'artist' && normalizedArtist(item.artistName) === normalizedArtist(artist))
    if (matches.length > 1) return null
    if (matches.length === 1) {
      const href = httpsUrl(matches[0].artistLinkUrl)
      if (href) {
        const url = new URL(href)
        if (url.hostname === 'music.apple.com' && /^\/[a-z]{2}\/artist\/[^/]+\/\d+$/.test(url.pathname)) {
          url.search = ''
          return url.href
        }
      }
    }
  } catch (error) {
    if (signal.aborted) throw error
  }
  return resolveMusicBrainzArtist(artist, signal, ctx)
}

async function resolveMusicBrainzArtist(artist, signal, ctx) {
  const escaped = artist.replace(/[+\-!(){}\[\]^"~*?:\\/&|]/g, '\\$&')
  const params = new URLSearchParams({ query: `artist:"${escaped}"`, fmt: 'json', limit: '5' })
  const result = await artistRequest(`artist/?${params}`, signal, ctx)
  const matches = (result.artists ?? []).filter(item => normalizedArtist(item.name) === normalizedArtist(artist))
  // Never turn the top fuzzy match or an ambiguous artist name into a credit.
  if (matches.length !== 1 || matches[0].score !== 100 || result.artists.length === 5) return null
  const id = matches[0].id
  if (!/^[a-f0-9]{8}(?:-[a-f0-9]{4}){3}-[a-f0-9]{12}$/.test(id)) return null
  const profile = await artistRequest(`artist/${id}?inc=url-rels&fmt=json`, signal, ctx)
  return artistProfile(profile.relations)
}

function Signal({ player, ctx }) {
  const t = usePluginI18n(ID)
  const station = useValue(player.station)
  const status = useValue(player.status)
  const mode = useValue(player.meterMode)
  const stream = new URL(station.url)
  const endpoints = {
    'stream.nightride.fm': `${NIGHTRIDE}status-json.xsl`,
    'media01.gamingnow.net': 'https://media01.gamingnow.net:8010/status-json.xsl'
  }
  const endpoint = endpoints[stream.hostname]
  const metadata = useQuery({
    queryKey: [ID, 'metadata', endpoint],
    queryFn: ({ signal }) => fetchJson(endpoint, signal),
    enabled: status === 'live' && Boolean(endpoint),
    staleTime: 20000, refetchInterval: status === 'live' ? 25000 : false, retry: 1
  })
  const sources = metadata.data?.icestats?.source
  const source = (Array.isArray(sources) ? sources : sources ? [sources] : []).find(source => source.listenurl?.endsWith(stream.pathname))
  const track = status === 'live' && typeof source?.title === 'string' ? source.title : ''
  const credit = trackCredit(track ? source : null)
  const artist = useQuery({
    queryKey: [ID, 'artist-profile', credit.artist],
    queryFn: ({ signal }) => resolveArtist(credit.artist, signal, ctx),
    enabled: Boolean(credit.artist), staleTime: 86400000, gcTime: 86400000,
    retry: 1, retryDelay: 3000, refetchOnWindowFocus: false
  })
  const artistUrl = credit.artist ? artist.data : null
  const text = status === 'live' ? track || (mode === 'activity' ? t('audioOnly') : '') : t(status)
  return jsxs('div', { className: 'hermes-radio-signal', children: [
    jsx(Waveform, { player }),
    jsxs('div', { className: 'hermes-radio-track', 'data-track': Boolean(track), role: 'status', children: [
      status === 'connecting' && jsx(GlyphSpinner, { ariaLabel: t('connecting') }),
      credit.artist && (artistUrl ? jsx(Tip, { label: `${t('artistProfile')}: ${credit.artist}`, children: jsx(Button, {
        variant: 'text', size: 'inline', asChild: true, className: 'hermes-radio-artist',
        children: jsxs('a', { href: artistUrl, target: '_blank', rel: 'noopener noreferrer',
          'aria-label': `${t('artistProfile')}: ${credit.artist}`,
          onClick: event => { event.preventDefault(); void ctx.os.openExternal(artistUrl) },
          children: [jsx('span', { children: credit.artist }), jsx(icons.ExternalLink, { style: { width: 10, height: 10 }, 'aria-hidden': true })]
        })
      }) }) : jsx('span', { className: 'hermes-radio-track-text', children: credit.artist })),
      credit.artist && jsx('span', { className: 'hermes-radio-track-separator', 'aria-hidden': true, children: '—' }),
      jsx(Tip, { label: text || undefined, children: jsx('span', { className: 'hermes-radio-track-text', children: credit.artist ? credit.title : text }) })
    ] })
  ] })
}

function StationRow({ station, player }) {
  const t = usePluginI18n(ID)
  const current = sameStation(useValue(player.station), station)
  const status = useValue(player.status)
  const active = status === 'live' || status === 'connecting'
  const failed = current && status === 'error'
  const saved = useValue(player.favorites).some(item => sameStation(item, station))
  return jsxs('div', { className: 'hermes-radio-row', 'data-current': current, 'data-pinned': saved, children: [
    jsxs(RowButton, { className: 'hermes-radio-row-main', 'aria-label': `${t(failed ? 'retry' : current && active ? 'pause' : 'play')}: ${station.name}`, title: failed ? t('streamError') : undefined, 'aria-current': current ? 'true' : undefined,
      onClick: () => current ? player.toggle() : void player.play(station), children: [
        jsx('span', { className: 'hermes-radio-row-icon', role: failed ? 'status' : undefined, 'aria-label': failed ? t('error') : undefined, children: jsx(failed ? icons.AlertCircle : current && active ? icons.Pause : icons.Play, { size: 12 }) }),
        jsx('span', { className: 'hermes-radio-row-copy', children: jsx('span', { className: 'hermes-radio-row-name', children: station.name }) })
      ] }),
    jsx(SmallAction, { label: `${t(saved ? 'unsave' : 'save')}: ${station.name}`, icon: icons.Pin, onClick: () => player.favorite(station), size: 'icon-xs', pressed: saved, className: 'hermes-radio-pin' })
  ] })
}

function Stations({ player }) {
  const t = usePluginI18n(ID)
  const favorites = useValue(player.favorites)
  const current = useValue(player.station)
  const [browseStation, setBrowseStation] = useState(() => player.station.get())
  const listRef = useRef(null)
  const [search, setSearch] = useState('')
  const [query, setQuery] = useState('')
  const text = search.trim()
  useEffect(() => {
    if (listRef.current) listRef.current.scrollTop = 0
  }, [text])
  useEffect(() => {
    const timer = setTimeout(() => setQuery(text), 350)
    return () => clearTimeout(timer)
  }, [text])
  const result = useQuery({
    queryKey: [ID, 'search', query], queryFn: ({ signal }) => searchStations(query, signal),
    enabled: query.length >= 2 && text === query, staleTime: 300000, retry: false
  })
  const matches = item => [item.name, item.description, item.provider].some(value => value?.toLowerCase().includes(text.toLowerCase()))
  const local = uniqueStations([...favorites, ...PRESETS]).filter(matches)
  const remote = text.length >= 2 && query === text ? result.data ?? [] : []
  const results = uniqueStations([...local, ...remote])
  // Keep the previously prepended row until the search changes: selecting a
  // result must not remove a row above it and move the target under the pointer.
  const anchored = results.some(item => sameStation(item, browseStation)) ? results : [browseStation, ...results]
  const items = anchored.some(item => sameStation(item, current)) ? anchored : [current, ...anchored]
  const pending = text.length >= 2 && (text !== query || result.isFetching)
  const searchFailed = text.length >= 2 && query === text && result.isError
  return jsxs('div', { children: [
    jsx('div', { className: 'hermes-radio-search', 'data-filled': Boolean(search), children: jsx(SearchField, {
      value: search, onChange: value => { setBrowseStation(player.station.get()); setSearch(value) }, placeholder: t('search'),
      containerClassName: 'hermes-radio-search-field',
      trailingAction: pending ? jsx('span', { className: 'hermes-radio-search-loader', children: jsx(GlyphSpinner, { ariaLabel: t('searching') }) }) : null
    }) }),
    jsxs('div', { ref: listRef, className: 'hermes-radio-list', 'data-radio-pending': pending, children: [
      ...items.map(station => jsx(StationRow, { station, player }, station.id)),
      searchFailed && jsx('div', { className: 'hermes-radio-error', role: 'status', children: t('searchError') }),
      !pending && !searchFailed && text && results.length === 0 && jsx('div', { className: 'hermes-radio-error', role: 'status', children: t('noResults') })
    ] })
  ] })
}

function Transport({ player, ctx }) {
  const t = usePluginI18n(ID)
  const volume = useValue(player.volume)
  const station = useValue(player.station)
  const status = useValue(player.status)
  const active = status === 'live' || status === 'connecting'
  return jsxs('div', { className: 'hermes-radio-volume', children: [
    jsx(SmallAction, { label: t(active ? 'pause' : 'play'), icon: active ? icons.Pause : icons.Play, busy: status === 'connecting', size: 'icon-xs', onClick: player.toggle }),
    jsx(SmallAction, { label: t('next'), icon: NextArrow, size: 'icon-xs', onClick: player.next }),
    jsx('span', { className: 'hermes-radio-volume-space' }),
    jsx(SmallAction, { label: t(volume ? 'mute' : 'unmute'), icon: volume ? icons.Volume2 : icons.VolumeX, onClick: player.mute, size: 'icon-xs' }),
    jsx('input', { type: 'range', min: 0, max: 100, step: 1, value: volume, 'aria-label': t('volume'), onChange: event => player.setVolume(Number(event.target.value)) }),
    jsx('span', { className: 'hermes-radio-website', children: httpsUrl(station.homepage) && jsx(SmallAction, { label: t('visit'), icon: icons.ExternalLink, size: 'icon-xs', onClick: () => void ctx.os.openExternal(station.homepage) }) })
  ] })
}

function RadioBar({ player, ctx }) {
  const t = usePluginI18n(ID)
  const station = useValue(player.station)
  const status = useValue(player.status)
  const open = useValue(player.open)
  const triggerRef = useRef(null)
  const [browseWidth, setBrowseWidth] = useState(null)
  const active = status === 'live' || status === 'connecting'
  // Fit the name at rest. Freeze only while browsing so a selection cannot
  // move the open picker; there is no permanent empty label slot.
  const onOpenChange = value => {
    if (value) setBrowseWidth(triggerRef.current.getBoundingClientRect().width)
    player.open.set(value)
  }
  return jsxs('div', { className: 'hermes-radio-bar', 'data-tour': 'radio-player', 'data-radio-status': status, children: [
    jsx(Waveform, { player, compact: true }),
    jsxs(Popover, { open, onOpenChange, children: [
      jsx(PopoverTrigger, { asChild: true, children: jsx(Button, {
        ref: triggerRef, variant: 'ghost', size: 'micro', style: { width: open ? browseWidth : undefined },
        'aria-label': `${t('browse')}: ${station.name}`,
        children: jsx('span', { className: 'radio-name', children: station.name })
      }) }),
      jsx(PopoverContent, { side: 'top', align: 'end', className: 'hermes-radio-panel', 'aria-label': t('radio'), 'data-tour': 'radio-panel', children:
        jsxs('div', { children: [
          jsx(Stations, { player }), jsx(Signal, { player, ctx }), jsx(Transport, { player, ctx })
        ] })
      })
    ] }),
    jsx(SmallAction, { label: t(active ? 'pause' : 'play'), icon: active ? icons.Pause : icons.Play, busy: status === 'connecting', onClick: player.toggle }),
    jsx(SmallAction, { label: t('next'), icon: NextArrow, onClick: player.next })
  ] })
}

export { artistProfile, resolveArtist, trackCredit }

export default {
  id: ID,
  name: 'Radio',
  description: 'Free live radio with pinned stations, search, and an audio-reactive waveform.',
  defaultEnabled: false,
  register(ctx) {
    ctx.i18n.register(LOCALES)
    const style = document.createElement('style')
    style.textContent = CSS
    document.head.append(style)
    ctx.onDispose(() => style.remove())
    const player = createPlayer(ctx)
    ctx.register({ id: 'player', area: STATUSBAR_AREAS.right, order: 10, render: () => jsx(RadioBar, { player, ctx }) })
  }
}
