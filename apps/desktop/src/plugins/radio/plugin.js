import {
  atom, Button, createBudgetedLoop, GlyphSpinner, icons, Popover, PopoverContent, PopoverTrigger, randomUuid,
  RowButton, SearchField, STATUSBAR_AREAS, Tip, usePluginI18n,
  useQuery, useValue
} from '@hermes/plugin-sdk'
import { useEffect, useRef, useState } from 'react'
import { jsx, jsxs } from 'react/jsx-runtime'

const ID = 'radio'
// Optional launch behavior lives in plugin storage, beside the other Radio settings.
const AUTOPLAY_KEY = 'local.autoplay'

function autoplayOnLaunch(storage) {
  return storage?.get?.(AUTOPLAY_KEY, false) === true
}

function shouldAutoplay(storage, station) {
  return autoplayOnLaunch(storage) && validStation(station)
}
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
  search: 'Find a station… country:uk genre:jazz', loadingMore: 'Loading more stations…', volume: 'Volume', audioOnly: 'Audio playing · visualizer unavailable',
  mute: 'Mute radio', unmute: 'Unmute radio', save: 'Pin station', unsave: 'Unpin station',
  visit: 'Visit station website', artistProfile: 'Visit artist profile',
  noResults: 'No stations found', searchHint: 'Search by name, or narrow it with country:uk and genre:jazz.',
  searchError: 'Station search is unavailable. Your presets still work.', searching: 'Searching stations',
  directory: 'Search powered by Radio Browser', note: 'Live radio · next switches stations',
  streamError: 'This stream could not connect. Try again or choose another station.',
  popularError: 'The popular list is unavailable. Your stations still work.',
  shoutcastEmpty: 'No Shoutcast stations found', shoutcastError: 'Shoutcast is unavailable. Try again, or restart Hermes if the Radio backend is not loaded.',
  tuneinEmpty: 'No TuneIn stations found', tuneinHint: 'Type a station name. TuneIn has no top list, but genre:jazz works.',
  tuneinError: 'TuneIn is unavailable. Try again, or restart Hermes if the Radio backend is not loaded.',
  sourceAll: 'All', sourcePopular: 'Popular', sourceShoutcast: 'Shoutcast', sourceTunein: 'TuneIn',
  hlsNote: 'HLS streams are hidden in this edition.', skippedNote: 'unplayable results hidden',
  close: 'Close radio', nowPlaying: 'Now playing', elsewhere: 'Paused — playing in another window',
  settingsTitle: 'Radio', autoplayLabel: 'Start the last station on launch',
  autoplayDescription: 'Plays the station you listened to last when Hermes opens. Off by default; with it off nothing plays until you press play.',
  restoreLabel: 'Restore the default stations', restoreDescription: 'Adds back any station that shipped with Radio and is no longer in your list. Stations you added and names you changed stay as they are.',
  restoreAction: 'Restore', restoreNone: 'All default stations are already in your list.', restoreSome: 'Restored',
  rename: 'Rename station', remove: 'Delete station', confirmRemove: 'Click again to delete',
  add: 'Add station', addName: 'Station name', addUrl: 'Stream URL (http or https)', addSave: 'Add', cancel: 'Cancel',
  addInvalid: 'Enter an http or https stream address.', addExists: 'That stream URL is already in your list.',
  edit: 'Edit station', editName: 'Station name', editUrl: 'Stream URL', editSave: 'Save', editInvalid: 'Enter an http or https stream address.',
  'chillsynthDescription': 'Soft focus · warm synths', 'nightrideDescription': 'Synthwave · after hours',
  'darksynthDescription': 'Dark electronics · high energy', 'spacesynthDescription': 'Cosmic synths · retro futures',
  'paradise-mainDescription': 'Eclectic · human selected', 'paradise-mellowDescription': 'A gentler pace'
}
const LOCALES = {
  en: EN,
  ja: { ...EN, artistProfile: 'アーティストのプロフィールを開く', audioOnly: '再生中 · 波形を表示できません', radio: 'ラジオ', browse: 'ステーションを選択', play: 'ラジオを再生', pause: 'ラジオを一時停止', next: '次のステーション', live: 'ライブ', paused: '一時停止', connecting: '接続中', error: '再生できません', retry: '再試行', search: 'ステーションを検索…', volume: '音量', mute: 'ミュート', unmute: 'ミュート解除', save: 'ステーションを固定', unsave: '固定を解除', visit: '公式サイトを開く', noResults: '見つかりませんでした', searchHint: '名前で検索。jazz、ambient、house など。', searchError: '検索できません。おすすめは再生できます。', searching: '検索中', directory: 'Radio Browser による検索', note: 'ライブ放送 · 次へでステーションを切り替え', streamError: '接続できませんでした。再試行するか別のステーションを選んでください。', close: 'ラジオを閉じる', nowPlaying: '再生中', elsewhere: '別のウィンドウで再生中', 'chillsynthDescription': 'やさしい集中 · 暖かなシンセ', 'nightrideDescription': 'シンセウェーブ · 深夜', 'darksynthDescription': 'ダークな電子音 · 高揚感', 'spacesynthDescription': '宇宙的シンセ · レトロな未来', 'paradise-mainDescription': '多彩な選曲 · 人がキュレーション', 'paradise-mellowDescription': 'ゆったりした時間', restoreLabel: '既定のステーションを復元', restoreAction: '復元' },
  zh: { ...EN, artistProfile: '访问艺人主页', audioOnly: '正在播放 · 无法显示波形', radio: '电台', browse: '选择电台', play: '播放电台', pause: '暂停电台', next: '下一个电台', live: '直播', paused: '已暂停', connecting: '正在连接', error: '无法播放', retry: '重试', search: '搜索电台…', volume: '音量', mute: '静音', unmute: '取消静音', save: '置顶电台', unsave: '取消置顶', visit: '访问电台网站', noResults: '没有找到电台', searchHint: '按名称搜索，例如 jazz、ambient 或 house。', searchError: '暂时无法搜索，精选电台仍可使用。', searching: '正在搜索', directory: '搜索由 Radio Browser 提供', note: '直播电台 · 下一首将切换电台', streamError: '无法连接，请重试或选择其他电台。', close: '关闭电台', nowPlaying: '正在播放', elsewhere: '已暂停，正在其他窗口播放', 'chillsynthDescription': '轻松专注 · 温暖合成器', 'nightrideDescription': '合成器浪潮 · 深夜', 'darksynthDescription': '暗黑电子 · 充满能量', 'spacesynthDescription': '宇宙合成器 · 复古未来', 'paradise-mainDescription': '多元风格 · 人工精选', 'paradise-mellowDescription': '放慢节奏', restoreLabel: '恢复默认电台', restoreAction: '恢复' },
  'zh-hant': { ...EN, artistProfile: '造訪藝人主頁', audioOnly: '播放中 · 無法顯示波形', radio: '電台', browse: '選擇電台', play: '播放電台', pause: '暫停電台', next: '下一個電台', live: '直播', paused: '已暫停', connecting: '正在連線', error: '無法播放', retry: '重試', search: '搜尋電台…', volume: '音量', mute: '靜音', unmute: '取消靜音', save: '釘選電台', unsave: '取消釘選', visit: '造訪電台網站', noResults: '找不到電台', searchHint: '按名稱搜尋，例如 jazz、ambient 或 house。', searchError: '暫時無法搜尋，精選電台仍可使用。', searching: '正在搜尋', directory: '搜尋由 Radio Browser 提供', note: '直播電台 · 下一首會切換電台', streamError: '無法連線，請重試或選擇其他電台。', close: '關閉電台', nowPlaying: '正在播放', elsewhere: '已暫停，正在其他視窗播放', 'chillsynthDescription': '輕鬆專注 · 溫暖合成器', 'nightrideDescription': '合成器浪潮 · 深夜', 'darksynthDescription': '暗黑電子 · 充滿能量', 'spacesynthDescription': '宇宙合成器 · 復古未來', 'paradise-mainDescription': '多元風格 · 人工精選', 'paradise-mellowDescription': '放慢節奏', restoreLabel: '恢復預設電台', restoreAction: '恢復' }
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
.hermes-radio-row-edit{opacity:0;pointer-events:none}
.hermes-radio-row:hover .hermes-radio-row-edit,.hermes-radio-row:focus-within .hermes-radio-row-edit{opacity:1;pointer-events:auto}
@media(hover:none){.hermes-radio-row-edit{opacity:1;pointer-events:auto}}
.hermes-radio-row[data-armed=true] .hermes-radio-row-remove{color:var(--ui-accent)}
.hermes-radio-edit,.hermes-radio-add{display:flex;flex-direction:column;gap:4px;padding:5px 6px 6px;width:100%}
.hermes-radio-edit input,.hermes-radio-add input{width:100%;min-width:0;height:22px;padding:0 5px;border-radius:4px;border:1px solid var(--ui-stroke-secondary);background:transparent;color:var(--ui-text-primary);font:inherit;font-size:12px}
.hermes-radio-edit input:focus-visible,.hermes-radio-add input:focus-visible{outline:1px solid var(--ui-accent);outline-offset:-1px}
.hermes-radio-edit-actions,.hermes-radio-add-actions{display:flex;align-items:center;gap:4px;flex-wrap:wrap}
.hermes-radio-add-note{font-size:11px;line-height:16px;color:var(--ui-text-secondary)}
.hermes-radio-actions{display:flex;align-items:center;gap:3px;padding:0 4px 5px}
.hermes-radio-chip{height:20px;padding:0 8px;border:0;border-radius:10px;font-family:inherit;font-size:11px;line-height:20px;background:transparent;color:var(--ui-text-tertiary);opacity:.7;cursor:pointer;transition:background .12s,color .12s,opacity .12s}
.hermes-radio-chip:hover{background:var(--chrome-action-hover);color:var(--ui-text-secondary);opacity:1}
.hermes-radio-chip[data-active=true]{background:var(--ui-bg-quaternary);color:var(--ui-text-primary);opacity:1;box-shadow:inset 0 0 0 .5px var(--ui-stroke-secondary)}
.hermes-radio-chip:focus-visible{outline:1px solid var(--ui-accent);outline-offset:1px}
.hermes-radio-sources-space{flex:1;min-width:6px}
.hermes-radio-addbtn{display:flex;align-items:center;justify-content:center;width:20px;height:20px;padding:0;border:0;border-radius:999px;background:var(--ui-bg-quaternary);color:var(--ui-text-secondary);cursor:pointer;flex-shrink:0;opacity:1;pointer-events:auto}
.hermes-radio-addbtn[data-active=true]{color:var(--ui-accent);box-shadow:inset 0 0 0 1px var(--ui-accent)}
.hermes-radio-volume{display:flex;align-items:center;gap:4px;height:28px;padding:2px 2px 0}
.hermes-radio-volume-space{flex:1;min-width:6px}
.hermes-radio-volume input{width:76px;min-width:0;height:3px;accent-color:var(--ui-accent);cursor:pointer}
.hermes-radio-error{font-size:11px;line-height:17px;color:var(--ui-text-secondary);padding:4px 4px 12px}
.hermes-radio-more{display:flex;align-items:center;justify-content:center;padding:6px 0;font-size:11px;opacity:.6}
.hermes-radio-setting{display:flex;align-items:center;justify-content:space-between;gap:16px;padding:10px 0}
.hermes-radio-setting-text{display:flex;flex-direction:column;gap:4px;max-width:62ch}
.hermes-radio-setting-label{font-size:13px;font-weight:500}
.hermes-radio-setting-note{font-size:12px;opacity:.72;line-height:1.45}
`

function httpsUrl(value) {
  try {
    const url = new URL(value)
    return url.protocol === 'https:' && !url.username && !url.password ? url.href : null
  } catch { return null }
}

function validStation(value) {
  return Boolean(value) && typeof value.id === 'string' && typeof value.name === 'string' && Boolean(streamUrl(value.url))
}

// Stream addresses are handed to an <audio> element, not fetched with credentials.
// Shoutcast still serves many stations over plain http, so both schemes are allowed;
// website and artist links keep the stricter https rule above.
function streamUrl(value) {
  try {
    const url = new URL(value)
    if (!['http:', 'https:'].includes(url.protocol) || url.username || url.password) return null
    return url.href
  } catch { return null }
}

async function nowPlaying(ctx, url) {
  if (typeof ctx.rest !== 'function') return { ok: false, title: '' }
  return ctx.rest(`/now-playing?${new URLSearchParams({ url })}`, { timeoutMs: 12000 })
}

async function fetchJson(url, signal) {
  const response = await fetch(url, { signal: AbortSignal.any([signal, AbortSignal.timeout(10000)]), credentials: 'omit' })
  if (!response.ok) throw new Error(`HTTP ${response.status}`)
  return response.json()
}

let directoryServers = ['https://de1.api.radio-browser.info']
let discovered = false
async function ensureDirectory(signal) {
  if (discovered) return
  try {
    const servers = await fetchJson(`${directoryServers[0]}/json/servers`, signal)
    const hosts = [...new Set(servers.map(server => server.name).filter(name => /^[a-z0-9-]+\.api\.radio-browser\.info$/.test(name)))]
    if (hosts.length) directoryServers = hosts.map(name => `https://${name}`).sort(() => Math.random() - 0.5)
    discovered = true
  } catch (error) {
    if (signal.aborted) throw error
  }
}

// Radio Browser is the one directory the renderer may call directly: it sends
// CORS headers. Other directories go through the bundled backend.
function browseStation(row) {
  const url = streamUrl(row.url_resolved)
  if (!url || !row.stationuuid || !row.name) return []
  return [{
    id: row.stationuuid,
    name: row.name.trim(),
    url,
    provider: 'Radio Browser',
    description: row.tags?.split(',').slice(0, 2).join(' · ') || '',
    homepage: httpsUrl(row.homepage)
  }]
}

function isHlsRow(row) {
  return Boolean(row?.hls) || /\.m3u8(?:$|\?)/i.test(row?.url_resolved ?? row?.url ?? '')
}

function rowsToStations(rows) {
  const stations = []
  let hls = 0
  const list = Array.isArray(rows) ? rows : []
  for (const row of list) {
    if (isHlsRow(row)) { hls += 1; continue }
    stations.push(...browseStation(row))
  }
  // The raw count, not the station count: a full page means there is probably
  // another page behind it even when some rows did not map to a playable URL.
  return { stations, hls, count: list.length }
}

async function browseRows(path, signal) {
  for (const server of directoryServers.slice(0, 3)) {
    try {
      return await fetchJson(`${server}/json/${path}`, signal)
    } catch (error) {
      if (signal.aborted) throw error
    }
  }
  throw new Error('Directory unavailable')
}

const PAGE_SIZE = 40
// The directory sends its whole list at once, so pages from the two proxied
// sources are smaller: every row costs a resolve and probe before it reaches UI.
const SHOUTCAST_PAGE = 30
const TUNEIN_PAGE = 30
const TAG_PATTERN = /\b(country|genre):("[^"]+"|\S+)/gi

function parseQuery(text) {
  let country = ''
  let genre = ''
  const name = String(text ?? '')
    .replace(TAG_PATTERN, (match, key, value) => {
      const clean = String(value).replace(/^"|"$/g, '').trim()
      if (clean) {
        if (key.toLowerCase() === 'country') country = clean
        else genre = clean
      }
      return ' '
    })
    .replace(/\s+/g, ' ')
    .trim()
  return { name, country, genre }
}

const COUNTRY_ALIASES = new Map([
  ['uk', 'GB'], ['gb', 'GB'], ['britain', 'GB'], ['england', 'GB'],
  ['scotland', 'GB'], ['wales', 'GB'], ['northern ireland', 'GB'],
  ['usa', 'US'], ['uae', 'AE'], ['holland', 'NL']
])

function searchParams({ name = '', country = '', genre = '', limit = PAGE_SIZE, offset = 0 } = {}) {
  const params = new URLSearchParams({
    limit: String(limit), hidebroken: 'true', order: 'votes', reverse: 'true'
  })
  if (offset > 0) params.set('offset', String(offset))
  if (name) params.set('name', name)
  if (country) {
    const alias = COUNTRY_ALIASES.get(country.toLowerCase())
    if (alias) params.set('countrycode', alias)
    else if (country.length === 2) params.set('countrycode', country.toUpperCase())
    else params.set('country', country)
  }
  if (genre) params.set('tag', genre)
  return params
}

async function browserQuery(query, signal) {
  return rowsToStations(await browseRows(`stations/search?${searchParams(query)}`, signal))
}

const CYRILLIC_LATIN = {
  а: 'a', б: 'b', в: 'v', г: 'g', д: 'd', е: 'e', ж: 'zh', з: 'z', и: 'i', й: 'y', к: 'k', л: 'l', м: 'm',
  н: 'n', о: 'o', п: 'p', р: 'r', с: 's', т: 't', у: 'u', ф: 'f', х: 'h', ц: 'ts', ч: 'ch', ш: 'sh', щ: 'sht',
  ъ: 'a', ь: '', ю: 'yu', я: 'ya'
}

function transliterate(text) {
  return String(text ?? '').toLowerCase().replace(/[\u0400-\u04ff]/g, letter => CYRILLIC_LATIN[letter] ?? letter)
}

function fold(text) {
  return transliterate(text).normalize('NFKD').replace(/[\u0300-\u036f]/g, '')
}

function searchTerms(text) {
  const typed = String(text ?? '').trim()
  const latin = transliterate(typed).replace(/\s+/g, ' ').trim()
  return latin && latin !== typed.toLowerCase() ? [typed, latin] : [typed]
}

function escapeRegExp(value) {
  return String(value).replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
}

function matchScore(item, terms) {
  const name = fold(item.name)
  let best = 0
  for (const term of terms) {
    const needle = fold(term).trim()
    if (!needle) continue
    const wholeWord = new RegExp(`(^|[^a-z0-9])${escapeRegExp(needle)}($|[^a-z0-9])`).test(name)
    if (name === needle) best = Math.max(best, 12)
    else if (wholeWord && name.startsWith(needle)) best = Math.max(best, 9)
    else if (wholeWord) best = Math.max(best, 7)
    else if (name.includes(needle)) best = Math.max(best, 2)
    else {
      const words = needle.split(/\s+/).filter(word => word.length > 2)
      const hits = words.filter(word => name.includes(word)).length
      if (hits) best = Math.max(best, 1 + hits)
    }
  }
  return best
}

async function searchStations(text, signal, { offset = 0, limit = PAGE_SIZE } = {}) {
  await ensureDirectory(signal)
  const parsed = parseQuery(text)
  const terms = parsed.name ? searchTerms(parsed.name) : ['']
  const results = await Promise.all(terms.map(name => browserQuery({ ...parsed, name, offset, limit }, signal)))
  const seen = uniqueStations(results.flatMap(result => result.stations))
  const ranked = parsed.name
    ? seen.map((item, index) => ({ item, index, score: matchScore(item, terms) }))
      .sort((a, b) => b.score - a.score || a.index - b.index)
      .map(entry => entry.item)
    : seen
  return {
    stations: ranked,
    hls: results.reduce((total, result) => total + result.hls, 0),
    hasMore: results.some(result => result.count >= limit),
    nextOffset: offset + limit
  }
}

async function popularStations(signal, { offset = 0, limit = PAGE_SIZE } = {}) {
  await ensureDirectory(signal)
  const result = rowsToStations(await browseRows(`stations/topvote/${limit}?hidebroken=true&offset=${offset}`, signal))
  return { stations: uniqueStations(result.stations), hls: result.hls, hasMore: result.count >= limit, nextOffset: offset + limit }
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
  const stored = ctx.storage.get('local.stations', null)
  // Seed once, then trust the stored list: a renamed or deleted station must
  // not come back on the next launch.
  const stations = atom(uniqueStations(Array.isArray(stored) ? stored.filter(validStation) : PRESETS.map(item => ({ ...item }))))
  const previous = ctx.storage.get('local.station', null)
  const restored = validStation(previous)
    ? uniqueStations([...stations.get(), ...favorites.get()]).find(item => sameStation(item, previous)) ?? null
    : null
  const station = atom(restored ?? stations.get()[0] ?? PRESETS[0])
  const storedVolume = ctx.storage.get('local.volume', 25)
  const volume = atom(Number.isFinite(storedVolume) ? Math.max(0, Math.min(100, storedVolume)) : 25)
  const status = atom('paused')
  const meterMode = atom('levels')
  const lastError = atom('')
  const plainStreams = new Set()
  const open = atom(false)
  let audio = null
  let audioContext = null
  let source = null
  let analyser = null
  let output = null
  const waveformData = new Float32Array(2048)
  let generation = 0
  let playGeneration = -1
  let timeout = null
  let disposed = false
  let lastVolume = volume.get() || 25
  const bus = new BroadcastChannel('hermes:radio:playback')
  const windowId = randomUuid()

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
    lastError.set('')
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
        return
      }
      const reason = mediaFailure(element, next)
      lastError.set(reason)
      reportFailure(ctx, next, reason)
      stop('error')
    }
    element.addEventListener('playing', () => {
      if (current()) { clearTimeout(timeout); playGeneration = token; status.set('live') }
    })
    // Browser/media controls can pause the element outside our buttons.
    // Reflect that state instead of showing a frozen trace as live playback.
    // Non-destructive: an external controller (macOS media keys, a dictation
    // tool pausing audio to listen) may resume the same element afterwards,
    // so the stream must survive. The plugin's own Pause control stays
    // destructive — toggle() advances the generation first, so the pause
    // event it triggers fails the current() guard and still runs stop().
    element.addEventListener('pause', () => {
      if (current() && element.paused && !element.ended) status.set('paused')
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
    else if (!resume()) void play()
  }

  // Dictation tools (Fluid Voice) pause the stream to hear the microphone,
  // then deliver the transcript as a paste-like insertion into the focused
  // field. No media-session resume event ever follows, so the only
  // renderer-visible "dictation is done" signal is the text itself arriving
  // in the composer. On that signal the plugin resumes the preserved stream.
  // Never resumed on window focus — that would override an intentional pause.
  // Returns false when the stream is not intact (fresh start, the plugin's
  // own destructive Pause, a stop from another window), so callers can fall
  // back to a full reconnect.
  function resume() {
    if (disposed || !audio || generation !== playGeneration || status.get() !== 'paused') return false
    const token = generation
    status.set('connecting')
    void audio
      .play()
      .then(() => {
        if (!disposed && token === generation) { clearTimeout(timeout); status.set('live') }
      })
      .catch(() => {
        if (!disposed && token === generation) stop('error')
      })
    return true
  }

  function next() {
    const queue = uniqueStations([...favorites.get(), ...stations.get(), station.get()])
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

  function saveStations(next) {
    stations.set(next)
    ctx.storage.set('local.stations', next)
  }

  function missingDefaults() {
    const list = stations.get()
    return PRESETS.filter(seed => !list.some(item => item.id === seed.id || item.url === seed.url))
  }

  function restoreDefaults() {
    const missing = missingDefaults()
    if (!missing.length) return 0
    saveStations([...stations.get(), ...missing.map(item => ({ ...item }))])
    return missing.length
  }

  function tidyName(value) {
    return String(value ?? '').replace(/\s+/g, ' ').trim().slice(0, 80)
  }

  function updateStation(item, changes) {
    const label = tidyName(changes.name)
    const address = streamUrl(changes.url ?? item.url)
    if (!label || !address) return false
    const apply = value => sameStation(value, item) ? { ...value, name: label, url: address } : value
    saveStations(stations.get().map(apply))
    if (favorites.get().some(value => sameStation(value, item))) {
      const nextFavorites = favorites.get().map(apply)
      favorites.set(nextFavorites)
      ctx.storage.set('local.favorites', nextFavorites)
    }
    if (!sameStation(station.get(), item)) return true
    const before = station.get()
    const updated = apply(before)
    station.set(updated)
    ctx.storage.set('local.station', updated)
    if (updated.url !== before.url && (status.get() === 'live' || status.get() === 'connecting')) void play(updated)
    return true
  }

  function removeStation(item) {
    saveStations(stations.get().filter(value => !sameStation(value, item)))
    if (favorites.get().some(value => sameStation(value, item))) {
      const nextFavorites = favorites.get().filter(value => !sameStation(value, item))
      favorites.set(nextFavorites)
      ctx.storage.set('local.favorites', nextFavorites)
    }
    if (!sameStation(station.get(), item)) return
    const remaining = uniqueStations([...favorites.get(), ...stations.get()])
    if (remaining.length) void play(remaining[0])
    else stop()
  }

  function addStation({ name, url }) {
    const address = streamUrl(url)
    if (!address) return { station: null, added: false }
    const existing = uniqueStations([...stations.get(), ...favorites.get()]).find(value => value.url === address)
    if (existing) return { station: existing, added: false }
    const host = new URL(address).hostname
    const item = { id: `custom-${randomUuid()}`, name: tidyName(name) || host, url: address, provider: 'Custom station', description: host, homepage: null }
    saveStations([...stations.get(), item])
    return { station: item, added: true }
  }

  function startIfRequested() {
    if (!shouldAutoplay(ctx.storage, restored)) return false
    void play(restored)
    return true
  }

  bus.onmessage = event => {
    if (event.data?.type === 'play' && event.data.sender !== windowId) stop('elsewhere')
  }

  // Dictation completion: Fluid Voice delivers the transcript as a paste into
  // the focused editable field (its Electron delivery path is a guarded ⌘V,
  // a real DOM paste event). resume() no-ops unless the stream was paused
  // externally and is still intact, so ordinary pastes while stopped, errored
  // or intentionally paused (the plugin's own Pause is destructive) do nothing.
  const onTextInserted = event => {
    const target = event.target
    const editable = target instanceof HTMLTextAreaElement
      || target instanceof HTMLInputElement
      || (target instanceof HTMLElement && target.isContentEditable)
    if (editable) void resume()
  }
  document.addEventListener('paste', onTextInserted)
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
    document.removeEventListener('paste', onTextInserted)
    stop()
    bus.close()
    if (audioContext) void audioContext.close()
  })
  return { station, stations, favorites, volume, status, meterMode, lastError, open, play, stop, toggle, next, setVolume, mute, favorite, updateStation, removeStation, addStation, waveform, startIfRequested, restoreDefaults }
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

const MEDIA_ERRORS = { 1: 'aborted', 2: 'network error', 3: 'decode error', 4: 'unsupported stream format' }

function mediaFailure(element, station) {
  let host = station.url
  try { host = new URL(station.url).hostname } catch {}
  return `${MEDIA_ERRORS[element?.error?.code] ?? 'the stream did not connect'} (${host})`
}

async function checkStream(ctx, url) {
  if (typeof ctx.rest !== 'function') return null
  try {
    const verdict = await ctx.rest(`/probe?${new URLSearchParams({ url })}`, { timeoutMs: 15000 })
    return verdict && typeof verdict.ok === 'boolean' ? verdict : null
  } catch { return null }
}

// A failed stream leaves two trails: the panel's reason and one line in the
// backend's station log, so "why did this station not play" stays answerable.
function reportFailure(ctx, station, reason) {
  console.error(`[radio] ${station.name} <${station.url}>: ${reason}`)
  if (typeof ctx.rest !== 'function') return
  void ctx.rest('/log', {
    method: 'POST',
    body: { name: station.name, url: station.url, reason, at: new Date().toISOString() }
  }).catch(() => {})
  void checkStream(ctx, station.url).then(verdict => {
    if (!verdict?.kind) return
    console.error(`[radio] probe ${station.url}: ${verdict.kind} ${verdict.status} ${verdict.contentType} ${verdict.reason ?? ''}`)
  })
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

function liveTrack(announced, station, status) {
  if (status !== 'live') return ''
  return announced || station.nowPlaying || ''
}

function signalText(status, track, activityNote, lastError, t) {
  if (status === 'live') return track || (activityNote ? t('audioOnly') : '')
  if (status === 'error' && lastError) return lastError
  return t(status)
}

function Signal({ player, ctx }) {
  const t = usePluginI18n(ID)
  const station = useValue(player.station)
  const status = useValue(player.status)
  const mode = useValue(player.meterMode)
  const lastError = useValue(player.lastError)
  // The stream announces its track through ICY metadata, which the renderer
  // cannot read: most stations send no CORS headers, and the metadata is
  // interleaved into the audio. The plugin route reads it for every station,
  // so directory rows and addresses added by hand get the same track line.
  const metadata = useQuery({
    queryKey: [ID, 'now-playing', station.url],
    queryFn: () => nowPlaying(ctx, station.url),
    enabled: status === 'live' && typeof ctx.rest === 'function',
    staleTime: 15000, refetchInterval: status === 'live' ? 20000 : false, retry: 0
  })
  // A Shoutcast row already knows what was on air when it was fetched. Keep
  // that for the moment before the stream's first announcement.
  const announced = metadata.data?.title || ''
  const track = liveTrack(announced, station, status)
  const credit = trackCredit(track ? { title: track } : null)
  const artist = useQuery({
    queryKey: [ID, 'artist-profile', credit.artist],
    queryFn: ({ signal }) => resolveArtist(credit.artist, signal, ctx),
    enabled: Boolean(credit.artist), staleTime: 86400000, gcTime: 86400000,
    retry: 1, retryDelay: 3000, refetchOnWindowFocus: false
  })
  const artistUrl = credit.artist ? artist.data : null
  const text = signalText(status, track, mode === 'activity', lastError, t)
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
  const editable = useValue(player.stations).some(item => sameStation(item, station))
  const active = status === 'live' || status === 'connecting'
  const failed = current && status === 'error'
  const saved = useValue(player.favorites).some(item => sameStation(item, station))
  const [editing, setEditing] = useState(false)
  const [armed, setArmed] = useState(false)
  const [draftName, setDraftName] = useState(station.name)
  const [draftUrl, setDraftUrl] = useState(station.url)
  const [note, setNote] = useState('')
  useEffect(() => {
    if (!armed) return
    const timer = setTimeout(() => setArmed(false), 4000)
    return () => clearTimeout(timer)
  }, [armed])
  const open = () => { setDraftName(station.name); setDraftUrl(station.url); setNote(''); setArmed(false); setEditing(true) }
  const cancel = () => { setEditing(false); setNote('') }
  const commit = () => {
    if (!streamUrl(draftUrl) || !player.updateStation(station, { name: draftName, url: draftUrl })) { setNote(t('editInvalid')); return }
    setEditing(false)
    setNote('')
  }
  return jsxs('div', { className: 'hermes-radio-row', 'data-current': current, 'data-pinned': saved, 'data-armed': armed, 'data-editing': editing, children: [
    editing
      ? jsxs('div', { className: 'hermes-radio-edit', children: [
        jsx('input', { value: draftName, autoFocus: true, 'aria-label': t('editName'), placeholder: t('editName'), onChange: event => setDraftName(event.target.value),
          onKeyDown: event => { if (event.key === 'Enter') { event.preventDefault(); commit() } else if (event.key === 'Escape') { event.preventDefault(); cancel() } } }),
        jsx('input', { value: draftUrl, 'aria-label': t('editUrl'), placeholder: t('editUrl'), spellCheck: false, onChange: event => setDraftUrl(event.target.value),
          onKeyDown: event => { if (event.key === 'Enter') { event.preventDefault(); commit() } else if (event.key === 'Escape') { event.preventDefault(); cancel() } } }),
        jsxs('div', { className: 'hermes-radio-edit-actions', children: [
          jsx(Button, { variant: 'secondary', size: 'micro', onClick: commit, children: t('editSave') }),
          jsx(Button, { variant: 'ghost', size: 'micro', onClick: cancel, children: t('cancel') }),
          note && jsx('span', { className: 'hermes-radio-add-note', role: 'status', children: note })
        ] })
      ] })
      : jsxs(RowButton, { className: 'hermes-radio-row-main', 'aria-label': `${t(failed ? 'retry' : current && active ? 'pause' : 'play')}: ${station.name}`, title: failed ? t('streamError') : undefined, 'aria-current': current ? 'true' : undefined,
        onClick: () => current ? player.toggle() : void player.play(station), children: [
          jsx('span', { className: 'hermes-radio-row-icon', role: failed ? 'status' : undefined, 'aria-label': failed ? t('error') : undefined, children: jsx(failed ? icons.AlertCircle : current && active ? icons.Pause : icons.Play, { size: 12 }) }),
          jsx('span', { className: 'hermes-radio-row-copy', children: jsx('span', { className: 'hermes-radio-row-name', children: station.name }) })
        ] }),
    editable && jsx(SmallAction, { label: `${t('edit')}: ${station.name}`, icon: icons.Pencil, size: 'icon-xs', className: 'hermes-radio-row-edit', onClick: open }),
    editable && jsx(SmallAction, { label: `${t(armed ? 'confirmRemove' : 'remove')}: ${station.name}`, icon: icons.Trash2, size: 'icon-xs',
      className: `hermes-radio-row-edit${armed ? ' hermes-radio-row-remove' : ''}`,
      onClick: () => { if (armed) { setArmed(false); player.removeStation(station) } else setArmed(true) } }),
    jsx(SmallAction, { label: `${t(saved ? 'unsave' : 'save')}: ${station.name}`, icon: icons.Pin, onClick: () => player.favorite(station), size: 'icon-xs', pressed: saved, className: 'hermes-radio-pin' })
  ] })
}

async function shoutcastQuery(ctx, text, { offset = 0, limit = SHOUTCAST_PAGE } = {}) {
  const path = text
    ? `/shoutcast/search?${new URLSearchParams({ q: text, limit: String(limit), offset: String(offset) })}`
    : `/shoutcast/top?${new URLSearchParams({ limit: String(limit), offset: String(offset) })}`
  const payload = await ctx.rest(path, { timeoutMs: 30000 })
  const rows = Array.isArray(payload?.stations) ? payload.stations : []
  return {
    stations: uniqueStations(rows.flatMap(row => {
      const url = streamUrl(row?.url)
      if (!url || typeof row.id !== 'string' || typeof row.name !== 'string') return []
      return [{ id: row.id, name: row.name.trim(), url, provider: 'Shoutcast', description: row.description ?? '', nowPlaying: row.nowPlaying ?? '', homepage: httpsUrl(row.homepage) }]
    })),
    skipped: Number(payload?.skipped) || 0,
    hasMore: Number(payload?.next ?? 0) < Number(payload?.count ?? 0),
    nextOffset: Number(payload?.next ?? 0)
  }
}

async function shoutcastStations(ctx, text, options = {}) {
  if (typeof ctx.rest !== 'function') throw new Error('The Shoutcast route is not loaded')
  const parsed = parseQuery(text)
  const query = parsed.name || parsed.genre
  if (String(text ?? '').trim() && !query) return { stations: [], skipped: 0, hasMore: false, nextOffset: 0 }
  const results = await Promise.all(searchTerms(query).map(term => shoutcastQuery(ctx, term, options)))
  return {
    stations: uniqueStations(results.flatMap(result => result.stations)),
    skipped: results.reduce((total, result) => total + result.skipped, 0),
    hasMore: results.some(result => result.hasMore),
    nextOffset: Math.max(...results.map(result => result.nextOffset))
  }
}

async function tuneinQuery(ctx, text, { offset = 0, limit = TUNEIN_PAGE } = {}) {
  const payload = await ctx.rest(`/tunein/search?${new URLSearchParams({ q: text, limit: String(limit), offset: String(offset) })}`, { timeoutMs: 30000 })
  const rows = Array.isArray(payload?.stations) ? payload.stations : []
  return {
    stations: uniqueStations(rows.flatMap(row => {
      const url = streamUrl(row?.url)
      if (!url || typeof row.id !== 'string' || typeof row.name !== 'string') return []
      return [{ id: row.id, name: row.name.trim(), url, provider: 'TuneIn', description: row.description ?? '', nowPlaying: row.nowPlaying ?? '', homepage: '' }]
    })),
    skipped: Number(payload?.skipped) || 0,
    hasMore: Number(payload?.next ?? 0) < Number(payload?.count ?? 0),
    nextOffset: Number(payload?.next ?? 0)
  }
}

async function tuneinStations(ctx, text, options = {}) {
  if (typeof ctx.rest !== 'function') throw new Error('The TuneIn route is not loaded')
  const parsed = parseQuery(text)
  const query = parsed.name || parsed.genre
  if (!query) return { stations: [], skipped: 0, hasMore: false, nextOffset: 0 }
  const results = await Promise.all(searchTerms(query).map(term => tuneinQuery(ctx, term, options)))
  return {
    stations: uniqueStations(results.flatMap(result => result.stations)),
    skipped: results.reduce((total, result) => total + result.skipped, 0),
    hasMore: results.some(result => result.hasMore),
    nextOffset: Math.max(...results.map(result => result.nextOffset))
  }
}

const SOURCE_LABELS = { all: 'sourceAll', popular: 'sourcePopular', shoutcast: 'sourceShoutcast', tunein: 'sourceTunein' }
const SOURCE_PAGE = { all: PAGE_SIZE, popular: PAGE_SIZE, shoutcast: SHOUTCAST_PAGE, tunein: TUNEIN_PAGE }
const EMPTY_LABELS = { shoutcast: 'shoutcastEmpty', tunein: 'tuneinEmpty' }

function listRows(results, { browseStation, current, ownList }) {
  const rows = Array.isArray(results) ? results : []
  const anchored = ownList && browseStation && !rows.some(item => sameStation(item, browseStation))
    ? [browseStation, ...rows]
    : rows
  return ownList && current && !anchored.some(item => sameStation(item, current))
    ? [current, ...anchored]
    : anchored
}

function Stations({ player, ctx }) {
  const t = usePluginI18n(ID)
  const favorites = useValue(player.favorites)
  const stored = useValue(player.stations)
  const current = useValue(player.station)
  const [browseStation, setBrowseStation] = useState(() => player.station.get())
  const listRef = useRef(null)
  const [search, setSearch] = useState('')
  const [query, setQuery] = useState('')
  const [source, setSource] = useState('all')
  const [more, setMore] = useState([])
  const [moreState, setMoreState] = useState('idle')
  const moreOffset = useRef(0)
  const moreRows = useRef([])
  const moreBusy = useRef(false)
  const queryRef = useRef('')
  const [adding, setAdding] = useState(false)
  const [draftName, setDraftName] = useState('')
  const [draftUrl, setDraftUrl] = useState('')
  const [addNote, setAddNote] = useState('')
  const text = search.trim()
  useEffect(() => { if (listRef.current) listRef.current.scrollTop = 0 }, [text, source])
  useEffect(() => {
    const timer = setTimeout(() => setQuery(text), 350)
    return () => clearTimeout(timer)
  }, [text])
  useEffect(() => {
    queryRef.current = query
    setMore([])
    moreRows.current = []
    setMoreState('idle')
    moreOffset.current = 0
  }, [query, source])
  const browser = useQuery({
    queryKey: [ID, 'search', query], queryFn: ({ signal }) => searchStations(query, signal),
    enabled: source === 'all' && query.length >= 2 && text === query, staleTime: 300000, retry: false
  })
  const popular = useQuery({
    queryKey: [ID, 'popular'], queryFn: ({ signal }) => popularStations(signal),
    enabled: source === 'popular', staleTime: 600000, retry: false
  })
  const shoutcast = useQuery({
    queryKey: [ID, 'shoutcast', query], queryFn: () => shoutcastStations(ctx, query),
    enabled: source === 'shoutcast' && text === query, staleTime: 300000, retry: false
  })
  const tunein = useQuery({
    queryKey: [ID, 'tunein', query], queryFn: () => tuneinStations(ctx, query),
    enabled: source === 'tunein' && text === query, staleTime: 300000, retry: false
  })
  const sourceQuery = { all: browser, popular, shoutcast, tunein }
  const fetchPage = {
    all: (token, offset) => searchStations(token, new AbortController().signal, { offset }),
    popular: (token, offset) => popularStations(new AbortController().signal, { offset }),
    shoutcast: (token, offset) => shoutcastStations(ctx, token, { offset }),
    tunein: (token, offset) => tuneinStations(ctx, token, { offset })
  }
  const parsedText = parseQuery(text)
  const filterText = parsedText.name
  const tagQuery = Boolean(parsedText.country || parsedText.genre)
  const matches = item => !filterText || [item.name, item.description, item.provider].some(value => fold(value).includes(fold(filterText)))
  const active = sourceQuery[source]
  const canPage = (source !== 'all' || text.length >= 2) && Boolean(active.data?.hasMore) && moreState !== 'done'
  const loadMore = async () => {
    if (!canPage || moreBusy.current || query !== queryRef.current) return
    moreBusy.current = true
    const token = query
    const offset = moreOffset.current || SOURCE_PAGE[source]
    setMoreState('loading')
    try {
      const next = await fetchPage[source](token, offset)
      if (queryRef.current !== token) return
      moreOffset.current = next.nextOffset || offset + PAGE_SIZE
      const merged = uniqueStations([...moreRows.current, ...next.stations])
      const added = merged.length > moreRows.current.length
      moreRows.current = merged
      setMore(merged)
      setMoreState(added && next.hasMore ? 'idle' : 'done')
    } catch {
      if (queryRef.current === token) setMoreState('idle')
    } finally {
      moreBusy.current = false
    }
  }
  const local = source === 'all' && !tagQuery ? uniqueStations([...favorites, ...stored]).filter(matches) : []
  const remoteResult = source === 'all'
    ? (text.length >= 2 && query === text ? browser.data : null)
    : source === 'popular'
      ? popular.data
      : (text === query ? active.data : null)
  const remote = [...(remoteResult?.stations ?? []), ...more]
    .filter(item => source !== 'popular' || !text || matches(item))
  const results = uniqueStations([...local, ...remote])
  const items = listRows(results, { browseStation, current, ownList: source === 'all' })
  const pending = source === 'all'
    ? text.length >= 2 && (text !== query || browser.isFetching)
    : active.isFetching || (source !== 'popular' && text !== query)
  const failed = active.isError && (source !== 'all' || text.length >= 2)
  const errorText = t({ popular: 'popularError', shoutcast: 'shoutcastError', tunein: 'tuneinError' }[source] ?? 'searchError')
  const empty = !pending && !failed && results.length === 0 && (text.length > 0 || source !== 'all')
  const hls = remoteResult?.hls ?? 0
  const skipped = remoteResult?.skipped ?? 0
  const openAdd = () => { setAddNote(''); setDraftName(''); setDraftUrl(''); setAdding(true) }
  const closeAdd = () => { setAdding(false); setDraftName(''); setDraftUrl(''); setAddNote('') }
  const save = () => {
    const outcome = player.addStation({ name: draftName, url: draftUrl })
    if (!outcome.station) { setAddNote(t('addInvalid')); return }
    if (!outcome.added) { setAddNote(t('addExists')); return }
    closeAdd()
  }
  return jsxs('div', { children: [
    jsx('div', { className: 'hermes-radio-search', 'data-filled': Boolean(search), children: jsx(SearchField, {
      value: search, onChange: value => { setBrowseStation(player.station.get()); setSearch(value) }, placeholder: t('search'),
      containerClassName: 'hermes-radio-search-field',
      trailingAction: pending ? jsx('span', { className: 'hermes-radio-search-loader', children: jsx(GlyphSpinner, { ariaLabel: t('searching') }) }) : null
    }) }),
    jsxs('div', { className: 'hermes-radio-actions', children: [
      Object.entries(SOURCE_LABELS).map(([value, label]) => jsx('button', {
        type: 'button', className: 'hermes-radio-chip', 'data-active': source === value, 'aria-pressed': source === value,
        onClick: () => { setBrowseStation(player.station.get()); setSource(value) }, children: t(label)
      }, value)),
      jsx('span', { className: 'hermes-radio-sources-space' }),
      jsx(Tip, { label: t('add'), children: jsx('button', { type: 'button', className: 'hermes-radio-addbtn', 'data-active': adding, 'aria-label': t('add'), 'aria-expanded': adding,
        onClick: () => adding ? closeAdd() : openAdd(), children: jsx(icons.Plus, { size: 12 }) }) })
    ] }),
    adding && jsxs('div', { className: 'hermes-radio-add', children: [
      jsx('input', { value: draftName, autoFocus: true, placeholder: t('addName'), 'aria-label': t('addName'), onChange: event => setDraftName(event.target.value) }),
      jsx('input', { value: draftUrl, placeholder: t('addUrl'), 'aria-label': t('addUrl'), spellCheck: false, onChange: event => setDraftUrl(event.target.value),
        onKeyDown: event => { if (event.key === 'Enter') { event.preventDefault(); save() } else if (event.key === 'Escape') { event.preventDefault(); closeAdd() } } }),
      jsxs('div', { className: 'hermes-radio-add-actions', children: [
        jsx(Button, { variant: 'secondary', size: 'micro', onClick: save, children: t('addSave') }),
        jsx(Button, { variant: 'ghost', size: 'micro', onClick: closeAdd, children: t('cancel') }),
        addNote && jsx('span', { className: 'hermes-radio-add-note', role: 'status', children: addNote })
      ] })
    ] }),
    jsxs('div', {
      ref: listRef, className: 'hermes-radio-list', 'data-radio-pending': pending,
      onScroll: event => {
        const list = event.currentTarget
        if (list.scrollHeight - list.scrollTop - list.clientHeight < 160) void loadMore()
      },
      children: [
        ...items.map(station => jsx(StationRow, { station, player }, station.id)),
        moreState === 'loading' && jsx('div', { className: 'hermes-radio-more', role: 'status', children: t('loadingMore') }),
        pending && !items.length && jsx('div', { className: 'hermes-radio-error', role: 'status', children: t('searching') }),
        failed && jsx('div', { className: 'hermes-radio-error', role: 'status', children: errorText }),
        empty && !failed && jsx('div', { className: 'hermes-radio-error', role: 'status', children: source === 'tunein' && !text ? t('tuneinHint') : t(EMPTY_LABELS[source] ?? 'noResults') }),
        hls > 0 && jsx('div', { className: 'hermes-radio-error', children: `${hls} ${t('hlsNote')}` }),
        skipped > 0 && jsx('div', { className: 'hermes-radio-error', children: `${skipped} ${t('skippedNote')}` })
      ]
    })
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
          jsx(Stations, { player, ctx }), jsx(Signal, { player, ctx }), jsx(Transport, { player, ctx })
        ] })
      })
    ] }),
    jsx(SmallAction, { label: t(active ? 'pause' : 'play'), icon: active ? icons.Pause : icons.Play, busy: status === 'connecting', onClick: player.toggle }),
    jsx(SmallAction, { label: t('next'), icon: NextArrow, onClick: player.next })
  ] })
}

function RadioSettingsPage({ ctx, player }) {
  const t = usePluginI18n(ID)
  const [on, setOn] = useState(autoplayOnLaunch(ctx.storage))
  const [restored, setRestored] = useState(null)
  const restore = () => setRestored(typeof player?.restoreDefaults === 'function' ? player.restoreDefaults() : 0)
  return jsxs('div', { children: [
    jsxs('div', { className: 'hermes-radio-setting', children: [
      jsxs('div', { className: 'hermes-radio-setting-text', children: [
        jsx('span', { className: 'hermes-radio-setting-label', children: t('autoplayLabel') }),
        jsx('span', { className: 'hermes-radio-setting-note', children: t('autoplayDescription') })
      ] }),
      jsx('input', { type: 'checkbox', checked: on, 'aria-label': t('autoplayLabel'), onChange: event => { setOn(event.target.checked); ctx.storage.set(AUTOPLAY_KEY, event.target.checked) } })
    ] }),
    jsxs('div', { className: 'hermes-radio-setting', children: [
      jsxs('div', { className: 'hermes-radio-setting-text', children: [
        jsx('span', { className: 'hermes-radio-setting-label', children: t('restoreLabel') }),
        jsx('span', { className: 'hermes-radio-setting-note', children: t('restoreDescription') })
      ] }),
      jsx(Button, { variant: 'secondary', size: 'micro', onClick: restore, children: t('restoreAction') })
    ] }),
    restored !== null && jsx('div', { className: 'hermes-radio-setting-note', role: 'status', children: restored ? `${t('restoreSome')} ${restored}` : t('restoreNone') })
  ] })
}

export {
  artistProfile, resolveArtist, trackCredit, autoplayOnLaunch, shouldAutoplay, RadioSettingsPage,
  nowPlaying, mediaFailure, checkStream, reportFailure, streamUrl, validStation, PRESETS, createPlayer, parseQuery, searchParams, searchStations, popularStations,
  rowsToStations, transliterate, fold, searchTerms, matchScore, listRows, isHlsRow,
  shoutcastStations, tuneinStations, Stations
}

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
    if (typeof ctx.registerSettingsPage === 'function') {
      ctx.registerSettingsPage({ id: ID, title: 'Radio', icon: 'radio-tower', order: 70, render: () => jsx(RadioSettingsPage, { ctx, player }) })
    }
    if (autoplayOnLaunch(ctx.storage)) setTimeout(() => player.startIfRequested(), 1200)
  }
}
