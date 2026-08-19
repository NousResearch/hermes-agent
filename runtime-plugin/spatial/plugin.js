/** Spatial — power-user command center (Hermes projects + Paperclip pulse). Plain ESM.
 *  Theme: inherits Hermes tokens only (--background/--dt-*, --card, --foreground, --border, --primary/--ui-accent).
 *  Follows :root.dark / .dark — no independent skin or theme toggle.
 */
import {
  Badge, cn, Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle,
  host, icons, KEYBINDS_AREA, PALETTE_AREA, ROUTES_AREA, SIDEBAR_NAV_AREA
} from '@hermes/plugin-sdk'
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { jsx as _j, jsxs as _js } from 'react/jsx-runtime'

const ID = 'spatial'
const PC = 'http://127.0.0.1:3100'
const PREF = 'spatial.cmd.v1'
const clamp = (n, a, b) => Math.min(b, Math.max(a, n))
const LS = {
  get() { try { return JSON.parse(localStorage.getItem(PREF) || '{}') } catch { return {} } },
  set(v) { try { localStorage.setItem(PREF, JSON.stringify(v)) } catch {} }
}

function togglePin(projectId, path) {
  const pref = LS.get()
  const pins = { ...(pref.pins || {}) }
  const list = Array.isArray(pins[projectId]) ? pins[projectId].slice() : []
  const i = list.indexOf(path)
  if (i >= 0) list.splice(i, 1)
  else if (path) list.push(path)
  pins[projectId] = list
  pref.pins = pins
  LS.set(pref)
  return pref
}


const CSS = [
'.sp{--bg:var(--background,var(--dt-background,#e9e9eb));--paper:var(--midground,var(--dt-midground,color-mix(in srgb,var(--background,#e9e9eb) 92%,var(--foreground,#1c1c1e))));--card:var(--card,var(--dt-card,#fff));--ink:var(--foreground,var(--dt-foreground,#1c1c1e));--mut:var(--muted-foreground,var(--dt-muted-foreground,#6c6c70));--line:var(--border,var(--dt-border,rgba(0,0,0,.08)));--acc:var(--primary,var(--dt-primary,var(--ui-accent,#0a84ff)));--sh:0 .5px .5px rgba(0,0,0,.04),0 2px 6px rgba(0,0,0,.045),0 12px 28px rgba(0,0,0,.08),0 32px 64px rgba(0,0,0,.07);--shh:0 1px 2px rgba(0,0,0,.05),0 14px 32px rgba(0,0,0,.12),0 40px 80px rgba(0,0,0,.14);--spr:cubic-bezier(.34,1.45,.64,1);--ease:cubic-bezier(.22,1,.36,1);position:absolute;inset:0;display:flex;flex-direction:column;overflow:hidden;color:var(--ink);background:var(--bg);font:12.5px/1.35 ui-sans-serif,system-ui,-apple-system,sans-serif;-webkit-font-smoothing:antialiased;min-height:100%;height:100%;width:100%}',
':root.dark .sp,.dark .sp{--sh:0 1px 2px color-mix(in srgb,var(--ink) 25%,transparent),0 16px 36px color-mix(in srgb,#000 55%,transparent);--shh:0 4px 16px color-mix(in srgb,#000 50%,transparent),0 28px 64px color-mix(in srgb,#000 55%,transparent)}',
'.sp-st{position:relative;flex:1 1 auto;min-height:0;height:100%;overflow:hidden;cursor:grab;touch-action:none;background:radial-gradient(80% 60% at 50% 38%,color-mix(in srgb,var(--paper) 88%,transparent) 0%,transparent 55%),radial-gradient(100% 90% at 50% 100%,color-mix(in srgb,var(--ink) 6%,transparent),transparent 45%),radial-gradient(38% 32% at 16% 74%,color-mix(in srgb,var(--acc) 8%,transparent),transparent 55%),radial-gradient(34% 28% at 84% 18%,color-mix(in srgb,var(--acc) 6%,transparent),transparent 50%),var(--bg)}',
'.sp-st.p{cursor:grabbing;user-select:none}.sp-st.p .sp-i{transition:none!important;animation:none!important}',
'.sp-w{position:absolute;left:0;top:0;transform-origin:0 0;will-change:transform}','.sp-w::after{content:"";position:absolute;left:5%;right:5%;bottom:8%;height:42%;pointer-events:none;background:radial-gradient(60% 55% at 50% 100%,color-mix(in srgb,var(--ink) 6%,transparent),transparent 70%);z-index:0}',
'.sp-i{position:absolute;transform-origin:center;transform:rotate(var(--r,0deg));animation:sp-in .45s var(--spr) both;animation-fill-mode:both;transition:transform .32s var(--spr),filter .2s var(--ease);contain:layout paint;filter:drop-shadow(0 12px 22px rgba(0,0,0,.11))}',
'.sp-i:hover{z-index:40!important;filter:drop-shadow(0 22px 40px rgba(0,0,0,.18));transform:translateY(-14px) scale(1.05) rotate(var(--r,0deg))!important}',
'.sp-i.on{z-index:50!important}@keyframes sp-in{from{opacity:.01;transform:translateY(14px) scale(.94) rotate(var(--r,0deg))}to{opacity:1;transform:translateY(0) scale(1) rotate(var(--r,0deg))}}',
'.sp-p,.sp-c,.sp-n,.sp-x{width:100%;height:100%;border:0;border-radius:20px;text-align:left;cursor:pointer;color:inherit;position:relative;overflow:hidden}',
'.sp-p,.sp-c{background:linear-gradient(180deg,color-mix(in srgb,var(--card) 100%,transparent) 0%,color-mix(in srgb,var(--card) 94%,var(--ink)) 100%);border:1px solid var(--line);box-shadow:var(--sh);padding:0;display:flex;flex-direction:column}.sp-p{box-shadow:var(--sh),inset 0 1px 0 color-mix(in srgb,var(--card) 80%,#fff),inset 0 -1px 0 color-mix(in srgb,var(--ink) 6%,transparent);background-image:linear-gradient(180deg,color-mix(in srgb,var(--card) 96%,transparent),color-mix(in srgb,var(--paper) 70%,var(--card))),repeating-linear-gradient(0deg,transparent,transparent 2px,color-mix(in srgb,var(--ink) 3%,transparent) 2px,color-mix(in srgb,var(--ink) 3%,transparent) 3px)}',
'.sp-p::before,.sp-c::before{content:"";position:absolute;inset:0;border-radius:inherit;pointer-events:none;background:linear-gradient(180deg,rgba(255,255,255,.78),transparent 32%),radial-gradient(120% 80% at 50% 0%,rgba(255,255,255,.35),transparent 50%);z-index:1}',
':root.dark .sp .sp-p::before,:root.dark .sp .sp-c::before,.dark .sp .sp-p::before,.dark .sp .sp-c::before{background:linear-gradient(180deg,color-mix(in srgb,var(--ink) 10%,transparent),transparent 30%)}',
'.sp-i:hover .sp-p,.sp-i:hover .sp-c,.sp-i:hover .sp-n,.sp-i:hover .sp-x{box-shadow:var(--shh)}',
'.sp-p__h{height:48%;min-height:76px;background:linear-gradient(168deg,color-mix(in srgb,var(--a,var(--acc)) 38%,var(--card)) 0%,color-mix(in srgb,var(--a,var(--acc)) 12%,var(--paper)) 40%,var(--paper) 100%),repeating-linear-gradient(-11deg,transparent,transparent 7px,color-mix(in srgb,var(--ink) 4%,transparent) 7px,color-mix(in srgb,var(--ink) 4%,transparent) 8px),radial-gradient(90% 70% at 18% 0%,color-mix(in srgb,var(--card) 70%,transparent),transparent 52%);border-bottom:1px solid var(--line)}',
'.sp-p__b{position:relative;z-index:2;padding:12px 14px;display:flex;flex-direction:column;gap:6px;flex:1;min-height:0}','.sp-p__rules{display:flex;flex-direction:column;gap:7px;margin-top:4px}','.sp-p__rule{height:5px;border-radius:3px;background:color-mix(in srgb,var(--mut) 16%,transparent)}','.sp-p__rule:nth-child(2){width:92%}','.sp-p__rule:nth-child(3){width:78%}','.sp-p__rule:nth-child(4){width:84%}',
'.sp-k{font:700 9px/1 ui-sans-serif,system-ui,sans-serif;letter-spacing:.12em;text-transform:uppercase;color:var(--a,var(--acc))}',
'.sp-t{font-size:13.5px;font-weight:650;letter-spacing:-.02em;line-height:1.2}',
'.sp-s{font-size:11px;color:var(--mut);display:-webkit-box;-webkit-line-clamp:3;-webkit-box-orient:vertical;overflow:hidden}',
'.sp-m{display:flex;flex-wrap:wrap;gap:4px;margin-top:auto;padding-top:4px}',
'.sp-ch{padding:2px 7px;border-radius:999px;border:1px solid var(--line);font:500 8px/1.2 ui-monospace,Menlo,monospace;color:var(--mut);text-transform:uppercase;letter-spacing:.03em}',
'.sp-n{padding:14px;display:flex;flex-direction:column;justify-content:space-between;color:var(--ink);background:linear-gradient(155deg,color-mix(in srgb,var(--a,#ffd60a) 88%,var(--card)),color-mix(in srgb,var(--a,#ffd60a) 70%,var(--paper)));box-shadow:0 1px 1px color-mix(in srgb,var(--ink) 8%,transparent),0 8px 18px color-mix(in srgb,var(--a,#ffd60a) 35%,transparent),0 22px 44px color-mix(in srgb,var(--ink) 12%,transparent),inset 0 1px 0 color-mix(in srgb,var(--card) 70%,transparent);border-radius:13px}',
':root.dark .sp .sp-n,.dark .sp .sp-n{color:var(--ink);background:color-mix(in srgb,var(--a,#ffd60a) 42%,var(--card))}','.sp-n::before{content:"";position:absolute;inset:0;border-radius:inherit;pointer-events:none;background:linear-gradient(160deg,rgba(255,255,255,.22),transparent 42%),repeating-linear-gradient(-8deg,transparent,transparent 11px,rgba(0,0,0,.03) 11px,rgba(0,0,0,.03) 12px);mix-blend-mode:soft-light}','.sp-n::after{content:"";position:absolute;top:-3px;right:18px;width:28px;height:10px;border-radius:2px;background:linear-gradient(180deg,rgba(255,255,255,.35),rgba(255,255,255,.12));box-shadow:0 1px 2px rgba(0,0,0,.08);transform:rotate(2deg)}',
'.sp-x{background:#0e0e10;box-shadow:var(--sh),inset 0 0 0 1px rgba(255,255,255,.12);border-radius:20px}',
'.sp-x__a{position:absolute;inset:0;background:radial-gradient(95% 85% at 20% 15%,color-mix(in srgb,var(--a,#0a84ff) 80%,#fff),transparent 50%),radial-gradient(85% 75% at 85% 80%,color-mix(in srgb,var(--b,#bf5af2) 60%,transparent),transparent 46%),radial-gradient(circle at 72% 28%,rgba(255,255,255,.18),transparent 26%),linear-gradient(155deg,#22222a,#0b0b0e 55%,#14141a);filter:saturate(1.08) contrast(1.05)}','.sp-x__a[data-pat=grid]{background:linear-gradient(rgba(255,255,255,.07) 1px,transparent 1px) 0 0/24px 24px,linear-gradient(90deg,rgba(255,255,255,.07) 1px,transparent 1px) 0 0/24px 24px,radial-gradient(65% 55% at 68% 28%,color-mix(in srgb,var(--a) 60%,transparent),transparent 58%),#0f0f14}','.sp-x__a[data-pat=soft]{background:radial-gradient(75% 65% at 42% 38%,color-mix(in srgb,var(--a) 48%,var(--card)),transparent 58%),linear-gradient(185deg,var(--card),color-mix(in srgb,var(--paper) 80%,var(--ink)));filter:none}','.sp-x__a[data-pat=split]{background:linear-gradient(108deg,color-mix(in srgb,var(--a) 88%,#0a0a0c) 0 40%,#0c0c10 40% 100%)}','.sp-x__a[data-pat=orb]{background:radial-gradient(circle at 50% 55%,color-mix(in srgb,var(--a) 90%,#fff) 0%,transparent 26%),radial-gradient(circle at 50% 55%,transparent 30%,#0a0a0c 32%),radial-gradient(80% 60% at 20% 20%,color-mix(in srgb,var(--b) 45%,transparent),transparent 50%),#0c0c10}',
'.sp-x__c{position:absolute;left:0;right:0;bottom:0;z-index:1;padding:24px 12px 12px;background:linear-gradient(180deg,transparent,rgba(0,0,0,.72));color:#f5f5f7;font-weight:650;font-size:12px}',
'.sp-tb{position:absolute;bottom:22px;left:50%;z-index:60;transform:translateX(-50%);display:flex;gap:2px;align-items:center;padding:6px 10px;border-radius:999px;border:1px solid rgba(255,255,255,.12);background:rgba(18,18,20,.82);color:#f5f5f7;box-shadow:0 12px 40px rgba(0,0,0,.36),inset 0 1px 0 rgba(255,255,255,.1);backdrop-filter:blur(22px) saturate(160%);-webkit-backdrop-filter:blur(22px) saturate(160%)}',
'.sp-tb input{width:min(200px,34vw);padding:7px 12px;border:0;border-radius:999px;background:rgba(255,255,255,.1);color:#f5f5f7;font:inherit;font-size:12px;outline:0}',
'.sp-tb input::placeholder{color:rgba(245,245,247,.4)}',
'.sp-tb button{width:32px;height:32px;border:0;border-radius:999px;background:transparent;color:rgba(245,245,247,.72);cursor:pointer;display:grid;place-items:center;transition:transform .18s var(--spr),background .15s}',
'.sp-tb button:hover{background:rgba(255,255,255,.12);color:#fff;transform:scale(1.08)}',
'.sp-tb .pct{min-width:2.6rem;text-align:center;font:500 10px ui-monospace,Menlo,monospace;opacity:.55}',
'.sp-top{position:absolute;top:12px;left:50%;z-index:50;transform:translateX(-50%);display:flex;gap:8px;align-items:center;padding:6px 12px;border-radius:999px;border:1px solid var(--line);background:color-mix(in srgb,var(--card) 86%,transparent);box-shadow:var(--sh);backdrop-filter:blur(12px);pointer-events:none;font-size:11px;color:var(--mut)}',
'.sp-top b{color:var(--ink);font-weight:650}',
'.sp-cr{position:absolute;z-index:55;display:flex;gap:8px}.sp-cr.l{left:18px;bottom:22px}.sp-cr.r{right:18px;bottom:22px}',
'.sp-cr button{width:38px;height:38px;border-radius:999px;border:1px solid rgba(255,255,255,.1);background:rgba(22,22,24,.72);color:rgba(245,245,247,.75);box-shadow:0 10px 28px rgba(0,0,0,.28);backdrop-filter:blur(14px);cursor:pointer;display:grid;place-items:center;transition:transform .18s var(--spr)}',
'.sp-cr button:hover{transform:scale(1.07);color:#fff}',
'@media (prefers-reduced-motion:reduce){.sp-i{animation:none!important;transition:none!important}.sp-i:hover{transform:rotate(var(--r,0deg))!important;filter:none!important}}'
].join('\n')

function css() {
  if (typeof document === 'undefined') return
  let el = document.getElementById('sp-css')
  if (!el) {
    el = document.createElement('style')
    el.id = 'sp-css'
    document.head.appendChild(el)
  }
  if (el.textContent !== CSS) el.textContent = CSS
}

function usePanZoom() {
  const [t, setT] = useState({ s: 0.86, x: 10, y: 10 })
  const drag = useRef(null)
  const [pan, setPan] = useState(0)
  const zoomAt = useCallback((f, cx = 0, cy = 0) => {
    setT(p => {
      const s = clamp(p.s * f, 0.3, 2.8)
      const k = s / p.s
      return { s, x: cx - k * (cx - p.x), y: cy - k * (cy - p.y) }
    })
  }, [])
  const onWheel = useCallback(e => {
    e.preventDefault(); e.stopPropagation()
    const r = e.currentTarget.getBoundingClientRect()
    zoomAt(e.deltaY < 0 ? 1.08 : 1 / 1.08, e.clientX - r.left, e.clientY - r.top)
  }, [zoomAt])
  const onDown = useCallback(e => {
    if (e.button !== 0) return
    e.currentTarget.setPointerCapture(e.pointerId)
    setT(p => { drag.current = { x: e.clientX - p.x, y: e.clientY - p.y }; return p })
    setPan(1)
  }, [])
  const onMove = useCallback(e => {
    if (!drag.current) return
    const d = drag.current
    setT(p => ({ ...p, x: e.clientX - d.x, y: e.clientY - d.y }))
  }, [])
  const end = useCallback(() => { drag.current = null; setPan(0) }, [])
  return {
    pan, s: t.s,
    reset: () => setT({ s: 0.86, x: 10, y: 10 }),
    zin: () => zoomAt(1.18), zout: () => zoomAt(1 / 1.18),
    stage: { onWheel, onPointerDown: onDown, onPointerMove: onMove, onPointerUp: end, onPointerCancel: end, onPointerLeave: end },
    world: { transform: 'translate(' + t.x + 'px,' + t.y + 'px) scale(' + t.s + ')' }
  }
}

const hn = (s, m) => { let h = 0; const t = String(s || ''); for (let i = 0; i < t.length; i++) h = (h * 31 + t.charCodeAt(i)) | 0; return Math.abs(h) % m }
const rot = id => (hn(id, 90) - 45) / 9
const ST = ['#ffd60a', '#30d158', '#64d2ff', '#ff9f0a', '#bf5af2', '#ff375f']
const AC = ['#0a84ff', '#30d158', '#bf5af2', '#ff9f0a', '#64d2ff', '#ff375f']
const PATS = ['grad', 'grid', 'soft', 'split', 'orb', 'grad']

function slots(n) {
  // Banded freeform: readable ops hierarchy with organic jitter (not a dump, not a rigid grid).
  const bands = [
    { y: 24, x0: 16, cols: 6, dx: 250, dy: 0, n: 6 },   // projects
    { y: 230, x0: 40, cols: 5, dx: 220, dy: 0, n: 8 },  // hot stickies
    { y: 430, x0: 80, cols: 5, dx: 240, dy: 0, n: 6 },  // doctrine / secondary
    { y: 640, x0: 48, cols: 6, dx: 255, dy: 0, n: 8 }   // media arc
  ]
  const out = []
  let i = 0
  for (const b of bands) {
    for (let c = 0; c < b.n && i < n; c++, i++) {
      const jx = hn('x' + i, 56) - 28 + (c % 2) * 22
      const jy = hn('y' + i, 48) - 24 + (c % 3) * 12
      out.push({ x: b.x0 + c * b.dx + jx, y: b.y + jy })
    }
  }
  while (out.length < n) {
    const k = out.length
    out.push({ x: 40 + (k % 6) * 240, y: 40 + ((k / 6) | 0) * 180 })
  }
  return out
}

async function pc(path) {
  try {
    const r = await fetch(PC + path, { headers: { Accept: 'application/json' } })
    if (!r.ok) return null
    return await r.json()
  } catch { return null }
}

async function hermesProjects() {
  try {
    const p = await host.request('projects.list')
    return (p && p.projects) || []
  } catch { return [] }
}












function layoutBands(items) {
  const projects = items.filter(it => it.k === 'paper').slice(0, 6)
  const stickies = items.filter(it => it.k === 'sticky').slice(0, 6)
  const media = items.filter(it => it.k === 'media').slice(0, 5)
  const mid = stickies.slice(0, 7)

  const placeBand = (arr, y0, x0, x1) => {
    const n = arr.length
    if (!n) return
    // freeform seed positions (organic, not grid)
    arr.forEach((it, i) => {
      const t = n === 1 ? 0.5 : i / (n - 1)
      // non-linear x for pinboard feel
      const ease = t * t * (3 - 2 * t)
      const baseX = x0 + ease * (x1 - x0)
      const jx = hn('fx' + it.id, 70) - 35 + Math.sin(i * 2.1) * 36
      const jy = hn('fy' + it.id, 55) - 27 + Math.cos(i * 1.6) * 20
      it.x = baseX + jx
      it.y = y0 + jy
      it.r = (hn(it.id, 110) - 55) / 6 // stronger freeform tilt ~±9deg
      it.priority = 34 - i
      if (it.k === 'paper') {
        it.w = 198 + hn(it.id + 'w', 34)
        it.h = 176 + hn(it.id + 'h', 36)
      } else if (it.k === 'media') {
        it.w = 170 + hn(it.id + 'mw', 36)
        it.h = 202 + hn(it.id + 'mh', 40)
      } else {
        it.w = 150 + hn(it.id + 'sw', 20)
        it.h = 142 + hn(it.id + 'sh', 18)
      }
    })
    // soft separation: keep freeform but ensure faces legible (no deep bury)
    for (let pass = 0; pass < 4; pass++) {
      for (let i = 0; i < n; i++) {
        for (let j = i + 1; j < n; j++) {
          const a = arr[i], b = arr[j]
          const dx = (b.x + b.w / 2) - (a.x + a.w / 2)
          const dy = (b.y + b.h / 2) - (a.y + a.h / 2)
          const gapX = (a.w + b.w) * 0.42
          const gapY = (a.h + b.h) * 0.38
          const ox = gapX - Math.abs(dx)
          const oy = gapY - Math.abs(dy)
          if (ox > 0 && oy > 0) {
            const push = 0.35
            const sx = (dx === 0 ? 1 : Math.sign(dx)) * ox * push
            const sy = (dy === 0 ? 1 : Math.sign(dy)) * oy * push * 0.55
            a.x -= sx / 2
            b.x += sx / 2
            a.y -= sy / 2
            b.y += sy / 2
            // clamp Y to band corridor so hierarchy strata survive
            const lo = y0 - 36, hi = y0 + 48
            a.y = Math.max(lo, Math.min(hi, a.y))
            b.y = Math.max(lo, Math.min(hi, b.y))
          }
        }
      }
    }
  }

  placeBand(projects, 48, 30, 1540)
  placeBand(mid, 300, 50, 1520)
  placeBand(media, 560, 40, 1530)
  return projects.concat(mid, media)
}











function stop(e) { e.stopPropagation() }

function Card({ it, on }) {
  if (it.k === 'sticky') {
    return _js('button', { type: 'button', className: 'sp-n', style: { '--a': it.a }, onClick: () => on(it), onPointerDown: stop, children: [
      _j('div', { className: 'sp-t', children: it.t }),
      it.s && _j('div', { className: 'sp-s', children: it.s }),
      _j('div', { className: 'sp-k', style: { opacity: 0.55 }, children: it.tag || 'pulse' })
    ]})
  }
  if (it.k === 'media') {
    return _js('button', { type: 'button', className: 'sp-x', onClick: () => on(it), onPointerDown: stop, children: [
      _j('div', { className: 'sp-x__a', 'data-pat': it.pat || 'grad', style: { '--a': it.a || AC[0], '--b': it.b || AC[2] } }),
      _j('div', { className: 'sp-x__c', children: it.t })
    ]})
  }
  const hasBody = !!(it.s && String(it.s).trim())
  return _js('button', { type: 'button', className: 'sp-p', style: { '--a': it.a }, onClick: () => on(it), onPointerDown: stop, children: [
    _j('div', { className: 'sp-p__h', 'data-tone': it.tone || 'a' }),
    _js('div', { className: 'sp-p__b', children: [
      _j('div', { className: 'sp-k', children: it.tag || 'project' }),
      _j('div', { className: 'sp-t', children: it.t }),
      hasBody
        ? _j('div', { className: 'sp-s', children: it.s })
        : _js('div', { className: 'sp-p__rules', children: [0,1,2,3].map(i => _j('div', { key: i, className: 'sp-p__rule' })) }),
      _js('div', { className: 'sp-m', children: (it.chips || []).slice(0, 4).map((c, i) =>
        _j('span', { key: i, className: 'sp-ch', children: c })
      ) })
    ]})
  ]})
}

function buildHome(hp, companies, pcProjects, issues, prefs) {
  const items = []
  const sl = slots(Math.max(hp.length + 36, 48))
  const byPath = new Map()
  for (const p of pcProjects || []) {
    const cwd = p.codebase && (p.codebase.effectiveLocalFolder || p.codebase.localFolder)
    if (cwd) byPath.set(String(cwd).replace(/[/]+$/, ''), p)
    byPath.set(String(p.name || '').toLowerCase(), p)
  }
  hp.forEach((p, i) => {
    const path = p.primary_path || (p.folders && p.folders[0] && p.folders[0].path) || ''
    const key = String(path).replace(/[/]+$/, '')
    const pcP = byPath.get(key) || byPath.get(String(p.name || '').toLowerCase())
    const iss = (issues || []).filter(x => pcP && x.projectId === pcP.id)
    const open = iss.filter(x => !/done|cancelled|canceled/i.test(x.status || '')).length
    const hot = iss.filter(x => /in_progress|in_review|blocked/i.test(x.status || '')).slice(0, 2)
    const pin = (prefs.pins && prefs.pins[p.id]) || []
    items.push({
      id: p.id, k: 'paper', t: p.name || p.slug || 'project',
      s: p.description || path || 'Hermes project',
      tag: pcP ? 'hermes · paperclip' : 'hermes',
      a: AC[i % AC.length],
      chips: [
        pcP ? (pcP.status || 'pc') : 'local',
        open ? open + ' open' : (pcP ? '0 open' : 'no pc'),
        pin.length ? pin.length + ' pin' : null,
        hot[0] ? (hot[0].identifier || hot[0].status) : null
      ].filter(Boolean),
      x: sl[i].x, y: sl[i].y, r: rot(p.id), w: 178 + hn(p.id, 36), h: 164 + hn(p.id + 'h', 40),
      d: Math.min(i * 0.03, 0.45),
      href: pcP ? PC + '/' : null,
      meta: { hermes: p, pc: pcP, issues: iss, pins: pin }
    })
  })
  // company pulse stickies
  const pulse = (issues || []).filter(x => !/done|cancelled|canceled/i.test(x.status || '')).slice(0, 8)
  pulse.forEach((iss, j) => {
    const i = hp.length + j
    const sl0 = sl[i] || { x: 900 + j * 40, y: 600 }
    items.push({
      id: 'iss-' + iss.id, k: 'sticky', t: iss.identifier || 'issue',
      s: iss.title || '', tag: iss.status, a: ST[j % ST.length],
      x: sl0.x, y: sl0.y, r: rot(iss.id), w: 158 + hn(iss.id, 16), h: 148 + hn(iss.id + 'h', 18),
      d: Math.min(i * 0.03, 0.5),
      href: PC + '/issues/' + (iss.id || ''), meta: { issue: iss }
    })
  })
  // media tiles for company goals
  const co = (companies || [])[0]
  if (co) {
    const i = items.length
    const sl0 = sl[i] || { x: 1000, y: 200 }
    items.push({
      id: 'co-' + co.id, k: 'media', t: co.name + ' · live OS', a: '#0a84ff', b: '#bf5af2',
      x: sl0.x, y: sl0.y, r: rot(co.id), w: 200, h: 240, d: 0.2, href: PC + '/',
      meta: { company: co }
    })
  }
  // density fillers (doctrine stickies) so field is not sparse
  const fill = [
    { t: 'Kanban = execution', s: 'Spatial = scope + ops pulse', a: '#ffd60a' },
    { t: 'Paperclip owns runs', s: 'Read-only. Deep work in PC UI.', a: '#30d158' },
    { t: 'Pins stay local', s: 'localStorage only.', a: '#64d2ff' }
  ]
  fill.forEach((f, j) => {
    const i = items.length + j
    const sl0 = sl[i % sl.length] || { x: 700 + j * 30, y: 700 }
    items.push({
      id: 'fill-' + j, k: 'sticky', t: f.t, s: f.s, tag: 'doctrine', a: f.a,
      x: sl0.x + 28, y: sl0.y + 18, r: rot('f' + j), w: 156 + hn('fw'+j, 18), h: 146 + hn('fh'+j, 16), d: 0.32
    })
  })
  // abstract media tiles pad empty field (reference place-memory)
  for (let m = 0; m < 5; m++) {
    const i = items.length
    const sl0 = sl[i % sl.length] || { x: 900 + m * 40, y: 200 + m * 50 }
    items.push({
      id: 'med-' + m, k: 'media', t: ['Mood', 'Clip', 'Depth', 'Light', 'Chrome', 'Study'][m],
      a: AC[m % AC.length], b: AC[(m + 2) % AC.length], pat: PATS[m % PATS.length],
      x: sl0.x, y: sl0.y, r: rot('m' + m), w: 176 + hn('mw'+m, 36), h: 210 + hn('mh'+m, 44), d: 0.4
    })
  }
  return { w: 1600, h: 880, items: layoutBands(items) }
}

function buildProject(meta, prefs) {
  const items = []
  const iss = (meta.issues || []).slice().sort((a, b) => {
    const rank = s => (/in_progress/.test(s) ? 3 : /blocked|in_review/.test(s) ? 2 : /todo|backlog/.test(s) ? 1 : 0)
    return rank(b.status || '') - rank(a.status || '')
  })
  const sl = slots(Math.max(iss.length + 12, 24))
  iss.slice(0, 18).forEach((x, i) => {
    const hot = /in_progress|blocked|in_review/i.test(x.status || '')
    items.push({
      id: x.id, k: hot ? 'sticky' : 'paper', t: x.identifier || x.title,
      s: x.title, tag: x.status, a: hot ? ST[i % ST.length] : AC[i % AC.length],
      chips: [x.priority, x.status].filter(Boolean),
      x: sl[i].x, y: sl[i].y, r: rot(x.id), w: hot ? 160 : 210, h: hot ? 150 : 190,
      d: Math.min(i * 0.028, 0.45), href: PC + '/issues/' + (x.id || ''), meta: { issue: x }
    })
  })
  const pins = meta.pins || []
  pins.forEach((p, j) => {
    const i = iss.length + j
    const sl0 = sl[i] || { x: 100 + j * 40, y: 700 }
    items.push({
      id: 'pin-' + p, k: 'paper', t: String(p).split('/').pop(), s: p, tag: 'pinned',
      a: '#64d2ff', chips: ['pin'], x: sl0.x, y: sl0.y, r: rot(p), w: 200, h: 170,
      d: 0.3, meta: { path: p }
    })
  })
  if (!items.length) {
    items.push({ id: 'empty', k: 'paper', t: 'No open pulse', s: 'Paperclip issues will land here. Open full board for deep work.', tag: 'hint', a: '#8e8e93', chips: ['paperclip'], x: 400, y: 300, r: -1, w: 240, h: 180, d: 0, href: PC + '/' })
  }
  return { w: 1400, h: 960, items }
}

function SpatialPage() {
  css()
  const zp = usePanZoom()
  const [q, setQ] = useState('')
  const [sel, setSel] = useState(null)
  const [view, setView] = useState({ mode: 'home' }) // home | project
  const [desk, setDesk] = useState({ w: 1500, h: 1000, items: [] })
  const [st, setSt] = useState('…')
  const [tick, setTick] = useState(0)
  const prefs = useRef(LS.get())

  useEffect(() => {
    let dead = 0
    ;(async () => {
      setSt('sync')
      const [hp, cos] = await Promise.all([hermesProjects(), pc('/api/companies')])
      const companies = Array.isArray(cos) ? cos : []
      const co = companies[0]
      let pcProjects = [], issues = []
      if (co) {
        const [pp, iss] = await Promise.all([
          pc('/api/companies/' + co.id + '/projects'),
          pc('/api/companies/' + co.id + '/issues?limit=80')
        ])
        pcProjects = Array.isArray(pp) ? pp : []
        issues = Array.isArray(iss) ? iss : []
      }
      if (dead) return
      if (view.mode === 'project' && view.meta) {
        // refresh issues for open project; fall back to company open issues if no PC link
        const pid = view.meta.pc && view.meta.pc.id
        let mine = pid ? issues.filter(x => x.projectId === pid) : (view.meta.issues || [])
        if (!mine.length) mine = issues.filter(x => !/done|cancelled|canceled/i.test(x.status || '')).slice(0, 18)
        setDesk(buildProject({ ...view.meta, issues: mine, pins: (prefs.current.pins || {})[view.meta.hermes && view.meta.hermes.id] || view.meta.pins || [] }, prefs.current))
        setSt((mine && mine.length) + ' pulse')
      } else {
        const h = hp.length ? hp : [{ id: 'seed', name: 'Add a Hermes project', description: 'Projects appear from the sidebar workspace list.', primary_path: '' }]
        setDesk(buildHome(h, companies, pcProjects, issues, prefs.current))
        setSt(h.length + ' proj · ' + issues.filter(x => !/done|cancelled|canceled/i.test(x.status || '')).length + ' open')
      }
    })()
    return () => { dead = 1 }
  }, [tick, view.mode, view.key])

  // soft poll for 24/7 ops — 45s, no new backend
  useEffect(() => {
    const id = setInterval(() => setTick(t => t + 1), 45000)
    return () => clearInterval(id)
  }, [])

  const shown = useMemo(() => {
    const qq = q.trim().toLowerCase()
    if (!qq) return desk.items
    return desk.items.filter(it => [it.t, it.s, it.tag, ...(it.chips || [])].join(' ').toLowerCase().includes(qq))
  }, [desk, q])

  const open = it => {
    if (view.mode === 'home' && it.meta && it.meta.hermes) {
      setView({ mode: 'project', key: it.id, meta: it.meta })
      zp.reset()
      setSel(null)
      return
    }
    setSel(it)
  }

  const back = () => { setView({ mode: 'home' }); zp.reset(); setSel(null) }
  const openPc = () => { try { window.open(PC + '/', '_blank', 'noopener') } catch {} }

  return _js('div', {
    className: 'sp', 'data-testid': 'spatial-desk-root',
    children: [
      _js('div', { className: 'sp-top', children: [
        _j('b', { children: view.mode === 'project' ? (view.meta && view.meta.hermes && view.meta.hermes.name) || 'Project' : 'Command' }),
        _j('span', { children: st })
      ]}),
      _j('div', {
        className: cn('sp-st', zp.pan && 'p'), 'data-testid': 'spatial-stage', ...zp.stage,
        children: _js('div', {
          className: 'sp-w',
          style: { width: desk.w, height: desk.h, ...zp.world },
          children: shown.map(it =>
            _j('div', {
              key: it.id,
              className: cn('sp-i', sel && sel.id === it.id && 'on'),
              style: { left: it.x, top: it.y, width: it.w, height: it.h, zIndex: 4 + Math.round((it.priority || 20) / 12), '--r': it.r + 'deg', animationDelay: (it.d || 0) + 's' },
              children: _j(Card, { it, on: open })
            })
          )
        })
      }),
      _js('div', { className: 'sp-tb', onPointerDown: stop, children: [
        _j('input', { 'aria-label': 'Filter', placeholder: view.mode === 'home' ? 'Filter projects…' : 'Filter pulse…', value: q, onChange: e => setQ(e.target.value) }),
        _j('button', { type: 'button', title: 'Zoom out', onClick: zp.zout, children: '−' }),
        _j('span', { className: 'pct', children: Math.round(zp.s * 100) + '%' }),
        _j('button', { type: 'button', title: 'Zoom in', onClick: zp.zin, children: '+' }),
        _j('button', { type: 'button', title: 'Reset', onClick: zp.reset, children: '⌂' }),
        _j('button', { type: 'button', title: 'Refresh', onClick: () => setTick(t => t + 1), children: '↻' }),
        _j('button', { type: 'button', title: 'Paperclip', onClick: openPc, children: 'P' })
      ]}),
      _js('div', { className: 'sp-cr l', onPointerDown: stop, children: [
        view.mode === 'project'
          ? _j('button', { type: 'button', title: 'Back', onClick: back, children: '←' })
          : _j('button', { type: 'button', title: 'Chat', onClick: () => host.navigate('/'), children: '⌘' })
      ]}),
      _js('div', { className: 'sp-cr r', onPointerDown: stop, children: [
        _j('button', { type: 'button', title: 'Kanban', onClick: () => host.navigate('/kanban'), children: 'K' }),
        _j('button', { type: 'button', title: 'Paperclip board', onClick: openPc, children: '↗' })
      ]}),
      _j(Dialog, {
        open: !!sel, onOpenChange: o => { if (!o) setSel(null) },
        children: _j(DialogContent, {
          className: 'max-w-md border-border/60 bg-background/95 backdrop-blur-xl',
          children: sel ? [
            _js(DialogHeader, { key: 'h', children: [
              _j(DialogTitle, { children: sel.t }),
              _j(DialogDescription, { children: sel.s || '' })
            ]}),
            _js('div', { key: 'b', style: { display: 'flex', flexWrap: 'wrap', gap: 6, marginTop: 8 }, children: [
              _j(Badge, { variant: 'muted', children: sel.tag || sel.k }),
              ...(sel.chips || []).map((c, i) => _j(Badge, { key: i, variant: 'outline', children: c })),
              sel.href && _j('button', {
                type: 'button', className: 'sp-ch', style: { cursor: 'pointer', color: 'var(--acc,#0a84ff)' },
                onClick: () => { try { window.open(sel.href, '_blank', 'noopener') } catch {} },
                children: 'Open Paperclip'
              }),
              view.mode === 'project' && view.meta && view.meta.hermes && (sel.meta && (sel.meta.path || (sel.meta.issue && sel.meta.issue.id))) && _j('button', {
                type: 'button', className: 'sp-ch', style: { cursor: 'pointer' },
                onClick: () => {
                  const path = (sel.meta && sel.meta.path) || ('issue:' + sel.meta.issue.id)
                  prefs.current = togglePin(view.meta.hermes.id, path)
                  setTick(t => t + 1)
                },
                children: 'Pin/Unpin'
              })
            ]})
          ] : null
        })
      })
    ]
  })
}

function go() { host.navigate('/spatial') }

export default {
  id: ID, name: 'Spatial', defaultEnabled: true,
  register(ctx) {
    css()
    ctx.registerMany([
      { id: 'page', area: ROUTES_AREA, data: { path: '/spatial' }, render: () => _j(SpatialPage, {}) },
      { id: 'nav', area: SIDEBAR_NAV_AREA, order: 54, data: { codicon: 'dashboard', label: 'Spatial', path: '/spatial' } },
      { id: 'open', area: PALETTE_AREA, data: { id: 'spatial.open', label: 'Spatial: Command center', keywords: ['spatial', 'paperclip', 'projects', 'command'], run: go } },
      { id: 'key', area: KEYBINDS_AREA, data: { id: 'spatial.open', category: 'view', defaults: ['mod+alt+s'], label: 'Spatial: Command center', run: go } }
    ])
  }
}
