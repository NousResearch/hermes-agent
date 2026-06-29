#!/usr/bin/env python3
"""Render a self-contained Trial console from a run's status.json.

Dependency-free (stdlib only). RTL-aware, bilingual chrome (ar/en, fallback en).
The Trial orchestrator calls this after every gate transition; the page reloads
itself every few seconds while the run is in progress, so the user watches the
tribunal live.

Usage:
  python3 render_console.py --status <run>/status.json --out <run>/trial-console.html
If --out is omitted, the HTML is written next to status.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

TEMPLATE = r"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Trial</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=Readex+Pro:wght@300;400;600;700&family=Tajawal:wght@400;500;700&display=swap');
*{box-sizing:border-box}
html,body{margin:0;min-height:100vh;color:#f4f1e8;
  font-family:'Readex Pro','Tajawal',-apple-system,BlinkMacSystemFont,'Segoe UI','Geeza Pro',Tahoma,sans-serif;
  background:radial-gradient(120% 80% at 80% -10%,rgba(230,196,98,.10),transparent 60%),
             radial-gradient(100% 90% at 10% 110%,rgba(120,90,255,.08),transparent 55%),#06080f;}
#app{max-width:920px;margin:0 auto;padding:26px 22px 32px}
.head{display:flex;align-items:center;justify-content:space-between;gap:16px;flex-wrap:wrap}
.brand{display:flex;align-items:center;gap:12px}
.logo{width:40px;height:40px;color:#e6c462;filter:drop-shadow(0 0 10px rgba(230,196,98,.45))}
.word{font-size:30px;font-weight:700;letter-spacing:7px;line-height:1;
  background:linear-gradient(180deg,#fff3cf,#e6c462 55%,#b98f1e);-webkit-background-clip:text;background-clip:text;color:transparent}
.sub{font-size:12px;color:#98a2b3;margin-top:3px}
.tag{font-size:12px;color:#d9cfa3;max-width:330px;line-height:1.6;opacity:.9}
.chips{display:flex;gap:8px;flex-wrap:wrap;margin:16px 0 18px}
.chip{font-size:12px;color:#cdd5e0;background:rgba(255,255,255,.04);border:1px solid rgba(230,196,98,.16);padding:5px 11px;border-radius:999px}
.chip b{color:#e6c462}
.rail{display:flex;justify-content:space-between;position:relative;margin-bottom:20px}
.rail::before{content:"";position:absolute;top:13px;left:3%;right:3%;height:2px;background:rgba(230,196,98,.16)}
.step{position:relative;z-index:1;flex:1;text-align:center;font-size:11px;color:#98a2b3}
.step .d{width:26px;height:26px;border-radius:50%;margin:0 auto 7px;display:flex;align-items:center;justify-content:center;
  font-size:12px;background:#0e1426;border:2px solid rgba(255,255,255,.12);color:#98a2b3;transition:.4s}
.step.active{color:#e6c462}
.step.active .d{border-color:#e6c462;background:rgba(230,196,98,.14);color:#e6c462;box-shadow:0 0 0 4px rgba(230,196,98,.08)}
.step.done{color:#bfe9d4}
.step.done .d{border-color:#4ee0a3;color:#4ee0a3;background:#0f1c19}
.grid{display:grid;grid-template-columns:1fr 252px;gap:16px}
@media(max-width:640px){.grid{grid-template-columns:1fr}}
.card{background:rgba(255,255,255,.04);border:1px solid rgba(230,196,98,.16);border-radius:16px;padding:16px}
.phl{display:flex;justify-content:space-between;align-items:center;font-size:12px;color:#98a2b3;margin-bottom:14px}
.phl b{color:#f4f1e8;font-size:14px}
.rb{font-size:11px;color:#e6c462;background:rgba(230,196,98,.08);border:1px solid rgba(230,196,98,.16);padding:3px 9px;border-radius:999px}
.judges{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-bottom:14px}
.judge{background:#0c1325;border:1px solid rgba(255,255,255,.08);border-radius:14px;padding:12px 8px;text-align:center;transition:.4s}
.judge .ic{width:26px;height:26px;margin:0 auto 6px;color:#8b95a7;transition:.4s}
.judge .nm{font-size:12px;font-weight:600}
.judge .st{font-size:11px;color:#98a2b3;margin-top:4px}
.judge.think{border-color:#fbbf24}
.judge.think .ic,.judge.think .st{color:#fbbf24}
.judge.think .ic{animation:pulse 1s ease-in-out infinite}
.judge.pass{border-color:#4ee0a3;background:#0c1c17}
.judge.pass .ic,.judge.pass .st{color:#4ee0a3}
.judge.cond{border-color:#fbbf24;background:#1a1608}
.judge.cond .ic,.judge.cond .st{color:#fbbf24}
.judge.fail{border-color:#fb7185;background:#1d0f14}
.judge.fail .ic,.judge.fail .st{color:#fb7185}
@keyframes pulse{0%,100%{opacity:.55;transform:scale(1)}50%{opacity:1;transform:scale(1.1)}}
.builders{display:flex;flex-direction:column;gap:9px}
.builder{display:grid;grid-template-columns:132px 1fr 42px;gap:10px;align-items:center;font-size:11.5px;color:#cfd6e2}
.bar{height:8px;background:#0e1426;border:1px solid rgba(255,255,255,.06);border-radius:99px;overflow:hidden}
.bar i{display:block;height:100%;border-radius:99px;background:linear-gradient(90deg,#caa12a,#e6c462);transition:width .6s ease}
.builder.rework .bar i{background:linear-gradient(90deg,#b91c1c,#fb7185)}
.pct{font-size:10px;color:#98a2b3;text-align:end;font-variant-numeric:tabular-nums}
.ledger h4{margin:0 0 10px;font-size:12px;color:#e6c462;font-weight:600}
.log{display:flex;flex-direction:column;gap:8px}
.entry{display:flex;gap:8px;align-items:flex-start;font-size:11px}
.dot{width:7px;height:7px;border-radius:50%;margin-top:5px;flex:none;background:#98a2b3}
.entry.ok .dot{background:#4ee0a3}
.entry.info .dot{background:#5bc8f7}
.entry.warn .dot{background:#fbbf24}
.entry.bad .dot{background:#fb7185}
.etext{flex:1;color:#d7dde7;line-height:1.45}
.etime{font-size:10px;color:#5b6577;font-variant-numeric:tabular-nums}
.stamp{margin-top:16px;text-align:center}
.seal{display:inline-flex;align-items:center;gap:8px;font-size:17px;font-weight:700;padding:12px 24px;border-radius:14px;
  border:2px solid #4ee0a3;color:#4ee0a3;background:rgba(7,20,15,.6)}
.seal.esc{border-color:#fbbf24;color:#fbbf24;background:rgba(26,22,8,.6)}
.credit{text-align:center;margin-top:18px;font-size:11px;color:#5b6577}
.credit b{color:#e6c462}
</style>
</head>
<body>
<div id="app"></div>
<script>
const S = __STATUS_JSON__;
const LANG = (S.lang || 'en').toLowerCase();
const RTL = ['ar','he','fa','ur','ps'].indexOf(LANG.split('-')[0]) >= 0;
document.documentElement.dir = RTL ? 'rtl' : 'ltr';
document.documentElement.lang = LANG;
const GL = {
  en:["Framing","Research","Decision","Council","Build","Judging","Review","Delivery"],
  ar:["تأطير","بحث","قرار","مجلس","تنفيذ","تحكيم","مراجعة","تسليم"]
};
const C = {
  en:{sub:"Tri-judge methodology",tag:"No execution before the decision is approved; no delivery before an independent verdict.",phase:"Current gate",round:"Round",ledger:"Ledger",empty:"No verdicts yet.",delivered:"Delivered — approved",escalated:"Escalated — needs you",running:"in session",gates:"gates",by:"by"},
  ar:{sub:"منهجية الحكم الثلاثي",tag:"لا تنفيذ قبل اعتماد القرار، ولا تسليم قبل حكم مستقل.",phase:"المرحلة الحالية",round:"الجولة",ledger:"السِجلّ",empty:"لا أحكام بعد.",delivered:"تم التسليم — مُعتمد",escalated:"تصعيد — يحتاجك",running:"الجلسة منعقدة",gates:"بوابات",by:"من"}
};
const t = C[LANG] || C.en;
const gl = GL[LANG] || GL.en;
const num = RTL ? (s)=>(''+s).replace(/[0-9]/g, d=>'٠١٢٣٤٥٦٧٨٩'[d]) : (s)=>(''+s);
function esc(s){return (s==null?'':''+s).replace(/[&<>"]/g, c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));}
const scales = '<svg class="logo" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M12 3.2v17.6"/><path d="M7.5 20.8h9"/><path d="M5 7h14"/><path d="M5 7 3 11.6M5 7 7 11.6"/><path d="M2.4 11.6a2.6 2.6 0 0 0 5.2 0z"/><path d="M19 7 17 11.6M19 7 21 11.6"/><path d="M16.4 11.6a2.6 2.6 0 0 0 5.2 0z"/></svg>';
const jicon = '<svg class="ic" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7"><circle cx="12" cy="12" r="8"/><circle cx="12" cy="12" r="3"/></svg>';

const cur = S.current_gate || 0;
const gates = (S.gates && S.gates.length) ? S.gates : [1,2,3,4,5,6,7,8].map(n=>({n:n}));
function step(g){
  const n = g.n;
  const st = g.state || (n < cur ? 'done' : n === cur ? 'active' : 'pending');
  const lab = gl[n-1] || ('#'+n);
  return '<div class="step '+st+'"><div class="d">'+(st==='done'?'✓':num(n))+'</div>'+esc(lab)+'</div>';
}
const doneCount = gates.filter(g => g.state==='done' || g.n < cur).length;
const head = '<div class="head"><div class="brand">'+scales+'<div><div class="word">Trial</div><div class="sub">'+esc(t.sub)+'</div></div></div><div class="tag">'+esc(S.tagline || t.tag)+'</div></div>';
const chips = '<div class="chips"><span class="chip"><b>'+num(doneCount)+'</b>/<b>'+num(8)+'</b> '+esc(t.gates)+'</span><span class="chip">'+esc(S.mode||'standard')+'</span><span class="chip">'+esc(t.running)+'</span></div>';
const rail = '<div class="rail">'+gates.map(step).join('')+'</div>';
const curName = gl[(cur||1)-1] || '';
const phl = '<div class="phl"><span>'+esc(t.phase)+': <b>'+esc(curName)+'</b></span><span class="rb">'+esc(t.round)+' '+num(S.round||1)+'</span></div>';
const judges = (S.judges||[]).map(j => '<div class="judge '+esc(j.state||'idle')+'">'+jicon+'<div class="nm">'+esc(j.name||j.lens||'')+'</div><div class="st">'+esc(j.note||'')+'</div></div>').join('');
const builders = (S.builders||[]).map(b => {
  const pct = Math.max(0, Math.min(100, Math.round(b.pct||0)));
  return '<div class="builder '+esc(b.state||'')+'"><span>'+esc(b.name||'')+'</span><span class="bar"><i style="width:'+pct+'%"></i></span><span class="pct">'+num(pct)+'%</span></div>';
}).join('');
let stamp = '';
if (S.verdict === 'delivered') stamp = '<div class="stamp"><span class="seal">✦ '+esc(t.delivered)+'</span></div>';
else if (S.verdict === 'escalated') stamp = '<div class="stamp"><span class="seal esc">▲ '+esc(t.escalated)+'</span></div>';
const stage = '<div class="card">'+phl+(judges?'<div class="judges">'+judges+'</div>':'')+(builders?'<div class="builders">'+builders+'</div>':'')+stamp+'</div>';
const led = (S.ledger||[]).map(e => '<div class="entry '+esc(e.tone||'info')+'"><span class="dot"></span><span class="etext">'+esc(e.text||'')+'</span><span class="etime">'+esc(e.time||'')+'</span></div>').join('') || '<div class="entry"><span class="etext" style="color:#98a2b3">'+esc(t.empty)+'</span></div>';
const ledger = '<div class="card ledger"><h4>'+esc(t.ledger)+'</h4><div class="log">'+led+'</div></div>';
const credit = '<div class="credit">'+esc(t.by)+' <b>'+esc(S.creator||'Da7_Tech')+'</b> · Trial · '+esc(S.task||'')+'</div>';
document.getElementById('app').innerHTML = head + chips + rail + '<div class="grid">'+stage+ledger+'</div>' + credit;
if (S.verdict !== 'delivered' && S.verdict !== 'escalated') { setTimeout(()=>location.reload(), 4000); }
</script>
</body>
</html>
"""


def main() -> None:
    ap = argparse.ArgumentParser(description="Render the Trial live console.")
    ap.add_argument("--status", required=True, help="path to status.json")
    ap.add_argument("--out", default=None, help="output html path (default: next to status.json)")
    args = ap.parse_args()

    status_path = Path(args.status).expanduser()
    try:
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if not isinstance(status, dict):
            raise ValueError("status.json must be a JSON object")
    except Exception as exc:  # never crash the run over a malformed status file
        status = {
            "task": "(status.json unreadable: %s)" % exc,
            "lang": "en", "mode": "standard", "current_gate": 0, "round": 1,
            "verdict": "in-progress", "gates": [], "judges": [], "builders": [], "ledger": [],
        }

    out_path = Path(args.out).expanduser() if args.out else status_path.with_name("trial-console.html")
    embedded = json.dumps(status, ensure_ascii=False).replace("</", "<\\/")
    out_path.write_text(TEMPLATE.replace("__STATUS_JSON__", embedded), encoding="utf-8")
    print(str(out_path))


if __name__ == "__main__":
    main()
