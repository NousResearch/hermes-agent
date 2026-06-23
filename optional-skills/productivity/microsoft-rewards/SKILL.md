---
name: microsoft-rewards
description: Automate Microsoft Rewards points via Edge + cua-driver.
version: 1.0.0
author: community
license: MIT
platforms: [macos]
metadata:
  hermes:
    tags: [microsoft-rewards, edge, points, automation, productivity, macos]
    related_skills: [macos-computer-use]
---

# Microsoft Rewards Skill

Automate claiming Microsoft Rewards (Edge积分/Bing Rewards) — daily searches,
daily activities, claimable points, and bonus activities (quizzes, puzzles,
punchcards, search bonuses) — by driving your local Edge browser via
cua-driver. Works with both Chinese (中文) and English (英文) Microsoft
Rewards dashboards.

Requires macOS + `cua-driver` installed + Apple Events permission granted
to the browser (one-time setup via `enable_javascript_apple_events`).

## When to Use

- The user says "领积分", "Edge 积分", "Microsoft Rewards", "Bing Rewards"
- You need to claim daily rewards points automatically

## Prerequisites

- `cua-driver` installed (`hermes tools` enabled Computer Use)
- Apple Events permission granted (one-time):
  `cua-driver permissions grant` on `com.microsoft.edgemac`
- Edge browser installed and user logged into Microsoft account
- **After initial setup, permissions are already in place — proceed directly**

## Quick Reference

```
CUA_DRIVER=cua-driver  # ensure on $PATH: `which cua-driver`
```

| Action | Command |
|--------|---------|
| Launch Edge + open URL | `open -a "Microsoft Edge" 'https://rewards.bing.com/dashboard'` |
| Resize window | `call page` JS: `window.moveTo(0,0); window.resizeTo(1400,900)` |
| Read buttons | JS: `JSON.stringify(Array.from(document.querySelectorAll("button"))...)` |
| Read points status | Read button 0 (total) and button 2 (claimable) |
| Click claim | Expand card, then click "领取积分" button in panel |
| Navigate to URL | `call page navigate` — preferred over `window.location.href` |
| Bing searches | `call page navigate` to `https://www.bing.com/search?q=<term>` |
| Kill & relaunch | `kill -9 <pid>; sleep 2;` then `launch_app` + `list_windows` again |

## Procedure

### 0. Pre-flight Check

Before touching Edge or cua-driver, always validate prerequisites first:

```python
import shutil, subprocess, json, time

# 1. cua-driver available?
if not shutil.which("cua-driver"):
    raise RuntimeError("cua-driver not found — run `hermes tools` to enable Computer Use")

# 2. Daemon running?
r = subprocess.run(["cua-driver", "status"], capture_output=True, text=True)
if "not running" in r.stdout:
    subprocess.Popen(["cua-driver", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(2)

# 3. Edge already open? Avoid relaunching unnecessarily
r = subprocess.run(["cua-driver", "call", "list_windows"], capture_output=True, text=True)
windows = json.loads(r.stdout).get("windows", [])
edge_windows = [w for w in windows if w.get("app_name", "").lower() == "microsoft edge"]
if edge_windows:
    print(f"Edge already running, {len(edge_windows)} window(s)")

# 4. Permissions are one-time setup — do NOT re-ask or re-prompt
```

**Important**: Always write complex JSON/JavaScript to a `.py` or `.js` file
rather than inlining it in a shell command — shell escaping errors are silent
and waste time. The `scripts/rewards-buttons.js` file is a good example.

Only after pre-flight passes, proceed to step 1.

### 1. Launch Edge & Open Dashboard (one step)

Always use `open -a` to launch Edge with the URL pre-loaded — saves one navigation:

```python
import subprocess, json, time

# Launch Edge directly to the dashboard URL
subprocess.run(['open', '-a', 'Microsoft Edge',
    'https://rewards.bing.com/dashboard'], capture_output=True, text=True)
time.sleep(3)  # wait for Edge to open and page to start loading

# Find the Edge window
r = subprocess.run([CUA_DRIVER, 'call', 'list_windows'], text=True, capture_output=True)
windows = json.loads(r.stdout)['windows']
# Filter for Edge windows (title contains "Rewards" or "Microsoft")
edge_windows = [w for w in windows if w.get('app_name','').lower() == 'microsoft edge']
if not edge_windows:
    # Fall back to any window from pid
    pid = None
    for w in windows:
        if w.get('app_name','').lower() == 'microsoft edge':
            pid = w['pid']
            break
    raise RuntimeError(f"Could not find Edge window (pid={pid})")
# Pick the on-screen window with the most relevant title
edge_windows.sort(key=lambda w: 0 if w.get('is_on_screen') else 1)
wid = edge_windows[0]['window_id']
pid = edge_windows[0]['pid']
print(f"Edge: pid={pid}, wid={wid}")

# Resize for full rendering
js = 'window.moveTo(0,0); window.resizeTo(1400,900); window.scrollTo(0,0)'
subprocess.run([CUA_DRIVER, 'call', 'page'],
    input=json.dumps({'action':'execute_javascript','pid':pid,'window_id':wid,'javascript':js}),
    capture_output=True, text=True)
time.sleep(1)  # let resize settle

# Use page navigate to ensure we're on the dashboard (reliable even if Edge was slow)
subprocess.run([CUA_DRIVER, 'call', 'page'],
    input=json.dumps({'action':'navigate','pid':pid,'window_id':wid,
        'url':'https://rewards.bing.com/dashboard'}),
    capture_output=True, text=True)
time.sleep(2)  # wait for full render
```

### 2. Read Points Status

Read the rewards-check.js script to get button list. Key buttons (index-based, locale-independent):

| Index | Chinese (中文) | English |
|-------|---------------|---------|
| 0 | 总计积分 (e.g. `5,565 SO`) | Total points |
| 2 | 可领取 XXX 领取 | Claim XXX available |
| 16 | 每日连签 N 天 | Daily streak N days |
| 22 | 必应 搜索: X/Y | Bing searches: X/Y |
| 23 | 每日活动 活动: X/Y | Daily activities: X/Y |
| 24 | Microsoft Edge 分钟: X/Y | Microsoft Edge minutes: X/Y |
| 25 | 移动应用 签到: X/Y | Mobile app check-in: X/Y |

### 3. Claim Available Points

If claimable > 0:

```python
# IMPORTANT: React Aria buttons do NOT respond to .click(). Use aria-controls.

# Step 1: Get the panel ID from button[2]'s aria-controls attribute
js = 'document.querySelectorAll("button")[2].getAttribute("aria-controls")'
r = subprocess.run([CUA_DRIVER, 'call', 'page'],
    input=json.dumps({'action':'execute_javascript','pid':pid,'window_id':wid,'javascript':js}),
    capture_output=True, text=True)
panel_id = r.stdout.strip()
time.sleep(1)

# Step 2: Click the '领取积分' button inside the panel
# The panel has 3 buttons: [0]=empty, [1]="权益说明", [2]="领取积分"
js = f'document.getElementById("{panel_id}").querySelectorAll("button")[2].click()'
subprocess.run([CUA_DRIVER, 'call', 'page'],
    input=json.dumps({'action':'execute_javascript','pid':pid,'window_id':wid,'javascript':js}),
    capture_output=True, text=True)
time.sleep(1.5)

# Step 3: Navigate back to dashboard to see updated state
subprocess.run([CUA_DRIVER, 'call', 'page'],
    input=json.dumps({'action':'navigate','pid':pid,'window_id':wid,
        'url':'https://rewards.bing.com/dashboard'}),
    capture_output=True, text=True)
time.sleep(2)
```

### 4. Do Bing Searches

If search count < target:

Use `page navigate` instead of `window.location.href` — the navigate action
detects page load and returns faster:

```python
for q in ['cats','dogs','music','sports','travel','food','today','news',
           'weather','art','nature','ocean','computers','ai','games','health',
           'fitness','photo','books','science']:
    subprocess.run([CUA_DRIVER, 'call', 'page'],
        input=json.dumps({'action':'navigate','pid':pid,'window_id':wid,
            'url':f'https://www.bing.com/search?q={q}'}),
        capture_output=True, text=True)
    time.sleep(1.5)  # shorter wait — navigate blocks until page starts loading
```

Total for 20 searches: ~30s (was ~40s with href + 2s sleep).

### 5. Click Daily Activities

Prefer a JS-based approach — avoid AX tree which can be empty in background mode:

```python
# Find <a> tags whose innerText contains "+10" and click them
js = '''(function(){
  var links = document.querySelectorAll("a");
  var r = [];
  links.forEach(function(l, i) {
    var t = (l.innerText || "").replace(/\\s+/g, " ").trim();
    if (t.includes("+10")) {
      l.click();
      r.push(i + ": " + t.substring(0, 60));
    }
  });
  return r.join("||");
})()'''
r = subprocess.run([CUA_DRIVER, 'call', 'page'],
    input=json.dumps({'action':'execute_javascript','pid':pid,'window_id':wid,'javascript':js}),
    capture_output=True, text=True)
print('CLICKED:', r.stdout)
time.sleep(3)

# Navigate back to dashboard
subprocess.run([CUA_DRIVER, 'call', 'page'],
    input=json.dumps({'action':'navigate','pid':pid,'window_id':wid,
        'url':'https://rewards.bing.com/dashboard'}),
    capture_output=True, text=True)
time.sleep(2)

# Repeat for remaining activities (check button 23 for "活动: X/3")
```

### 6. Find & Complete Bonus Activities

After completing the daily quota, navigate to the earn page and look for bonus
activities that offer extra points beyond the daily limit. Use `page navigate`
instead of JS href:

```python
# Navigate to earn page
subprocess.run([CUA_DRIVER, 'call', 'page'],
    input=json.dumps({'action':'navigate','pid':pid,'window_id':wid,
        'url':'https://rewards.bing.com/earn'}),
    capture_output=True, text=True)
time.sleep(2)
```

Use this script to find bonus activity links (look for `+5`, `+10`, `+15`, `+50` patterns):

```javascript
JSON.stringify(Array.from(document.querySelectorAll("a"))
  .map((b,i)=>({
    i,text:(b.innerText||"").replace(/\\s+/g," ").trim().substring(0,120),
    href:b.href||""
  }))
  .filter(t=>t.text.length>5)
)
```

**Types of bonus activities** (appear on earn page):

| Type | Chinese indicator | English indicator | Points | Action |
|------|------------------|-------------------|--------|--------|
| Search bonus | `+15`, `冥想的好处` | `+15`, `meditation` | +10 to +15 | Navigate to the Bing search URL |
| Quiz | `你是否知道答案` | `Bing homepage quiz` | +5 | Navigate to quiz URL, click answer |
| Puzzle | `完成此拼图` | `Complete the puzzle` | +5 | Navigate to puzzle URL (often auto-completes) |
| Punchcard | `趋势`, `0/4 个任务` | `Punchcard`, `0/4 tasks` | +50 total | Navigate to quest URL, click tasks sequentially |
| Partner offers | `3 次搜索可得 10 分` | `Get 10 points` | +10 | Follow referral URL |

**To complete quiz activities** (section 6a):

Quizzes work the same in all locales — answer choices use `A.` / `B.` / `C.`
and the "Next" / "下一个" button advances:

```python
# Navigate to the quiz URL
subprocess.run([CUA_DRIVER, 'call', 'page'],
    input=json.dumps({'action':'navigate','pid':pid,'window_id':wid,
        'url':f'{quiz_url}'}),
    capture_output=True, text=True)
time.sleep(2)

# Find quiz answer buttons — Bing quiz answers are `<a>` tags with option text
r = subprocess.run([CUA_DRIVER, 'call', 'page'], capture_output=True, text=True,
    input=json.dumps({'action':'execute_javascript','pid':pid,'window_id':wid,
        'javascript':"""JSON.stringify(Array.from(document.querySelectorAll("a"))
          .filter(a=>a.innerText.trim().match(/^[A-C]\\./))
          .map(a=>({text:a.innerText.trim().substring(0,40)}))) """}))
# Click the first answer to submit it
js = """document.querySelectorAll("a").forEach(a=>{if(a.innerText.trim().match(/^A\\./))a.click()})"""
subprocess.run([CUA_DRIVER, 'call', 'page'],
    input=json.dumps({'action':'execute_javascript','pid':pid,'window_id':wid,'javascript':js}))

# Advance through quiz questions — matches "下一个" (zh) or "Next" (en)
time.sleep(1.5)
js = """Array.from(document.querySelectorAll("button,a,span,div")).forEach(el=>{
  const t = (el.innerText||"").trim();
  if(t === "下一个" || t === "Next") el.click();
})"""
subprocess.run([CUA_DRIVER, 'call', 'page'],
    input=json.dumps({'action':'execute_javascript','pid':pid,'window_id':wid,'javascript':js}))
```

**Note on punchcards**: Each punchcard has 3-4 tasks. Tasks 2-4 often require
24h waits between them (e.g., "请在完成第一天后等待 24 小时或更长时间" /
"Please wait 24 hours after completing day 1"). Only the first task can be
completed per day. Click task buttons by text — matches both Chinese and
English labels:

```python
# Find and click punchcard task buttons
js = """Array.from(document.querySelectorAll("span")).filter(s => {
  const t = s.innerText.trim();
  return ["查看赛程","查看笔记本电脑","发现优惠","View Schedule","View Laptops","Discover Deals"].includes(t);
}).forEach(s => s.click());"""
subprocess.run([CUA_DRIVER, 'call', 'page'],
    input=json.dumps({'action':'execute_javascript','pid':pid,'window_id':wid,'javascript':js}),
    capture_output=True, text=True)
time.sleep(3)
```

### 7. Verify & Report

Navigate back to dashboard, then use `page navigate` to ensure fresh state:

```python
subprocess.run([CUA_DRIVER, 'call', 'page'],
    input=json.dumps({'action':'navigate','pid':pid,'window_id':wid,
        'url':'https://rewards.bing.com/dashboard'}),
    capture_output=True, text=True)
time.sleep(2)
```

Then re-read button list and compare before/after points.
Report total points gained, which activities were completed, and note any
punchcard progress (`X/4 个任务`).

## Pitfalls

- **Performance**: The biggest time sink is Bing searches (20 × ~1.5s = 30s).
  Total expected runtime with this optimized approach: ~45-60s (vs 10-20min before).
  If Edge is already open at the dashboard, runtime drops to ~35-50s.
- **Cold start**: First Edge launch takes 3-5s just to open the app window. The
  `open -a "Microsoft Edge" '<url>'` pattern loads the page while Edge starts.
- **cua-driver daemon**: Every `cua-driver call` starts a daemon session if
  one isn't running. Use `cua-driver status` to check, or pre-start with
  `cua-driver serve &`.
- **`computer_use` tool may not connect**: The Hermes `computer_use` tool's MCP
  session can fail with "cua-driver session not started". **Workaround**: use
  cua-driver directly via CLI (`cua-driver call <tool>`) or pipe JSON to
  `cua-driver mcp` instead. Both bypass the tool wrapper and work reliably.
- **Permissions are one-time**: After initial `cua-driver permissions grant`,
  do NOT re-prompt or re-check. Proceed directly.

## Verification
Check button index 0 total points increased, index 2 claimable dropped to 0,
index 22 shows `1/1`, index 23 shows `3/3`.
