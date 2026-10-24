---
name: delegation-slot-config
description: "Configure delegation and track DevPass slot headroom."
version: 1.0.0
author: Khai Dong Nguyen (PawnmasterII), Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [delegation, slots, electronhub, concurrency, config]
    category: devops
    related_skills: [delegation-playbook]
---

# Delegation slot configuration & tracking (ElectronHub DevPass)

Configures Hermes for the 5-slot ek-dev- key and keeps a live ledger of who holds slots. Slot math model list + fan-out rules live in the `delegation-playbook` skill; this one is the HOW-TO-CONFIGURE + HOW-TO-TRACK half.

## When to Use
Run when delegation settings need changing, when starting/ending fan-out work, when 429s appear, or when stray Hermes instances may be leaking slots.

## 1. Configure (once, or when strategy changes)
```bash
hermes config set delegation.max_concurrent_children 4   # 5 slots - 1 reserved for parent
hermes config set delegation.model deepseek-v4.1-flash:dev  # strongest 1-slot :dev
hermes config get delegation   # verify both took
```
Never exceed 4 children while the user's session is live: parent turn + 4 children = 5/5. If the user will be idle (long unattended batch), 5 children is allowed — but re-check 429s after any user interjection.

## 2. Slot ledger (maintain every fan-out)
Track: `SLOTS_TOTAL=5 (Turbo; 2 on Lite) | glm-5.3:dev=2 | all other :dev=1 | parent live turn=1`. The ledger is a claim, not a guess — verify with a probe before declaring headroom:
```python
# Probe current key headroom (run before/at fan-out start and after batch):
import json, re, urllib.request
env = open(r"$HERMES_HOME/.env", encoding="utf-8").read()
dev = dict(re.findall(r'^([A-Z_]+)=(.+)$', env, re.M))["ELECTRONHUB_API_KEY"].strip()
req = urllib.request.Request("https://api.electronhub.ai/v1/chat/completions",
    data=json.dumps({"model":"glm-5.3-flash:dev","messages":[{"role":"user","content":"Say OK"}],"max_tokens":4}).encode(),
    headers={"Authorization": f"Bearer {dev}", "Content-Type":"application/json"})
with urllib.request.urlopen(req, timeout=45) as r:
    r.read(); print("status", r.status, "| rpm remaining", r.headers.get("x-ratelimit-remaining"))
```
Ledger rules:
- OPEN: when dispatching N children, log `used = children_slot_cost + (1 if user actively chatting else 0)`.
- CLOSE: when the batch-complete message lands, release the children's slots in the ledger.
- On any 429, read `x-ratelimit-remaining` FIRST: at/near 0 → RPM start budget (pace down; slow lane is 3/min burst 2 — wait for `x-ratelimit-reset`); healthy → slots: recount the ledger immediately (usual culprit: a live parent turn or a stray instance). `Retry-After ≈ 3` alone is ambiguous — full-speed RPM is one start per ~3s.
- `x-ratelimit-remaining` counts RPM starts (20/min burst 8), NOT slots — it is the disambiguator between the two 429 kinds.
- The pool is per-KEY and cross-model (proven 2026-10-15: 5 glm in flight → deepseek request also 429s). Mixing model families never adds capacity.

## 3. Stray-instance sweep (they silently consume slots)
Any other `hermes chat`, CLI session, or gateway process on this machine draws from the same 5 slots.
```powershell
Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'hermes_cli|hermes chat|gateway run|hermes serve' -or $_.Name -eq 'Hermes.exe' } |
  Select-Object ProcessId,ParentProcessId,Name,CommandLine
```
Kill rules:
- Map the desktop-app ancestry FIRST (`Hermes.exe` → cmd `hermes serve` → python base64 → kernel/tool children). NEVER kill a pid in the current session's own chain — that is the conversation you are running in.
- Safe to kill: `python -m hermes_cli.main gateway run` (and its children) and any `hermes chat` CLI leftovers, after confirming they are not ancestors of the current process. Kill with `taskkill /PID <pid> /T /F` (tree kill, catches node/bash children).
- After any kill: re-run the scan; expect exactly the desktop tree to remain. Then probe (section 2) to confirm the key still answers 200.

## 4. Verify after configuring
One real `delegate_task` dispatch with 2 trivial tasks; check the batch manifest (`cache/delegation/live/<id>/manifest.json`) shows the intended `model`, children return 200s, and no 429s appear in the batch result or the agent log (`$HERMES_HOME/logs/agent.log`). Children self-reporting their model id is the end-to-end proof.
