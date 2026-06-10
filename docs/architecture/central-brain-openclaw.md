# สถาปัตยกรรม Orchestration Core + Action Runtime

> เดิมเรียก "Central Brain + OpenClaw" — ดูเหตุผลที่เปลี่ยนศัพท์ใน §0
>
> **สถานะ:** Draft v2 (ตัดสินทิศหลักแล้ว) · **อัปเดต:** 2026-06-10 · **ขอบเขต:** เอกสารออกแบบเชิงสถาปัตยกรรมสำหรับการแยกบทบาท "สมองกลาง (Orchestration Core)" ออกจาก "ชั้นปฏิบัติการ (Action Runtime)" ในระบบ Hermes
>
> เอกสารนี้เป็น **design doc** สำหรับทีมใช้อ้างอิงและถกเถียง **ยังไม่ได้ implement** ส่วนที่เป็นข้อเสนอกำกับว่า *(เสนอ)* ส่วนที่เป็นของจริงในโค้ดวันนี้อ้างไฟล์ประกอบ

---

## 0. การตัดสินศัพท์ + Glossary (อ่านก่อน)

### 0.1 ทำไมเลิกใช้ชื่อ "OpenClaw" เรียกชั้น executor

คำว่า **OpenClaw** ในโค้ด Hermes มีความหมายเดิมที่ชัดมากอยู่แล้ว — เป็น **ผลิตภัณฑ์ agent อีกตัวที่ Hermes ย้ายข้อมูลออกมาจาก**: `hermes claw migrate`, ตรวจจับ `~/.openclaw` ตอน `hermes setup`, skill `openclaw-migration`, import ไปไว้ที่ `~/.hermes/skills/openclaw-imports/` (ดู `README.md:147-173`) และ `mcp_serve.py` เป็น *"the Hermes equivalent of OpenClaw's WebSocket gateway bridge"* ที่ mirror *"OpenClaw's 9-tool MCP channel bridge surface"* (`mcp_serve.py:8,208`)

ถ้าเอาชื่อ OpenClaw มาเรียกชั้น executor ของเราด้วย จะ **ชื่อชนกันตั้งแต่วันแรก** เราจึงตัดสินดังนี้:

> ### ✅ การตัดสิน (เดิมคือ §10 ข้อ 1)
> **เลือกข้อ (ข): สร้าง Action Runtime ที่เป็น Hermes-native เอง** โดยยืมแค่ *pattern / bridge compatibility* จาก OpenClaw — **ไม่ใช่** เรียก OpenClaw ตัวจริงมาเป็น executor หลัก
>
> เหตุผล: `mcp_serve.py` เป็น bridge ด้าน *messaging* ไม่ใช่ execution runtime ทั่วไป การลาก OpenClaw จริงมาเป็นแกนคือเพิ่ม coupling + ทำให้ชื่อสับสน

### 0.2 Glossary (ศัพท์ที่ล็อกแล้ว — ใช้ให้ตรงกันทั้งทีม)

| ศัพท์ | หมายถึง | วันนี้ ≈ |
|---|---|---|
| **Orchestration Core** (สมองกลาง) | ชั้น reasoning / planning / orchestration / memory / policy *(alias: Reasoning Core — สลับได้)* | `run_agent.py::AIAgent` + `agent/conversation_loop.py` + `agent/curator.py` |
| **Action Runtime** (แขนขา) | ชั้น execution / tool-use / automation / I/O กับโลกภายนอก | `tui_gateway/` dispatch + `tools/registry.py` + `providers/` |
| **`ExecutionTask` / `ExecutionResult`** *(เสนอ)* | สัญญา (contract) ที่ขอบเขต Core ↔ Runtime | ยังไม่มี — สร้างใน Phase 2 |
| **`SessionState`** *(เสนอ)* | object ที่จะแตกออกมาจาก dict `_sessions` พร้อมถือ lock ของตัวเอง | ยังไม่มี — สร้างใน Phase 3 |
| **"OpenClaw"** | **ต่อจากนี้หมายถึง *compatibility / reference เท่านั้น*** = (ก) ผลิตภัณฑ์เดิมที่ migrate ออกมา, (ข) ผิว messaging-bridge ที่ `mcp_serve.py` เลียนแบบ | `mcp_serve.py`, `README.md` migration section |
| **`action_runtime/`** *(ชื่อ package ที่จองไว้)* | ที่อยู่ของ Action Runtime abstraction ในอนาคต | **ยังไม่สร้างโฟลเดอร์** — สร้างตอน Phase 2 เมื่อมี skeleton + contract + test จริง (ไม่สร้างโฟลเดอร์เปล่าใน Phase 0) |

---

## 1. บทสรุป (TL;DR)

```
Orchestration Core "คิดและสั่งงาน"  →  Action Runtime "ลงมือทำ"  →  คืนผล  →  Core "ประเมินต่อ"
```

- **Orchestration Core (สมองกลาง)** = reasoning / planning / orchestration / memory ดูแลการคิด แตกงาน เลือกนโยบาย ถือบริบทและความจำ
- **Action Runtime (แขนขา)** = execution / tool-use / automation ลงมือรัน tool, จัดการไฟล์, เชื่อมต่อระบบอื่น แล้วคืนผล
- **หัวใจไม่ใช่ตัว component แต่ละชั้น แต่คือ "สัญญา (contract)" ที่ขอบเขตระหว่างสองชั้น** — ถ้าขอบนี้คลุมเครือ ระบบจะเปราะทันที (§5)
- ในโค้ด Hermes วันนี้ ทั้งสองบทบาท **อยู่ในโปรเซสเดียวกัน**: reasoning ทั้งหมดคือคลาส `AIAgent` ที่ถูก instantiate ซ้ำ **22 จุด** (นับด้วย grep ยืนยัน 2026-06-10) ทั่วทั้ง codebase การจะแยกจริงต้องค่อย ๆ ทำตาม phase ใน §11 — **ไม่เริ่มจากงานเสี่ยงสุด (ยก Core เป็น service)**

---

## 2. ภาพรวมสถาปัตยกรรม

```
┌──────────────────────────────────────────────────────────────────────┐
│                              ผู้ใช้ (User)                              │
│        CLI · TUI · Web Dashboard · Desktop · Chat (TG/Discord/…)        │
└───────────────────────────────────┬──────────────────────────────────┘
                                     │  user intent (ข้อความ/เป้าหมาย/คำสั่ง)
                                     ▼
┌──────────────────────────────────────────────────────────────────────┐
│                    ORCHESTRATION CORE (สมองกลาง)                       │
│        reasoning · planning · orchestration · memory · policy          │
│                                                                        │
│   1) รับเป้าหมายจากผู้ใช้      4) เลือก intent/นโยบาย (เช่น โมเดลไหน)     │
│   2) วิเคราะห์เป้าหมาย         5) ถือ context & long-term memory          │
│   3) แตกงานเป็น ExecutionTask  6) ประเมินผล → re-plan / retry             │
│                                                                        │
│        วันนี้ ≈ AIAgent + agent/conversation_loop.py + curator.py       │
└──────────────┬──────────────────────────────────────▲────────────────┘
               │                                       │
  ExecutionTask┤ (สัญญา: structured, มี success-       │── ExecutionResult (structured:
   (สั่งงาน)    │  criteria + constraints + ctx ref +    │   status/outputs/error/retryable/
               │  task_id + idempotency_key)            │   side_effects/progress + task_id)
               ▼                                       │
┌──────────────────────────────────────────────────────────────────────┐
│                       ACTION RUNTIME (แขนขา)                          │
│         execution · tool-use · automation · I/O กับโลกภายนอก            │
│                                                                        │
│   • รัน tool / shell / โค้ด      • เชื่อมต่อระบบอื่น (API, แพลตฟอร์ม)     │
│   • จัดการไฟล์ / automation      • คืน result + error + progress         │
│   • ควร stateless ต่อ 1 task (idempotent, retry ปลอดภัย)                 │
│                                                                        │
│   วันนี้ ≈ tui_gateway/ + tools/registry.py + providers/                │
│   (mcp_serve.py = OpenClaw-compat *messaging* bridge — แยกต่างหาก §5.5)  │
└──────────────────────────────────────────────────────────────────────┘
```

**หลักการสำคัญ 4 ข้อ:**

1. **สัญญาที่ขอบเขตคือหัวใจ** — Core ส่ง `ExecutionTask` มีโครงสร้าง, Runtime คืน `ExecutionResult` มีโครงสร้าง (§5)
2. **Core ถือ memory, Runtime ควร stateless ต่อ task** — recover/retry/parallelize ง่าย (§6)
3. **Core เลือก intent ระดับสูง ไม่ผูกกับ tool ตรง ๆ** — สลับ runtime ได้ภายหลัง (§7)
4. **Feedback loop ต้องพอให้ re-plan ได้** — Runtime คืนสัญญาณพอที่ Core วางแผนใหม่เมื่อพลาด (§8)

> **หลักการข้ามทุก phase (cross-cutting)** — ดู §12: *Additive-first compatibility · Test-first · Observability/trace · Idempotency*

---

## 2.1 Skeleton tree (verified 2026-06-10 ด้วย grep/wc — ตรงกับโค้ดจริง)

```text
hermes-agent/
├── 🧠 CORE — reasoning / planning / memory
│   ├── run_agent.py                 AIAgent — สมองตัวจริง (class@320, 5,272 บรรทัด)
│   ├── agent/conversation_loop.py   turn loop · dispatch · retry · compression (4,945)
│   ├── agent/curator.py             self-improvement: สร้าง/แก้ skill (1,843)
│   └── cron/scheduler.py            ตัดสิน "เมื่อไร" — spawn ตามเวลา (AIAgent@1772)
│
├── 🤝 CONTRACT — ขอบเขตระหว่าง 2 ระบบ
│   └── action_runtime/              ← Phase 2 (ใหม่, ยังไม่ commit)
│       ├── contract.py              ExecutionTask/Result + 8 class (Status/ErrorType/…)
│       ├── adapters.py              shell/cli/plugin/slash × (to_result/to_wire) byte-id
│       └── __init__.py              · ทดสอบ: tests/action_runtime/ (13 test เขียว)
│
├── 🦾 RUNTIME — execution / tool-use / IO
│   ├── tui_gateway/server.py        dispatcher 74 @method (8,551) + ถือ agent lifecycle
│   │     ↳ exec: cli.exec@6245 · slash.exec@7420 · shell.exec@8518
│   │       ทั้ง 3 route ผ่าน action_runtime (cli@6251 · plugin@7479 · slash@7513 · shell@8542)
│   ├── tui_gateway/{entry,transport,slash_worker}.py   loop · transport · subprocess
│   ├── tools/registry.py            tool discovery / invocation
│   └── providers/                   provider API transport
│
├── 💾 STATE / MEMORY
│   ├── hermes_state.py              SQLite state.db · FTS5 · branch chains (4,289)
│   └── (server.py) _spawn_trees_root@4087   spawn-tree บนดิสก์ แยกจาก state.db
│
├── 🖥️ FRONTENDS
│   ├── cli.py                       HermesCLI — frontend+runtime+host ปนกัน (AIAgent ×2)
│   ├── web/ · apps/desktop/         คุย backend ผ่าน gateway RPC
│   └── gateway/platforms/           TG · Discord · Feishu (AIAgent ×2)
│
└── 🔌 BRIDGE
    └── mcp_serve.py                 10 OpenClaw-compat tool (messaging ไม่ใช่ exec)
```

> **โมเดลความคิด:** `🧠 Core คิด → 🤝 ส่งงานข้ามขอบ → 🦾 Runtime ทำ → คืนผล → 🧠 Core ประเมินต่อ`
>
> **กุญแจสำคัญ:** ไม่มี Core process เดียว — `AIAgent` ถูกสร้าง **22 จุด** (§9 ข้อ 1) "Core" วันนี้จึงเป็น *library ฝังในทุก frontend* ไม่ใช่ service กลางตัวเดียว

---

## 3. นิยามบทบาทแต่ละชั้น

### 3.1 Orchestration Core (สมองกลาง)

| หน้าที่ | รายละเอียด | อยู่ในโค้ดวันนี้ที่ |
|---|---|---|
| รับคำสั่ง / เป้าหมาย | จากผู้ใช้ผ่าน frontend ใด ๆ | frontends (§4) ป้อนเข้า `AIAgent` |
| วิเคราะห์เป้าหมาย + วางแผน | reasoning loop, แตกงานเป็นขั้น | `run_agent.py::AIAgent`, `agent/conversation_loop.py` |
| ตัดสินใจเลือก tool/นโยบาย | ตอนนี้เลือก tool ตรง ๆ ใน loop | tool-call decisioning ใน conversation loop |
| ถือ context / memory / state | ความจำระยะยาว, สรุป session | `hermes_state.py` (SQLite), `~/.hermes/memories/`, curator |
| self-improvement | สร้าง/ปรับ skill, trim memory | `agent/curator.py` |
| ตัดสินใจ "เมื่อไร" | spawn งานตามเวลา | `cron/scheduler.py` |

**Core ไม่ควรทำ:** I/O กับโลกภายนอกโดยตรง (รัน shell, ยิง API ปลายทาง, เขียนไฟล์) — ส่งต่อให้ Action Runtime

### 3.2 Action Runtime (แขนขา)

| หน้าที่ | รายละเอียด | อยู่ในโค้ดวันนี้ที่ |
|---|---|---|
| รัน tool / command | shell, โค้ด, subprocess | `tui_gateway/server.py` handlers (`shell.exec`, `cli.exec`, `slash.exec`) |
| จัดการไฟล์ / automation | อ่าน/เขียน/ย้ายไฟล์, งานอัตโนมัติ | tools ใน `tools/registry.py` |
| เชื่อมต่อระบบอื่น | ยิง API ผู้ให้บริการ, ส่งข้อความข้ามแพลตฟอร์ม | `providers/`, `gateway/` platform adapters |
| คืนผล + error + progress | ป้อนกลับให้ Core ประเมิน | event stream ผ่าน JSON-RPC (§5) |

**Action Runtime ไม่ควรทำ:** ตัดสินใจเชิงเป้าหมาย/วางแผน/เลือกนโยบาย — รับ `ExecutionTask` มาทำตาม แล้วรายงานผลอย่างซื่อสัตย์

> **หมายเหตุ:** `mcp_serve.py` (OpenClaw-compat bridge) **ไม่ใช่** ส่วนหนึ่งของ Action Runtime ในความหมายนี้ — มันเป็น messaging-channel bridge แยกต่างหาก (§5.5)

---

## 4. Hermes วันนี้ map เข้ากับโมเดลนี้อย่างไร

> ที่มา: การสำรวจโค้ดจริง ความจริงสำคัญคือ **วันนี้ทั้งสองบทบาทอยู่ในโปรเซสเดียวกัน** — `AIAgent` ถูก instantiate ใหม่ 22 จุด ไม่ใช่ service กลางตัวเดียว

| Component (ไฟล์) | บทบาทตามโมเดล | หมายเหตุ |
|---|---|---|
| `run_agent.py::AIAgent` (~5,272 บรรทัด) | 🧠 **Core** | reasoning loop ตัวจริงตัวเดียวของระบบ |
| `agent/conversation_loop.py` (~4,945) | 🧠 Core | turn loop, tool dispatch, retry, compression |
| `agent/curator.py` | 🧠 Core (autonomous) | ตัวตัดสินใจ self-directed: สร้าง/ปรับ skill |
| `cron/scheduler.py` | 🧠 Core (เลือก "เมื่อไร") | spawn agent ตามเวลา |
| `cli.py::HermesCLI` (~16,180) | 🟠 **ปนกัน** | เป็นทั้ง frontend + runtime + agent host พร้อมกัน → จุดที่ขอบเขตเบลอที่สุด |
| `tui_gateway/server.py` (8,551 บรรทัด, 74 `@method`) | 🦾 **Action Runtime** (แต่ถือ agent lifecycle ด้วย) | JSON-RPC dispatcher; แต่ยังทำ `_make_agent`, `session.steer`, `agent.switch_model` เอง |
| `tui_gateway/entry.py`, `transport.py`, `slash_worker.py` | 🦾 Action Runtime | event loop + transport + subprocess executor |
| `tools/registry.py` + `providers/` | 🦾 Action Runtime | tool discovery/invocation + API transport |
| `agent/model_metadata.py`, `hermes_cli/models.py`, `model_switch.py` | 🦾 Action Runtime (plumbing บริสุทธิ์) | metadata/credential/validation — **sequential chain, ไม่มี policy เลย** = ส่วนที่สะอาดที่สุด |
| `mcp_serve.py` | 🔌 **OpenClaw-compat bridge** (แยก) | mirror OpenClaw 9-tool MCP surface — เป็น *messaging* ไม่ใช่ generic exec (§5.5) |
| `web/`, `apps/desktop/` | 🖥️ Frontend → Core | human interface; คุยกับ backend ผ่าน gateway RPC ล้วน |
| `hermes_state.py` (SQLite `state.db`) | 💾 Memory/State ของ Core | session, history, FTS5 search, branch chains |

---

## 5. สัญญาที่ขอบเขต Core ↔ Action Runtime (สำคัญที่สุด)

ขอบเขตต้องเป็น message-passing ที่มี schema ชัด ไม่ใช่ function call ตรง ๆ ที่ผูกกันแน่น

### 5.1 ของจริงวันนี้: JSON-RPC 2.0 gateway

Hermes มี contract อยู่แล้วในรูป **newline-delimited JSON-RPC 2.0** over stdio (TUI) หรือ WebSocket (`/api/ws`, web/desktop ผ่าน `web/src/lib/gatewayClient.ts`) — dispatcher อยู่ที่ `tui_gateway/server.py::dispatch()` กระจายไป 74 handler ที่ decorate ด้วย `@method(name)` ครอบคลุม session lifecycle, turn + streaming, delegation/multi-agent (`delegation.*`, `spawn_tree.*`), human-in-loop (`clarify/sudo/secret/approval.respond`), model control, raw exec (`shell.exec`, `cli.exec`)

### 5.2 ⚠️ จุดอ่อนของสัญญาปัจจุบัน: failure หน้าตาเหมือน success — และมีหลาย convention

ปัญหาไม่ใช่ "ยังไม่มีจุดเริ่ม" แต่คือ **"มี convention ad-hoc หลายแบบใน handler เดียวกัน"** — ตรวจสอบที่ `slash.exec`:

> **อัปเดต 2026-06-10:** สามบูลเล็ตล่างคือสภาพ *ก่อน Phase 2* (แรงจูงใจให้รวม exec path) ตอนนี้ exec handler ทั้ง 3 route ผ่าน `action_runtime/adapters.py` แล้ว (§13) — เลขบรรทัดอัปเดตเป็นตำแหน่งปัจจุบัน

- **เคส side-effect (เช่น `/model`)** — slash side-effect path ที่ `tui_gateway/server.py:7506-7517`: `_mirror_slash_side_effects` (`:7354`) คืน `_SlashSideEffect` ที่ถือ field `.kind` (`:7314`); `_slash_side_effect_warning` (`:7343`) แปลง kind → ข้อความ แล้ว `slash_to_wire(result, warning)` (`:7517`) ตัดสินว่าจะแนบ `error` หรือไม่ คอมเมนต์ในโค้ด (`server.py:7511`) ยังเตือนว่า *"consumers (TUI, web slashExec) read only output/warning"* → field `error` inert สำหรับ client เก่า (additive ปลอดภัย)
- **เคส plugin command error** ที่ `server.py:7479-7491`: เดิมยัด error ลง `output` เป็น string ตอนนี้ route ผ่าน `plugin_to_wire(plugin_to_result(exc=e))` (`:7491`) คืน `output` (compat) + `error` (structured) พร้อมกัน
- ฝั่ง client เขียน defensive: `apps/desktop/src/app/session/hooks/use-model-controls.ts:66-84` แกะ `result.error` เองแล้ว `throw` เพื่อ rollback optimistic update

> ⚠️ **แก้ชื่อ helper (doc เดิมเขียนผิด):** ไม่มีฟังก์ชัน `_slash_sync_warning_is_failure` ในโค้ดจริง — ตัวที่ตัดสินคือ `_slash_side_effect_warning` (`:7343`) อ่านจาก `_SlashSideEffect.kind` (`'' | 'warning' | 'failure' | 'busy'`, `:7314`)

แปลว่า **"RPC สำเร็จ ≠ การกระทำสำเร็จ"** และแต่ละ exec path มีกติกาของตัวเอง → งานจริงของ Phase 1a คือ **"แทนที่ convention ad-hoc หลายแบบด้วย honest status เดียว แบบ additive"**

### 5.3 *(เสนอ)* `ExecutionTask` — สิ่งที่ Core ส่งให้ Action Runtime

```jsonc
{
  "task_id": "uuid",                    // correlate ผล + แก้ race (§8)
  "idempotency_key": "string",          // retry ซ้ำได้โดยไม่ทำงานซ้ำ (ยา Broken-pipe)
  "intent": "string",                   // intent ระดับสูง (§7) — ไม่ผูก tool ตรง ๆ
  "goal": "string | structured",        // เป้าหมายที่ต้องการให้สำเร็จ
  "inputs": { },                        // พารามิเตอร์
  "constraints": {                      // ขอบเขตที่ Runtime ห้ามละเมิด
    "timeout_s": 60,
    "network": "allow | deny",
    "filesystem": "ro | rw | none",
    "budget_tokens": 50000
  },
  "success_criteria": "string",         // Core จะตัดสิน "สำเร็จ" อย่างไร
  "context_ref": "session_id|memory_ptr", // Core ถือ memory; ส่ง "ตัวชี้" ไม่ใช่ทั้งก้อน
  "trace_id": "string"                  // observability ร้อยทั้ง loop (§12)
}
```

### 5.4 *(เสนอ)* `ExecutionResult` — สิ่งที่ Action Runtime คืนกลับ (แก้จุดอ่อน §5.2)

```jsonc
{
  "task_id": "uuid",                    // ตรงกับ task — client ทิ้งผล stale ได้ (§8)
  "status": "succeeded | failed | partial | needs_input",  // ❗ ชัด ไม่ใช่ ok เปล่า ๆ
  "outputs": { },
  "error": {                            // เมื่อ failed/partial
    "type": "timeout | denied | not_found | provider_error | transport | ...",
    "retryable": true,                  // Core ใช้ตัดสิน retry vs re-plan (§8)
    "message": "string"
  },
  "side_effects": [ ],                  // โลกภายนอกเปลี่ยนอะไรไปบ้าง
  "progress": [ ],                      // event ที่ stream ระหว่างทำ
  "needs_input": {                      // เมื่อ status = needs_input (human-in-loop)
    "kind": "approval | clarify | secret",
    "prompt": "string"
  }
}
```

**กฎทอง:** `status` ต้องสะท้อน "การกระทำจริง" ไม่ใช่ "การส่งมอบ message" และทุกการเปลี่ยน contract **ต้อง additive** (เพิ่ม field ข้างของเดิม) แล้ว migrate client แล้วค่อยลบของเก่า — เพราะมี 3 client (TUI, web, desktop) พึ่ง `output`/`warning` อยู่ (§12)

### 5.5 ผิว OpenClaw-compat bridge ที่มีอยู่ (`mcp_serve.py`)

`mcp_serve.py` เปิด 9 tool แบบ OpenClaw: `conversations_list, conversation_get, messages_read, attachments_fetch, events_poll, events_wait, messages_send, permissions_list_open, permissions_respond` (+ `channels_list`)

> **ข้อจำกัดที่ต้องรู้:** ผิวนี้เป็น **messaging-channel bridge** ไม่ใช่ generic execution bridge สัญญา `ExecutionTask`/`ExecutionResult` ของเรา **จงใจไม่ได้ออกแบบตามผิว 9-tool ของ OpenClaw** — คนละ concern กัน "OpenClaw compatibility" จำกัดเฉพาะผิว messaging นี้เท่านั้น

---

## 6. State & Memory: ใครเป็นเจ้าของอะไร

**หลักการ:** Core ถือ long-term context/memory · Action Runtime ควร **stateless ที่สุดต่อ 1 task** (idempotent) → recover/parallelize/retry ปลอดภัย

ของจริงวันนี้ (state กระจายและ persistent):

| ที่อยู่ | เก็บอะไร | ความเป็นเจ้าของตามโมเดล |
|---|---|---|
| `~/.hermes/state.db` (SQLite, `hermes_state.py`) | session metadata, message history, FTS5, branch chains, cron jobs | 💾 **Core memory** (authoritative) |
| `~/.hermes/` tree | `config.yaml`, `.env`, `memories/`, `skills/`, `cron/` | 💾 Core memory + config |
| in-memory ต่อโปรเซส | credential pool, tool schemas, context cache | ⚠️ **rebuilt ใหม่ในทุก AIAgent site (22)** |
| `_sessions` dict ใน `server.py` | agent instance, slash-worker, history + **lock ฝังในตัว dict** (`history_lock`, `agent_build_lock`) | 🦾 Runtime session state (monolithic) |
| spawn-tree on disk (`server.py` `_spawn_trees_root`) | multi-agent tree | แยกจาก `state.db` (ad hoc) |
| metadata caches | TTL caches + `context_length_cache.yaml`, `models_dev_cache.json` | 🦾 Runtime cache |
| frontend stores | Nanostores atoms, React Query, localStorage | 🖥️ ephemeral |

> **ปัญหาเชิงโครงสร้างที่กระทบ Phase 3:** lock ระดับ session (`history_lock`, `agent_build_lock`) ฝังอยู่*ใน* dict `_sessions` (อ้าง 78 ครั้ง; `history_lock`@2647, `agent_build_lock`@665) บวก module-level lock อีก 5 ตัว (`_sessions_lock`, `_prompt_lock`, `_session_resume_lock`, `_cfg_lock`, `_stdout_lock` — `:131-138`) → **ย้าย agent lifecycle ออกไม่ได้สะอาดถ้าไม่แตก `_sessions` เป็น `SessionState` ที่ถือ lock ของตัวเองก่อน** (ดู Phase 3)

---

## 7. Tool-selection: Core เลือก "tool ตรง ๆ" หรือ "intent ระดับสูง"?

- **แบบ A — Core เลือก tool ตรง ๆ** (วันนี้เป็นแบบนี้): coupling สูง สลับ runtime ยาก
- **แบบ B — Core ส่ง intent ระดับสูง, Runtime resolve tool เอง** *(เสนอ, เป้าหมาย Phase 4)*: coupling ต่ำ สลับ/อัปเกรด runtime ได้โดยไม่แตะ Core

**คำแนะนำ:** ค่อย ๆ ขยับไปแบบ B ที่ขอบเขตหลัก (ผ่าน `task.submit` ใน Phase 4) โดยคง tool-call ละเอียดไว้ภายใน Runtime ปัจจุบันชั้น model ก็สะท้อนปัญหานี้: `model_switch.py::switch_model` hardcode ลำดับ fallback ไม่มี policy hook → Core เลือกโมเดลตามต้นทุน/ความสามารถไม่ได้ ทำได้แค่ validate

---

## 8. Feedback loop & error handling

```
Core: ExecutionTask ──► Runtime ──► ExecutionResult{status, error.type, error.retryable, task_id}
  ▲                                          │
  └──────── ประเมิน → ตัดสินใจ ───────────────┘
            • succeeded            → ทำขั้นถัดไป
            • failed + retryable   → retry (ใช้ idempotency_key, ไม่ทำซ้ำ)
            • failed + !retryable  → เปลี่ยนแผน / ถามผู้ใช้
            • needs_input          → ส่ง human-in-loop prompt กลับขึ้นไป
            • partial              → เก็บ side_effects, วางแผนส่วนที่เหลือ
```

- **กรณี `Broken pipe`** (ที่ทำบทสนทนาก่อนหน้าหลุด) = `error.type: "transport"`, `retryable: true` → Core retry ด้วย `idempotency_key` เดิมโดยไม่ทำงานซ้ำ แทนที่จะล้มทั้ง turn
- **กรณี optimistic-update race** ที่ `use-model-controls.ts:66-84` (สอง model switch แข่งกันแล้ว rollback ผิดตัว) → `task_id` ใน `ExecutionResult` คือยาตรง ๆ: client ทิ้งผลที่ task_id ไม่ตรงกับ switch ล่าสุด
- human-in-loop วันนี้รองรับผ่าน `clarify/sudo/secret/approval.respond` (`threading.Event` + `_pending`/`_answers`) แต่ machinery กระจัดกระจาย ควรรวมเป็น prompt-lifecycle เดียว

---

## 9. ช่องว่างที่ต้องปิด (เรียงตามความสำคัญ)

1. **ไม่มี Core process เดียว — `AIAgent` ถูกสร้างใหม่ 22 จุด** (นับด้วย grep ยืนยัน 2026-06-10). entrypoint หลัก: `cli.py` ×2 (`:5220`,`:9354`), `gateway/run.py` ×4 (`:9293/12683/13211/17936`), `tui_gateway/server.py` ×3 (`:2609/5135/5246`), `gateway/platforms/` ×2 (`api_server.py:1010`, `feishu_comment.py:1074`), `cron/scheduler.py:1772`, `acp_adapter/session.py:619`, `batch_runner.py:325`, `hermes_cli/oneshot.py:335`, `agent/background_review.py:402`, `agent/curator.py:1753`, `tools/delegate_tool.py:1140`, `gateway/stream_consumer.py:86`, `hermes_cli/prompt_size.py:42` + non-production 2 จุด (`run_agent.py:5197` example, `scripts/tool_search_livetest.py:386`) → Core เป็น library ฝังในแต่ละ frontend ไม่ใช่ service
2. **ไม่มีทางเดิน tool-invocation เดียว.** แต่ละชนิดมี handler แยก (`shell.exec`=subprocess, `slash.exec`=worker subprocess, `cli.exec`=hermes_cli subprocess, plugin=direct call) + tool dispatch ภายใน `AIAgent`/`tools/registry.py` แยกอีก
3. **สัญญาไม่บอกผลจริง + มีหลาย convention.** (§5.2) — ต้องมี honest status เดียว
4. **CLI เป็นทั้ง interface และ runtime และถูก duplicate เข้า gateway.** `cli.py::HermesCLI` ถูกห่อใน `slash_worker.py` + command resolution ซ้ำเป็น `command.resolve`/`command.dispatch`
5. **Gateway ถือ agent lifecycle ไม่ใช่แค่ dispatch.** `_make_agent`, callback wiring, `session.steer`, `agent.switch_model` + state ก้อนเดียวต่อ session + lock ฝังใน dict
6. **ชั้น model ไม่มี policy / ไม่ pluggable** (urgency ต่ำ — สะอาดแต่แข็ง)
7. **Delegation/spawn-tree มีอยู่แต่ ad hoc.** `delegation.*`, `subagent.interrupt`, `spawn_tree.*` persist แยกจาก `state.db`

---

## 10. การตัดสินที่ทำแล้ว + ที่ยังเปิด

| # | ประเด็น | สถานะ |
|---|---|---|
| 1 | OpenClaw layer = (ก) เรียก OpenClaw จริง หรือ (ข) สร้าง native เอง | ✅ **ตัดสินแล้ว: (ข)** Action Runtime แบบ Hermes-native (§0.1) |
| 2 | ชื่อชั้น executor | ✅ **Action Runtime** |
| 3 | ชื่อชั้น brain | ✅ **Orchestration Core** (alias: Reasoning Core) |
| 4 | ขอบเขต Core/Runtime: in-process / RPC ในเครื่อง / network service? | 🔶 เปิด — ตัดสินตอน Phase 3 |
| 5 | Brain หนึ่งต่อ Runtime หนึ่ง หรือ Core หนึ่งสั่ง Runtime หลายตัว? | 🔶 เปิด — เป้าหมาย Phase 5 |
| 6 | `delegation.*`/`spawn_tree.*` จะกลายเป็นกลไก orchestration หรือไม่ | 🔶 เปิด — Phase 5 |
| 7 | Memory/skills อยู่กับ Core อย่างเดียว หรือ share กับ Runtime | 🔶 เปิด |

---

## 11. เส้นทางนำไปใช้ (ลำดับที่ล็อกแล้ว)

> ทำ incremental ไม่ rewrite — ปิด gap §9 ทีละข้อ **ไม่เริ่มจากงานเสี่ยงสุด (Core-as-service)**

### Phase 0 — Rename concept + glossary (doc-only, ทำได้เลย)
ตั้งศัพท์ (Orchestration Core / Action Runtime), เขียน glossary, ระบุว่า "OpenClaw" = compat/reference เท่านั้น, **จองชื่อ package `action_runtime/` ในเอกสาร แต่ยังไม่สร้างโฟลเดอร์** (โฟลเดอร์เปล่า = สัญญาเปล่า สร้างตอน Phase 2 พร้อม skeleton+contract+test)
- **Acceptance:** เอกสารนี้สะท้อนการตัดสินครบ · ไม่มีโค้ดถูกแตะ · ไม่มีโฟลเดอร์เปล่า

### Phase 1a — Additive honest status + tests
โฟกัสคำเดียว: **อย่าให้ failure หน้าตาเหมือน success** ทุก exec handler ที่วันนี้ซ่อน failure ใน `output`/`ok` ให้ surface `status: failed` + `error` **แบบเพิ่ม field ไม่ลบของเดิม**
- **Acceptance:** (1) plugin error (`server.py:7491`) + model-switch failure (`:7517`) คืน `status`/`error` structured · (2) client เก่า (TUI, web slashExec) **ยังอ่าน `output`/`warning` ได้เหมือนเดิม ไม่พัง** · (3) `tests/tui_gateway/test_protocol.py` ครอบ plugin-error path + model-switch-failure path · (4) desktop ไม่ rollback เพราะ stale

### Phase 2 + 1b — Unified Action Runtime path + rich Result schema
สร้าง abstraction เดียวใน `action_runtime/` (`ExecutionTask`/`ExecutionResult`) ให้ `shell/cli/slash.exec` + plugin command ค่อย ๆ delegate ผ่านทางเดียว, schema เต็ม (`outputs`/`side_effects`/`retryable`/`task_id`) ลงที่ choke point เดียว (ไม่ retrofit ทีละ handler)
- **Acceptance:** (1) มี unified path จริง · (2) RPC เดิมยังทำงาน frontend ไม่พัง · (3) test_protocol ครอบ unified path · (4) schema เต็มอยู่ที่เดียว

### Phase 3 — SessionState first → Brain host module (dual-path accepted)
**step แรก = แตก `_sessions` เป็น `SessionState` ที่ถือ lock ของตัวเอง** (ไม่ใช่ย้าย lifecycle ก่อน) แล้วค่อยย้าย agent lifecycle ออกจาก `tui_gateway/server.py` เป็น host module
- **Acceptance:** (1) `_sessions` ถูกแตกเป็น `SessionState`, lock ไม่ลอยใน dict · (2) มี host module และ `server.py` ใช้มัน · (3) **success metric = host module exists + server.py uses it** (ไม่ใช่ "gateway บางลง" — อีก ~17 AIAgent sites ยัง bypass อยู่ ถือเป็น **dual-path interim ที่ตั้งใจและ document ไว้**)

### Phase 4 — `task.submit` (intent-based, ทดลอง)
เพิ่ม method `task.submit`/`executor.run` รับ `task_id`/`idempotency_key`/`retryable`, Core ส่ง intent ระดับสูงได้ แต่ tool-call เดิมยังทำงานเหมือนเดิม
- **Acceptance:** (1) re-submit `idempotency_key` เดิม **ไม่ทำงานซ้ำ** · (2) pilot บน task ที่ reversible/read-only ก่อน (model switch ดี, shell side-effect เสี่ยงกว่า) · (3) intent-based ทำงานคู่ tool-call เดิม

### Phase 5 — Multi-runtime orchestration
รวม `delegation.*`/`spawn_tree.*` เป็น state model เดียว ให้ Orchestration Core สั่ง Action Runtime หลายตัวได้ (ปลายทาง ไม่ใช่จุดเริ่ม)

```text
Phase 0:    Rename concept + decide glossary + doc-only namespace
Phase 1a:   Additive honest status + tests
Phase 2/1b: Unified Action Runtime path + rich Result schema
Phase 3:    SessionState first, then Brain host module, dual-path accepted
Phase 4:    task.submit with task_id/idempotency_key/retryable
Phase 5:    multi-runtime orchestration
```

### Phase 3 — design detail *(จาก scoping fan-out, 2026-06-10 — ยังไม่ implement)*

**3a. SessionState** *(เสนอ: `tui_gateway/session_state.py`)* — dataclass ที่ **ถือ lock เป็น field จริง** แทนที่จะลอยใน dict `_sessions[sid]` (วันนี้มี ~30 keys: `agent`, `history`, `history_lock`, `agent_build_lock`, `agent_ready` Event, `running`, `inflight_turn`, `history_version`, `transport`, `session_key`, …) + helper methods ดูด lock-dance ออกจาก handler: `begin_turn()` (CAS busy-guard, แทน server.py:4307-4329), `end_turn()`, `snapshot_history()`, `commit_compaction(new, expected_ver)` (CAS แทน :1585-1592), `build_once()` (แทน agent_build_lock guard :665-669)

> **lock ที่ย้ายเข้า SessionState:** `history_lock`, `agent_build_lock`, `agent_ready`, `notif_stop` (per-session) · **lock ที่ STAY module-level:** `_sessions_lock` (guard ตัว map), `_prompt_lock`, `_session_resume_lock`, `_cfg_lock`, `_stdout_lock` (cross-session/process-global)

**3b. การ adopt แบบ incremental (กุญแจสำคัญ — handler ไม่พังระหว่างทาง):**
1. **SessionState เป็น `MutableMapping`** (delegate `__getitem__/__setitem__/get/setdefault/pop/__contains__` ไป `__dict__`) → construct ที่ 2 creation sites (`_init_session` :2643, `session.create` :3071) แทน dict literal **ทุก `session["history"]` เดิมทำงานต่อได้ทันที zero churn** — flip type ได้โดย codebase รัน/compile ต่อได้เลย
2. ย้าย lock-core ก่อน (payoff สูง blast แคบ): turn lifecycle + compaction + build guard → methods
3. แปลง subscript → attribute access (`session['history']` → `session.history`) ทีละ handler-group (shim ให้ทั้งสองสไตล์อยู่ร่วมกัน)
4. เมื่อไม่มี subscript เหลือ → ทิ้ง shim, `_sessions: dict[str, SessionState]`
5. **Brain host half** — ย้าย agent-build + turn-exec ไปหลัง host

**3c. BrainHost** *(เสนอ: `agent/brain_host.py`)* — singleton factory ถือ state ที่แพง: `CredentialPoolRegistry` (pool ต่อ provider/profile สร้างครั้งเดียว), `ToolSchemaCache` (MCP discovery + tool-definition memoize ต่อ toolset signature), `MemoryProviderSessions` API: `BrainHost.get().build_agent(AgentSpec) -> AIAgent` (เรียก constructor `AIAgent` เดิม inject heavy objects — additive, ไม่แก้ constructor) · **AIAgent ถูกสร้าง 22 จุด** (gateway/cli/cron/platforms/oneshot/delegate/acp/batch/review/curator)

**3d. first site + dual-path:** route `tui_gateway/server.py:2609 (_make_agent)` ก่อน (funnel เดียว, ไม่มี LRU cache/streaming fan-out, dogfood ผ่าน TUI/desktop) · **gate ด้วย flag** (`HERMES_BRAIN_HOST=1`, default off) → rollback ทันที · อีก ~19 sites คงเดิม = **dual-path interim ที่ documented** + parity test (hosted vs direct agent เทียบ model/toolset/pool identity) · success metric = "host module มี + server.py ใช้" ไม่ใช่ "gateway บาง"

> **เหตุที่ยังไม่ลงมือ:** Phase 3 คือ gap #1 และเปลี่ยน concurrency model ที่ 74 handlers พึ่ง — ควรทำเป็น cycle เฉพาะที่ review design ก่อน ไม่ใช่รีบ ขั้นที่ปลอดภัยสุดคือ 3b-Step1 (MutableMapping shim)

---

## 12. หลักการข้ามทุก phase (Cross-cutting)

1. **Additive-first compatibility** — ห้ามทำ client 3 ตัว (TUI/web/desktop) พัง: เพิ่ม field ใหม่ข้างของเดิม → migrate client → ค่อยลบของเก่า (โค้ดเตือนเองที่ `server.py:7511`)
2. **Test-first** — ทุกการเปลี่ยน contract ต้อง anchor ด้วย `tests/tui_gateway/test_protocol.py` ก่อนแก้ handler
3. **Observability / trace** — ร้อย `trace_id` จาก `ExecutionTask`→`ExecutionResult` + structured decision events (วันนี้ "ไม่มี trace hook บอกไม่ได้ว่าทำไมเลือกโมเดล/ทำไม fallback") เพิ่มตอนออกแบบ contract แทบฟรี retrofit ทีหลังแพง
4. **Idempotency** — `task_id` + `idempotency_key` + `error.retryable` เป็น first-class ใน contract ตั้งแต่ Phase 1 (แม้ยังไม่ใช้เต็ม) = ยาตระกูล `Broken pipe`

---

## 13. Decision Log

| วันที่ | การตัดสิน |
|---|---|
| 2026-06-09 | สร้างเอกสาร v1 — สำรวจโค้ด, เสนอโมเดล Central Brain + OpenClaw |
| 2026-06-10 | **v2**: §10-Q1 = **(ข)** native Action Runtime, ยืมแค่ OpenClaw messaging-bridge compatibility · ล็อกศัพท์ **Orchestration Core** / **Action Runtime**; "OpenClaw" = reference/compat เท่านั้น · ล็อกลำดับ phase (0→1a→2/1b→3→4→5) · Phase 0 = doc-only, ไม่สร้างโฟลเดอร์เปล่า · เพิ่ม cross-cutting (compat/test-first/observability/idempotency) + acceptance criteria ต่อ phase |
| 2026-06-10 | **Phase 3 design delivered (ยังไม่ implement).** Scoping fan-out 3 agents → `SessionState` dataclass (lock เป็น field) + MutableMapping-shim incremental adoption (Step1 = zero handler churn) + `BrainHost` (`agent/brain_host.py`, flag-gated `HERMES_BRAIN_HOST`, additive) + map 20 AIAgent sites (first=`_make_agent`) + documented dual-path + parity test. เก็บใน §11 "Phase 3 — design detail". **Implementation ถือไว้เป็น cycle เฉพาะ** — decompose live session-state + locks ที่ 74 handlers + concurrency model พึ่ง (gap #1, เสี่ยงสุดในแผน) ไม่ควรรีบท้าย session |
| 2026-06-10 | **Phase 3 Step 2 landed — turn-lifecycle lock dances absorbed into SessionState.** 5 methods (`snapshot_history`/`commit_compaction`/`begin_turn`/`end_turn`/`build_once`) เป็น verbatim lift ของ block เดิม; 5 gateway sites migrate, sites ที่ critical section พ่วงงานอื่น (prompt.submit busy-gate+truncate, poller gates, mid-turn CAS ที่ log ใต้ lock) **คง inline โดยตั้งใจ** พร้อม note เหตุผล atomicity. 9 unit tests pin contract. suites: tui_gateway 165 + server 208 + combined 278 green |
| 2026-06-10 | **Phase 4 landed (pilot).** (1) `task.submit` RPC — intent="slash" เท่านั้นตามแผน pilot (reversible), intent อื่น → 4030; slash.exec core ถูก extract เป็น `_slash_exec_core` ใช้ร่วมกัน (legacy wire เดิม byte-identical / rich wire ใหม่). (2) **Idempotency**: in-process store (TTL 10 นาที, cap 1024) — re-submit key เดิมคืนผลเดิม + `replayed:true` ไม่รันซ้ำ. (3) `result_to_wire_rich` ครบ contract (`status`/`error.retryable`/`side_effects`/`task_id`). (4) `task_id` echo บน slash.exec แบบ additive. (5) **Desktop race fix**: in-flight token — stale switch ไม่ commit/ไม่ rollback (เก็บ error toast), race tests 2 ตัว. acceptance ครบทั้ง 3 ข้อของ §11 Phase 4 |
| 2026-06-10 | **Phase 3 Step 1 landed — SessionState type introduced (committed).** `tui_gateway/session_state.py`: `class SessionState(dict)` + typed accessors (history/history_lock/running/agent/…). Wrap 2 real creation literals (`_init_session`, `session.create`). **Zero behavioral change** (dict subclass — safety fan-out ยืนยัน: ไม่มี membership/bare-pop/iteration ที่พึ่ง session value dict, absent key คงหายไปจริง → ไม่ต้อง tune default; `agent_ready` คง absent บน eager path, `personality` คง config fallback). session lifecycle 208 + protocol 65 + SessionState unit 4 green. **Next: Step 2** (ย้าย history_lock turn-lifecycle + compaction CAS เข้า SessionState methods — concurrency refactor, ทำเป็น cycle เฉพาะ) |
| 2026-06-10 | **Phase 2 (exec handlers) complete — full honest-status across all exec paths.** slash.exec migrated ผ่าน Action Runtime (plugin path → `plugin_to_*`; worker+side-effect path → `slash_to_*` ใช้ `_SlashSideEffect.kind` ตามเดิม) แบบ byte-identical — model-switch failure/benign + plugin-error protocol tests เขียว (handler จริงผ่าน adapter = byte-compat oracle). ครบ 3 exec handlers (shell/cli/slash) + plugin funnel ผ่าน `ExecutionResult` schema เดียว. action_runtime 13 + protocol 65 + server-alone 208 green (เหลือ browser-env fail เดิม). Scope: RPC state-mutation อื่น (command.dispatch, session.compress/branch/resume, config.set, browser/skills.manage) **อยู่นอก Phase 2** ตาม surface map (เป็น state-mutation ไม่ใช่ exec). consumer ของ rich fields (retryable/idempotency/task_id race-fix) = Phase 4. ยังไม่ commit |
| 2026-06-10 | **Phase 2 started — Action Runtime contract + stateless exec handlers migrated.** สร้าง `action_runtime/` จริง (Phase 0 จองชื่อไว้, ตอนนี้มี skeleton+contract+test ตามเงื่อนไข): `contract.py` (`ExecutionTask`/`ExecutionResult`/`Status`/`ErrorType`/`ExecError`/`SideEffect`/`NeedsInput`/`Constraints`) + `adapters.py` (shell, cli) + `tests/action_runtime/` (round-trip byte-compat). Migrate **shell.exec + cli.exec** ผ่าน adapter แบบ **byte-identical** (wire ไม่ขยับ — guarded by Phase 1a tests + round-trip tests; truncation/`(no output)` คงไว้ใน handler ตาม wire contract). action_runtime 7 + protocol 65 green. design มาจาก fan-out 3 agents (surface map ~40 handlers, contract, migration). **ถัดไป: slash.exec** (exec handler สุดท้าย — เสี่ยงสุด เพราะแตะ live agent ผ่าน `_SlashSideEffect`; migration plan ให้ทำ last). ยังไม่ commit |
| 2026-06-10 | **Phase 1a complete (gateway exec paths).** plugin `slash.exec` failure now additive (`result.error`, `server.py:7471`) + `shell.exec`/`cli.exec` confirmed already-honest (real exit `code` / `_err`) พร้อม regression tests ใหม่ — full protocol suite 65 green. หมายเหตุ: structured-kind refactor ของ `_mirror_slash_side_effects` (string-prefix → `_SlashSideEffect.kind`, ตรงกับ hardening follow-up ที่ spawn ไว้) **ถูก user apply ลง working tree แล้ว** (task started). สอบสวน 5 failures ใน combined run (test_protocol + test_tui_gateway_server) → ยืนยัน **pre-existing 100% ที่ HEAD** (4 = cross-file test pollution, 1 = browser env) ไม่ได้เกิดจากงานนี้. ยังไม่ commit. **ถัดไป: Phase 2** |
| 2026-06-10 | **Phase 1a (model-switch slice) — landed + verified.** ลาก WIP model-picker/honest-status ที่ค้างอยู่ให้เขียว: `error` field แบบ additive บน `slash.exec` failures + client 3 ตัว consume (desktop picker rollback, desktop typed `/model`, web picker/toast). แก้ bug จาก review 6 จุด (keyless-local เขียนทับ explicit `models:` → 2 test regressions; Gemma4-26B cache re-probe thrash; bypass ทิ้ง stale-fallback; desktop typed-`/model` รายงาน fail เป็น success; `hermes` venv-version re-check; gateway PID-reuse SIGKILL) + WIP test regression (launchd SIGUSR1 path) พร้อม coverage ใหม่ Suite เขียว (protocol 60, model layer 286, gateway launchd 25); failures ที่เหลือเป็น environment ล้วน (macOS ไม่มี systemd, live ollama บนเครื่อง, ไม่มี pytest-asyncio). **ยังไม่ generalize** honest-status ไป plugin/shell/cli exec paths (ส่วนที่เหลือของ Phase 1a). การแก้อยู่ใน working tree ยังไม่ commit |
| 2026-06-10 | **Doc verification pass (fan-out 6 agents, grep/wc).** เช็คทุก claim กับโค้ดจริง — **ตรง:** Core 5 component, 74 `@method`, action_runtime wire 4 site (`cli@6251`/`plugin@7479`/`slash@7513`/`shell@8542` byte-identical), 13 test, 5 module-lock (`:131-138`), `_spawn_trees_root@4087`, mcp 10 tool, README migration `:147-173`. **แก้ที่คลาดเคลื่อน:** `cli.py` AIAgent ×3→**×2**; "~20+"→**22 จุด** (เติม 5 site ที่ตกหล่น: `prompt_size.py:42`/`stream_consumer.py:86`/`curator.py:1753`/`run_agent.py:5197`/`scripts/tool_search_livetest.py:386`); `_sessions` 58→**78** ref; `server.py` 8,498→**8,551**; slash.exec line refs `7438/7451/7467/7471`→ปัจจุบัน plugin@`7491`/slash@`7517` (route ผ่าน action_runtime); helper `_slash_sync_warning_is_failure` **ไม่มีจริง**→`_slash_side_effect_warning@7343`+`_SlashSideEffect.kind@7314`. เพิ่ม §2.1 skeleton tree (verified) |

---

## ภาคผนวก — ดัชนีไฟล์อ้างอิงหลัก

| ไฟล์ | บทบาท |
|---|---|
| `run_agent.py` | `AIAgent` — Orchestration Core |
| `agent/conversation_loop.py` | turn loop ที่แยกออกมา |
| `agent/curator.py` | self-improvement / memory policy |
| `cron/scheduler.py` | spawn ตามเวลา |
| `cli.py` | `HermesCLI` — frontend + runtime + agent host (จุดปนกัน) |
| `tui_gateway/server.py` | JSON-RPC dispatcher + agent lifecycle (74 handler, 8,551 บรรทัด); `slash.exec`@`:7420` (plugin@`:7491`, side-effect@`:7517` — route ผ่าน `action_runtime`) |
| `tui_gateway/entry.py`, `transport.py`, `slash_worker.py` | event loop / transport / subprocess executor |
| `tools/registry.py`, `providers/` | tool + provider transport |
| `agent/model_metadata.py`, `hermes_cli/models.py`, `hermes_cli/model_switch.py` | ชั้น model (Runtime plumbing) |
| `mcp_serve.py` | OpenClaw-compat 9-tool MCP bridge (messaging) |
| `hermes_state.py` | SQLite `state.db` — Core memory |
| `apps/desktop/src/app/session/hooks/use-model-controls.ts` | desktop model control (`:66-84` optimistic + rollback, หลักฐาน `ok-but-error`) |
| `tests/tui_gateway/test_protocol.py` | protocol test (anchor ของทุก contract phase) |
| `README.md:147-173` | "Migrating from OpenClaw" (OpenClaw ความหมาย reference) |

---

*เอกสารนี้สังเคราะห์จากการสำรวจโค้ด Hermes จริง ส่วนที่เป็นข้อเสนอ (schema, เส้นทาง) ยังไม่ได้ implement และเปิดให้ทีมแก้ไข*
