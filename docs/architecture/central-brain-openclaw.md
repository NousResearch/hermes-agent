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

### Phase 5 — design detail *(จาก scoping fan-out, 2026-06-10 — ยังไม่ implement)*

**แกนของ design: `AgentTaskRegistry` เดียว keyed ด้วย `task_id`** — รวม `_active_subagents` (delegate_tool, in-process) + spawn-tree files (TUI-assembled snapshots) + `_TASK_RESULTS` (idempotency ของ task.submit) เข้า model เดียวที่ Orchestration Core query/command ผ่าน contract เดิม (`ExecutionTask`/`ExecutionResult`)

#### สถานะปัจจุบัน (grounded)
## State locations, locks, and mechanics — grounded

### 1. Live subagent registry (in-process, ephemeral)

File: `/tmp/hermes-wt-sites/tools/delegate_tool.py`

- `_active_subagents: Dict[str, Dict[str, Any]]` at line 157 — module-level dict, keyed by `subagent_id` (format `sa-{index}-{8hex}`, generated at line 954).
- Guard: `_active_subagents_lock = threading.Lock()` at line 154. Both are process-global, spanning every `delegate_task` call in the process.
- Registration: `_register_subagent(record)` at line 177, called inside `_run_single_child` at line 1502, just before `_heartbeat_thread.start()`.
- Each record contains: `{subagent_id, parent_id, depth, goal, model, started_at, status, tool_count, agent}` — the `agent` key holds the live `AIAgent` reference (stripped before external exposure in `list_active_subagents()` at line 213).
- Unregistration: `_unregister_subagent` at line 185, called in the `finally` block of `_run_single_child` at line 1906 — guaranteed even on exception/timeout.
- Lifetime: **purely in-process**. The registry is empty at process start and has no relation to `state.db` or the spawn-tree files.

### 2. Spawn/pause flag

- `_spawn_paused: bool = False` at line 152, guarded by `_spawn_pause_lock` at line 151.
- Read by `delegate_task` at line 2001 (fast-fail before any spawn).
- Written via `set_spawn_paused()` at line 160.
- RPC bridge: `delegation.pause` at server.py line 4458 calls `set_spawn_paused`.
- No persistence; reset to `False` on every process restart.

### 3. Gateway RPC surface for delegation/spawn-tree

All in `/tmp/hermes-wt-sites/tui_gateway/server.py`:

- `delegation.status` (line 4438): returns `list_active_subagents()`, `is_spawn_paused()`, `_get_max_spawn_depth()`, `_get_max_concurrent_children()`. Read-only, zero side effects.
- `delegation.pause` (line 4458): calls `set_spawn_paused(bool)`.
- `subagent.interrupt` (line 4466): calls `interrupt_subagent(subagent_id)` in delegate_tool. That function reads `_active_subagents` under lock, gets the `agent` reference, and calls `agent.interrupt(...)`.
- `spawn_tree.save` (line 4539): TUI-written snapshot. Accepts `{session_id, subagents[], started_at, finished_at, label}` — writes a JSON file to `$HERMES_HOME/spawn-trees/<session_id>/<timestamp>.json` and appends a lightweight entry to `_index.jsonl` for O(1) list.
- `spawn_tree.list` (line 4582): reads `_index.jsonl`; falls back to directory scan for pre-index sessions. Accepts `cross_session` flag. Returns metadata entries (no full payload).
- `spawn_tree.load` (line 4633): returns the full JSON payload of a single snapshot, path-validated against `_spawn_trees_root()`.

### 4. On-disk spawn-tree persistence

- Root: `get_hermes_home() / "spawn-trees"` — constructed in `_spawn_trees_root()` at server.py line 4486.
- One subdirectory per session (alphanumeric-sanitized session_id).
- One JSON file per completed turn, named `<YYYYmmddTHHMMSS>.json`.
- Append-only index: `_index.jsonl` per session directory.
- **Completely separate from `state.db`** (confirmed: `hermes_state.py` has no `spawn_tree`, `subagent`, or `delegation` tables — only `sessions`, `messages`, `state_meta`, `compression_locks`).
- The TUI is the source of truth for what goes into spawn-tree files; the gateway merely writes what it is handed. The active-registry in `delegate_tool` is NOT persisted here — these files record completed-turn snapshots for replay/diff, not live state.

### 5. Subagent spawning mechanics (delegate_tool.py)

- Entry: `delegate_task(tasks, ...)` at line ~1985. Guards: `is_spawn_paused()` (line 2001), depth check against `_get_max_spawn_depth()` (line 2014), count check against `_get_max_concurrent_children()` (line 2054).
- Child construction: `_build_child_agent(...)` at line 904, called on the main thread (thread-safe). Creates a full `AIAgent` at line 1140. Assigns `child._subagent_id`, `child._parent_subagent_id`, `child._delegate_depth`, `child._delegate_role`.
- child is appended to `parent_agent._active_children` (list initialized in `agent/agent_init.py` line 466, guarded by `_active_children_lock` at line 467) — this is the per-AIAgent tree used for recursive `agent.interrupt()` propagation.
- Execution: single task runs `_run_single_child` directly; batch tasks run in `ThreadPoolExecutor(max_workers=max_children)` at line 2154. Each worker thread runs inside a nested `ThreadPoolExecutor(max_workers=1)` with a hard timeout (line 1544).
- `child_task_id` at line 1534 = `_subagent_id` (reuses the same key so `file_state`, `_active_subagents`, and TUI events share one identifier).
- `child.run_conversation(user_message=goal, task_id=child_task_id)` at line 1559.

### 6. Interrupt propagation tree

- `AIAgent.interrupt()` at run_agent.py line 2278: sets `self._interrupt_requested = True`, propagates to all `_active_children` recursively (line 2336-2340).
- `_active_children`: list on `AIAgent`, guarded by `_active_children_lock`, initialized in `agent/agent_init.py:466-467`.
- Two registries track the same running children: `_active_subagents` in delegate_tool (module-level, keyed by `subagent_id`, TUI-facing) and `parent._active_children` (per-AIAgent list, interrupt-propagation). They are populated/cleared independently with no shared synchronization.

### 7. task.submit pilot (Phase 4)

- `task.submit` at server.py line 8507. Only `intent="slash"` is accepted.
- Idempotency: in-process dict `_TASK_RESULTS` at line 8482, keyed by `idempotency_key`, TTL 600s, cap 1024. Not persisted to state.db.
- Execution: delegates to `_slash_exec_core` at line 8539, which builds an `ExecutionResult` via adapters and returns `result_to_wire_rich(result)`.
- The `task_id` supplied by the caller (or auto-generated uuid4) is echoed in the rich result. It is passed to the adapter but NOT written to any persistent store — it is a request-scoped identifier only.
- No `task.cancel`, `task.status`, or `task.list` RPC exists yet.

### 8. ExecutionTask/ExecutionResult contract (action_runtime/)

- `action_runtime/contract.py`: `ExecutionTask` (task_id, idempotency_key, intent, goal, inputs, constraints, success_criteria, context_ref, trace_id) and `ExecutionResult` (task_id, status, outputs, error, side_effects, needs_input, trace_id).
- `action_runtime/adapters.py`: per-handler adapters (shell, cli, plugin, slash) map native shapes to `ExecutionResult` and back to wire-identical dicts.
- Current users: `shell.exec`, `cli.exec`, `slash.exec`, `task.submit`. The delegation/spawn-tree surface has **no adapter** and is not routed through `ExecutionResult`.

### 9. Gap — the two registries are disconnected

`delegation.*` and `spawn_tree.*` operate on completely separate state from `task.submit`/`ExecutionResult`:

| Dimension | delegation/spawn-tree world | task.submit / Action Runtime world |
|---|---|---|
| State location | `_active_subagents` dict in delegate_tool module + JSON files in spawn-trees/ | `_TASK_RESULTS` in-process dict in server.py |
| Key | `subagent_id` (`sa-{i}-{8hex}`) | `task_id` (uuid4 or caller-supplied) |
| Identity overlap | `child_task_id = _subagent_id` (line 1534) — same value used for file_state and `run_conversation` | task_id echoed in rich result wire only |
| Persistence | Spawn-tree JSON files (post-turn snapshots only, no live state) | None (in-process TTL dict) |
| Lifecycle | Registered at thread-start, unregistered in finally block | Stored after execution, expired by TTL |
| Interrupt mechanism | `interrupt_subagent(subagent_id)` → `agent.interrupt()` | No cancel/interrupt RPC |
| Pause mechanism | `_spawn_paused` module global | None |
| Core visibility | TUI polls via `delegation.status` | Caller reads rich result |

#### ข้อเสนอ
## Unified multi-runtime state model for Phase 5

### Core idea

Replace the two disconnected tracking structures with a single **`AgentTaskRegistry`** keyed uniformly by `task_id`, where every live subagent is represented as an `ExecutionTask` (in-flight) and completed ones produce a persistent `ExecutionResult`. The Orchestration Core interacts with any number of Action Runtime instances through the same contract it already uses in Phase 4.

---

### Proposed data model

**`AgentTaskRecord`** (new, in `action_runtime/task_registry.py` — Step 1 landed):

```python
@dataclass
class AgentTaskRecord:
    task_id: str                  # = current subagent_id (sa-{i}-{8hex}) — no rename needed
    parent_task_id: Optional[str]
    depth: int                    # 0 = root parent; 1 = first-level child; etc.
    goal: str
    model: Optional[str]
    started_at: float
    status: TaskStatus            # RUNNING | SUCCEEDED | FAILED | PARTIAL | BLOCKED | NEEDS_INPUT
    agent_ref: Optional[Any]      # weakref to AIAgent — excluded from serialization
    tool_count: int
    last_tool: Optional[str]
    result: Optional[ExecutionResult]  # set on completion
    idempotency_key: Optional[str]
```

**`TaskStatus`** (Q3 ruling, option B): a **separate enum on the record** in `task_registry.py` — `RUNNING` plus mirrors of the five terminal values. `contract.py`'s `Status` is **untouched**: every `ExecutionResult` stays a terminal answer, so all existing adapters/clients/replay logic keep their invariant. Interrupted subagents map to `FAILED` with `ErrorType.TRANSPORT` (retryable=False, message="interrupted").

---

### Single registry

**`AgentTaskRegistry`** — a module-level singleton (or held by `BrainHost` when that is adopted):

```
_registry: Dict[str, AgentTaskRecord]
_registry_lock: threading.Lock
```

- `register(record: AgentTaskRecord)` — called at the same point as today's `_register_subagent`.
- `complete(task_id, result: ExecutionResult)` — transitions to terminal status, stores `ExecutionResult`, removes `agent_ref`.
- `interrupt(task_id) -> bool` — looks up `agent_ref`, calls `agent.interrupt()` (same mechanic as today's `interrupt_subagent`).
- `list_active() -> List[AgentTaskRecord]` — all records where `status == RUNNING`.
- `get(task_id) -> Optional[AgentTaskRecord]` — used by Core to query any task.
- `pause_spawns(bool)` — replaces the module-global `_spawn_paused` / `_spawn_pause_lock`.

Existing `_active_subagents` and `_spawn_paused` become thin wrappers (or are removed) delegating to this registry.

---

### Persistence: spawn-tree files fold into a registry-backed EventLog

Today's spawn-tree files are post-turn snapshots assembled by the TUI. In Phase 5, the registry writes them instead, keyed by task_id:

- **Hot path**: registry remains in-process (same as today).
- **Persistence**: on `complete()`, append one line to `$HERMES_HOME/spawn-trees/<session_id>/_tasks.jsonl` — a minimal `AgentTaskRecord` snapshot (no `agent_ref`). This replaces the TUI-assembled `spawn_tree.save` flow (the TUI can still call `spawn_tree.save` for UI-assembled rich snapshots; the registry is additive).
- **`state.db` vs spawn-tree files**: do NOT move to state.db. Spawn-tree data has no FK relationship to the `sessions` table that wouldn't require schema changes and a migration. Keep files. If state.db integration is ever needed (search, joins), a future phase can import the JSONL into a `subagent_tasks` table.
- **Idempotency store** (`_TASK_RESULTS` in server.py): fold it into the registry. When `complete()` is called with a record that has an `idempotency_key`, the registry stores the `ExecutionResult` and services replay. TTL eviction stays.

---

### How interrupt and pause map to the contract

| Current mechanism | Phase 5 equivalent |
|---|---|
| `interrupt_subagent(subagent_id)` → `agent.interrupt()` | `registry.interrupt(task_id)` → same `agent.interrupt()` on the stored `agent_ref` |
| `_spawn_paused` global flag, checked in `delegate_task` | `registry.pause_spawns(bool)`, same check point |
| `agent._active_children` list for recursive interrupt | unchanged — this is within `AIAgent`; registry does not replace it |
| TUI `subagent.interrupt` RPC | calls `registry.interrupt(task_id)` instead of `interrupt_subagent` |
| No cancellation for `task.submit` tasks | new `task.cancel` RPC: calls `registry.interrupt(task_id)` if record exists and is RUNNING |

---

### How Core commands multiple runtimes

Phase 5 makes the Orchestration Core runtime-agnostic by routing all task creation through `task.submit`:

1. Core calls `task.submit` with `intent="delegate"` (new intent, extending the Phase 4 pilot).
2. Gateway handler extracts the task list from `inputs.tasks`, creates an `ExecutionTask` per subagent, and calls `delegate_task` (the existing engine).
3. Each spawned child agent gets a `task_id` = its `subagent_id` (already the case at delegate_tool.py:1534).
4. Registry records every child as a live `AgentTaskRecord`.
5. Core can poll `task.status` (new RPC) or receive push events; on failure, the `ExecutionResult.error.retryable` flag tells it whether to retry.

This means Core issues one `task.submit` RPC and observes N `ExecutionResult` objects — the same contract shape it uses for single-task intents. It never calls `delegation.status` or `spawn_tree.*` directly; those remain as lower-level TUI observability tools backed by the same registry.

---

### What does NOT change

- `delegate_task`'s internal engine (ThreadPoolExecutor, child AIAgent construction, heartbeat, timeout) is untouched.
- The `_active_children` list on `AIAgent` (interrupt propagation) is untouched.
- Wire shape of `delegation.status`, `spawn_tree.*`, `subagent.interrupt` responses — all stay byte-identical (registry is the new backing, same data).
- `state.db` schema — no new tables.
- The `task.submit` idempotency protocol — generalized to also cover delegated tasks.

#### Migration steps
## Migration steps — incremental, additive-first, each verifiable

### Step 1 — Introduce `AgentTaskRegistry` alongside existing code (additive, zero behavior change)

Files: new `action_runtime/task_registry.py` (or `tools/agent_task_registry.py`).

- Define `AgentTaskRecord` dataclass and `AgentTaskRegistry` class with `register`, `complete`, `interrupt`, `list_active`, `pause_spawns`, `get`.
- Define `TaskStatus` enum (RUNNING + terminal mirrors) in `task_registry.py` — `action_runtime/contract.py` is untouched per the Q3 ruling (option B).
- Write unit tests: register/complete/interrupt round-trip, pause flag, TTL eviction for idempotency keys.
- **Verify**: tests green; no handler touched; grep confirms `_active_subagents` still used in production path.

### Step 2 — Dual-write: `delegate_tool` registers into both old dict and new registry

File: `tools/delegate_tool.py`.

- In `_run_single_child`, after the existing `_register_subagent(...)` call (line 1502), also call `registry.register(AgentTaskRecord(...))`.
- In the `finally` block (line 1906), after `_unregister_subagent`, call `registry.complete(task_id, result=None)` for timeout/interrupt paths or `registry.complete(task_id, result=make_result(entry))` for normal exits.
- `make_result(entry)` builds an `ExecutionResult` from the existing `entry` dict (status/summary/error → Status/ExecError).
- **Verify**: `delegation.status` still returns same data (still reads `_active_subagents`); new registry has matching entries; no functional change to TUI.

### Step 3 — Route `subagent.interrupt` and `delegation.pause` through the registry

File: `tui_gateway/server.py`.

- `subagent.interrupt` (line 4466): change from `interrupt_subagent(subagent_id)` to `registry.interrupt(subagent_id)` — internally still calls `agent.interrupt()`.
- `delegation.pause` (line 4458): change from `set_spawn_paused` to `registry.pause_spawns`.
- Keep `_spawn_paused` and `_active_subagents` populated (still dual-write from Step 2).
- **Verify**: TUI pause/interrupt still works; existing protocol tests green.

### Step 4 — Add `task.status` and `task.cancel` RPCs backed by the registry

File: `tui_gateway/server.py`.

- `task.status`: looks up `registry.get(task_id)`, returns `AgentTaskRecord` serialized (status, goal, depth, started_at, model, tool_count, result if terminal).
- `task.cancel`: calls `registry.interrupt(task_id)`, returns `{found, task_id}` — same shape as `subagent.interrupt`.
- Add to `_DISPATCH_LIST` (line 186 area).
- Write protocol tests (round-trip, cancel of non-existent task returns `found: false`).
- **Verify**: new RPCs work; no existing RPC changed.

### Step 5 — Extend `task.submit` to accept `intent="delegate"`

File: `tui_gateway/server.py`.

- In `task.submit` handler (line 8507), allow `intent="delegate"` (currently rejected with 4030).
- Extract `inputs.tasks` list, call `delegate_task` (same engine), wrap aggregate result as `ExecutionResult(status=SUCCEEDED|FAILED|PARTIAL, outputs={"results": [...]}, ...)`.
- Use the caller-supplied `task_id` as the parent task_id; each child gets its `subagent_id` as its own `task_id` in the registry.
- Fold `_TASK_RESULTS` idempotency into registry (registry stores `ExecutionResult` on `complete()`; `task.submit` checks `registry.get(idempotency_key)` first).
- **Verify**: `intent="delegate"` returns rich `ExecutionResult`; `intent="slash"` still returns same output (no regression); existing idempotency replay tests green.

### Step 6 — Registry-backed spawn-tree persistence (replace TUI-assembled saves as additive path)

File: `action_runtime/task_registry.py`, `tui_gateway/server.py`.

- On `registry.complete()`, append a `AgentTaskRecord` snapshot (no `agent_ref`) to `$HERMES_HOME/spawn-trees/<session_id>/_tasks.jsonl`.
- `spawn_tree.list` reads both `_index.jsonl` (legacy) and `_tasks.jsonl` (new), merges, deduplicates by path/task_id.
- `spawn_tree.save` (TUI path) continues to work unchanged — the TUI's rich assembled payload overwrites nothing; the two files coexist.
- **Verify**: `spawn_tree.list` returns entries from both sources; `spawn_tree.load` still works on legacy snapshot files; new `_tasks.jsonl` entries appear after subagent runs.

### Known gaps deferred from the Steps 1-6 adversarial review (2026-06-11)

ของจริงที่เจอจาก review 3-lens แต่**ตั้งใจเลื่อน** (แก้ตอน Step 7 หรือเมื่อ engine รองรับ) — ไม่ใช่ลืม:

1. **Children ของ `task.submit intent="delegate"` ยังไม่ link `parent_task_id` = caller task_id** ตามที่ Step 5 ร่างไว้ — child record ใช้ `_parent_subagent_id` ของ engine; การ link ต้อง plumb task_id ผ่าน `delegate_task` ซึ่งแตะ engine (ขัด "What does NOT change") → รอ Step 7 cutover.
2. **Q4 sub-clause `side_effects` ของ children ที่จบแล้วใน PARTIAL ยังว่าง** — engine entry ปัจจุบันไม่ surface side-effect data ต่อ child; ไม่ fabricate. เพิ่มเมื่อ engine ส่งข้อมูลนี้ขึ้นมา.
3. **`spawn_tree.load` บน path ของ registry-only entry** (ชี้ไป `_tasks.jsonl`) คืน graceful 5000 บน client เก่า — entry ใหม่มี `source` key ให้ client ใหม่แยกได้; UX fix รอรอบถัดไป.
4. **Pool starvation เกิน R6**: `intent="delegate"` ถือ worker ของ tui-rpc pool (default 4, floor 2) ได้นานระดับนาที — สอง batch พร้อมกันบน pool=2 บล็อก `session.resume`/`slash.exec` ได้. Documented บน handler แล้ว; mitigation จริง (dedicated delegate pool / async handoff) เป็นงาน Step 7+.

### Step 7 — Remove old module-globals (cutover, after Step 6 is stable)

File: `tools/delegate_tool.py`.

- Remove `_active_subagents`, `_active_subagents_lock`, `_spawn_paused`, `_spawn_pause_lock`, `_register_subagent`, `_unregister_subagent`, `set_spawn_paused`, `is_spawn_paused`.
- `list_active_subagents()` becomes a thin wrapper over `registry.list_active()`.
- `interrupt_subagent()` becomes a thin wrapper over `registry.interrupt()`.
- `delegation.status` in server.py calls `registry.*` directly.
- **Verify**: all existing protocol tests green; `delegation.status` / `subagent.interrupt` / `delegation.pause` RPCs produce identical output.

#### Risks
## Risks with file:line

### R1 — Dual-registry desync during Steps 2-6

During the dual-write window (Steps 2-6), `_active_subagents` and the new registry are populated by two separate code paths with no shared transaction. A crash or exception between the two writes could leave one registry populated and the other not. The existing `finally` block at delegate_tool.py:1894 guarantees `_unregister_subagent` runs; the new `registry.complete()` call must be placed in the same `finally` block to guarantee symmetric cleanup.

### R2 — `_active_children` vs `_active_subagents` — two separate interrupt trees

`AIAgent._active_children` (agent_init.py:466) and `_active_subagents` (delegate_tool.py:157) both track running children but are populated/cleared independently with no shared lock. If the new registry replaces one without the other, `agent.interrupt()` recursive propagation (run_agent.py:2336) will break silently. Phase 5 must NOT touch `_active_children` — it serves the AIAgent-internal interrupt chain. The registry replaces only the TUI-visible `_active_subagents`.

### R3 — `child_task_id` vs `subagent_id` dual identity

`delegate_tool.py:1534` sets `child_task_id = _subagent_id or uuid4(...)`. The `task_id` field in `ExecutionTask` is the Core-facing key; `subagent_id` is the TUI-facing key. They share the same value today, but the two namespaces drift if callers ever supply a `task_id` on `task.submit` that differs from the generated `subagent_id`. The unified registry must enforce that these are the same key — or explicitly track both with a cross-reference field.

### R4 — Spawn-tree file race: concurrent sessions writing `_tasks.jsonl`

`spawn_tree.save` already appends to `_index.jsonl` without a file lock (server.py:4509-4516, noted "cache — losing a line just means list() falls back to a directory scan"). Extending this with concurrent `registry.complete()` calls writing `_tasks.jsonl` from worker threads will produce the same no-lock append. This is safe for JSONL if each `json.dumps(...)` is a single write call (atomic on most FS for < 4 KB), but must be validated; the existing comment acknowledges the trade-off.

### R5 — `_TASK_RESULTS` idempotency dict is process-local

server.py:8482 stores idempotency results in `_TASK_RESULTS` (in-process, TTL 600s). If the gateway process restarts (e.g., after a crash during a long delegate run), all in-flight idempotency keys are lost. Moving this into the registry does not fix the persistence gap — the registry is also in-process. Folding into `_tasks.jsonl` would make replay durable across restarts, but requires reading the file on every `task.submit` with an `idempotency_key`, adding an I/O call to a hot path. Decision required before Step 5.

### R6 — `intent="delegate"` turns `task.submit` into a long-blocking call

Current `task.submit` with `intent="slash"` is milliseconds. With `intent="delegate"`, it can block for minutes (full subagent run). The gateway handles all RPC methods synchronously in server.py. Existing turns already call `delegate_task` synchronously, so this is not a new problem, but it means the caller cannot cancel via `task.cancel` once `task.submit` is in-flight (the cancel RPC would have to reach the gateway on a different socket connection). This is the same constraint as today's `subagent.interrupt` (works because it arrives on a separate RPC call). Document this explicitly.

### R7 — `Status.RUNNING` adds a non-terminal value to the enum shared with stateless exec handlers

`action_runtime/contract.py` `Status` is currently only used on completed results. Adding `RUNNING` means consumers of `ExecutionResult` must guard against a non-terminal status where today they assume all results are terminal. Adapters in `adapters.py` never produce `RUNNING`; the risk is that a future adapter forgets the guard. Consider whether `RUNNING` belongs on `AgentTaskRecord.status` (a separate field type) rather than in the `ExecutionResult.Status` enum. The two have different invariants: an `ExecutionResult` is always a final answer; an `AgentTaskRecord` represents live state.

#### คำถามเปิด — ตัดสินครบ 6/6 (2026-06-10)
## Open questions — all decided 2026-06-10

### Q1 — Registry singleton location: standalone module or folded into BrainHost?

`BrainHost` (`agent/brain_host.py`, flag-gated `HERMES_BRAIN_HOST=1`) is the planned singleton factory for AIAgent construction. The `AgentTaskRegistry` is a natural tenant for BrainHost (it needs process-global scope, one instance per process). But BrainHost is still feature-flagged and off by default. Should `AgentTaskRegistry` live there (accepting the `HERMES_BRAIN_HOST` dependency), or in a standalone module that both BrainHost and the current path can import? The answer affects Step 1 and all subsequent steps.

> **ตัดสินแล้ว (2026-06-10): standalone module** (`action_runtime/task_registry.py`). เหตุผลชี้ขาด: dual-write ใน Step 2 ต้องเขียนจาก default path ที่ flag ปิดอยู่ — ถ้า registry อยู่ใน BrainHost จะถูกล็อกหลัง `HERMES_BRAIN_HOST=1` ทำให้ default path มอง subagent ไม่เห็น. BrainHost ถือ reference ภายหลังได้.

### Q2 — Persist idempotency keys across gateway restarts?

R5 above: should `_TASK_RESULTS` / idempotency replay be durable (written to `_tasks.jsonl` or state.db) or stay ephemeral (current behavior)? Durable replay means subagent runs that survived a gateway restart can be replayed without re-execution. Ephemeral is simpler and avoids the I/O cost on every `task.submit`. The answer determines Step 5 design and whether `state.db` needs a new table.

> **ตัดสินแล้ว (2026-06-10): ephemeral ต่อไป.** ประเด็นชี้ขาดไม่ใช่ I/O cost — หลัง crash ไม่มีทางรู้ว่า task ที่ค้างอยู่ทำ side effects ไปแค่ไหน; replay ผลที่ "อาจทำไปครึ่งเดียว" ขัด honest-status. สิ่งที่ persist คือ **task records** (observability) ไม่ใช่ replay store; task ที่ค้างตอน crash ให้สถานะตามจริง. ไม่มี state.db table ใหม่ใน Phase 5.

### Q3 — Should `Status.RUNNING` be part of `ExecutionResult.Status` or live only on `AgentTaskRecord`?

R7 above: adding `RUNNING` to the contract enum changes the invariant that every `ExecutionResult` is a terminal answer. The cleaner option is a separate `TaskStatus` enum on `AgentTaskRecord` that includes `RUNNING`, while `ExecutionResult.Status` stays terminal-only. This is a design decision for `contract.py` that affects how the Core interprets results from `task.status`.

> **Impact sweep เสร็จแล้ว (2026-06-10, fan-out 3 agents) — หลักฐานชี้ option B เด็ดขาด; รอ maintainer ยืนยัน.**
>
> **Census (consumer ทั้งหมดของ Status/ExecutionResult):** มีแค่ 4 ไฟล์ Python ที่แตะ action_runtime (`__init__`, `adapters`, `server.py` ผ่าน 5 deferred imports, test). **Option A ต้องแก้ ~8 จุด + พัง 1 จุดจริง**: idempotency replay store (`server.py:8478-8500`) cache ผลแบบ immutable ไม่มี update path — ถ้า cache `status:"running"` จะ replay เป็นคำตอบถาวร 600s แล้วกันการ re-execute; legacy wire (output/warning/error) ไม่มีช่อง status → RUNNING จะ render เป็น success เงียบๆ ใน 4 client (slashExec.ts:74, use-model-controls.ts:104, use-prompt-actions.ts:782, useSubmission.ts:170); `ExecutionResult.ok == (status is SUCCEEDED)` ทำให้ RUNNING อ่านเป็น "ล้มเหลว". **Option B แตะ 0 จุดเดิม** — โค้ดใหม่ล้วน.
>
> **Visibility (ตอบข้อกังวล "จะรู้ได้ไงว่าอะไรค้างอยู่"):** ทุกช่องทางที่ผู้ใช้เห็นงาน in-flight วันนี้ (`delegation.status` บน `_active_subagents` → TUI /agents overlay, event stream `subagent.*`/`tool.*` → desktop agents view, bg-task counters) **อยู่นอก ExecutionResult ทั้งหมด** — enum ถูกอ่านเฉพาะตอนผลจบแล้ว. Visibility ใน Phase 5 มาจาก `task.status`/`task.list` RPC ใหม่ ซึ่งให้ข้อมูลเดียวกันทั้งสอง option; option B record ยัง carry live state ได้ rich กว่า (option A ต้อง fabricate ExecutionResult เกือบเปล่าๆ สำหรับงานที่ยังรัน). **ช่องโหว่จริงที่พบ:** `bg_*` (prompt.background) query ไม่ได้เลยหลัง client restart → แก้โดยให้ bg_*/preview_* ลงทะเบียนใน registry — ไม่เกี่ยวกับว่า RUNNING อยู่ enum ไหน.
>
> **Lifecycle:** delegate_tool.py:1515 มี precedent record-level `"status": "running"` อยู่แล้ว; crash recovery ภายใต้ option B = flip record เป็น BLOCKED โดยไม่แตะ stored result ใดๆ; ภายใต้ option A ต้องมี terminal-only guard + write path ที่สอง + reconciliation sweep. หมายเหตุ: doc บรรทัด ~509/517/594 ร่างไว้แบบ option A — ต้อง amend เมื่อตัดสิน B.
>
> **ตัดสินแล้ว (2026-06-10): option B** — `TaskStatus` enum แยกบน `AgentTaskRecord` (มี RUNNING), `ExecutionResult.Status` คง terminal-only; `task.status` ฝัง terminal ExecutionResult เมื่อจบ; bg_*/preview_* ต้องลงทะเบียนใน registry. (maintainer ยืนยันตามหลักฐาน sweep ข้างบน) **Q1-Q6 ครบแล้ว — Phase 5 Step 1 เริ่มได้.** หมายเหตุ implement: ต้อง amend ร่าง option A ที่บรรทัด ~509/517/594 ให้ตรง ruling นี้ด้วย.

### Q4 — What does `intent="delegate"` return for a partially-interrupted batch?

If the parent is interrupted mid-batch (some children succeeded, some did not), delegate_tool.py:2177-2210 already collects whatever finished and marks the rest as "interrupted". Should `task.submit` with `intent="delegate"` return `Status.PARTIAL` for this case (matching the existing `Status.PARTIAL` in the contract)? Or always `Status.FAILED`? The answer ties to how the Core is expected to re-plan.

> **ตัดสินแล้ว (2026-06-10): `Status.PARTIAL`** + per-child breakdown ใน `outputs` (+ side effects ของ children ที่จบแล้วใน `side_effects`). `FAILED` จะทิ้งงานจริงและบังคับ Core รัน children ที่สำเร็จแล้วซ้ำ — เปลืองและ side-effect ไม่ปลอดภัย; `PARTIAL` ตรง semantics ที่ contract มีไว้แต่แรก.

### Q5 — Does the TUI's `spawn_tree.save` RPC survive long-term or get deprecated?

Today the TUI assembles the subagent tree payload from the event stream and calls `spawn_tree.save` on turn-complete (server.py:4478-4479 comment). In Phase 5, the registry writes `_tasks.jsonl` directly. Should `spawn_tree.save` remain as the canonical rich-snapshot path (TUI-assembled, richer than the registry's record), or should the registry's write be the canonical source and `spawn_tree.save` be deprecated? The answer determines whether the two files (`_index.jsonl` + `_tasks.jsonl`) need merging logic in `spawn_tree.list` indefinitely.

> **ตัดสินแล้ว (2026-06-10): dual-write ชั่วคราว, deprecate ด้วยเกณฑ์ไม่ใช่วันที่.** Sunset criteria: (1) registry record ครอบ field ครบของ spawn-tree payload และ (2) `spawn_tree.list` เสิร์ฟจาก registry ได้ 100% → deprecate `spawn_tree.save` (รับ RPC เป็น no-op compat อีก 1 release สำหรับ TUI เก่า). กันไม่ให้ merge logic ลากยาวอนันต์.
>
> **Sunset progress (2026-06-11):** `AgentTaskRecord` ปิด gap เพิ่ม 3 field (additive, absent-when-unset ทั้งหมด): `label` (alias ของ goal ที่ delegate dual-write/bg helpers; batch ของ `task.submit intent="delegate"` ใช้ summary แบบเดียวกับ `summarizeLabel` ของ TUI), `tools` (tail ของชื่อ tool จาก `update_progress` — distinct-consecutive, cap 8 ตาม `pushTool = pushUnique(8)` ฝั่ง TUI; ชื่ออย่างเดียว ไม่มี preview), `error` (one-line summary จาก `result.error.message` ตอน `complete()`; path ที่ result=None ไม่ synthesize). `spawn_tree.list`/`load` prefer field ใหม่เมื่อมี — บรรทัดเก่า fallback พฤติกรรมเดิม. **เกณฑ์ (1) ยังไม่ผ่าน — field ที่ยังอยู่ TUI-side เท่านั้น** (per-subagent `SubagentProgress` ใน ui-tui types.ts; engine/registry ไม่ produce ข้อมูลพวกนี้วันนี้ — ห้าม fabricate): `apiCalls`, `costUsd`, `inputTokens`, `outputTokens`, `reasoningTokens`, `iteration`, `index`, `taskCount`, `notes[]`, `thinking[]`, `outputTail[]` (preview + isError ต่อ tool call), `filesRead[]`, `filesWritten[]`, `toolsets[]`; รวมถึง `tools` แบบมี preview (registry เก็บแค่ชื่อ) และโครงสร้าง multi-subagent tree ต่อ turn — registry เก็บหนึ่ง record ต่อ task ดังนั้น `spawn_tree.load` บน ledger สังเคราะห์ single-node tree เท่านั้น (ไม่ประกอบ children ของ turn เดียวกันเป็น snapshot เดียวเหมือน payload ที่ TUI ประกอบ). ปิดเมื่อ engine surface ข้อมูลขึ้น registry (Step 7+).

### Q6 — Cross-process / multi-gateway support in scope for Phase 5?

The current registry is in-process. If two gateway processes serve the same `HERMES_HOME` (e.g., desktop + TUI connected simultaneously), each has its own `_active_subagents`. Phase 5 design as written does not change this. Is cross-process observability (one process seeing the other's subagents) in scope? If yes, the registry needs to be backed by `state.db` or a shared file from the start, which significantly changes Step 1.

> **ตัดสินแล้ว (2026-06-10): ตัดออกจาก Phase 5.** Registry เป็น in-process แต่ออกแบบ API ให้ backing store สลับได้ (state.db-backed = Phase 6 candidate — "เดินท่อเผื่อไว้แต่ยังไม่ติดแท็งก์"). ระหว่างนี้ `_tasks.jsonl` เป็นช่องอ่านข้าม process แบบหยาบได้อยู่แล้ว.

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
| 2026-06-11 | **Phase 5 Steps 5-6 landed + adversarial review (3 lens, 19 findings) → 16 แก้แล้ว / 4 deferred มีบันทึก.** Step 5: `task.submit intent="delegate"` — Q4 mapping (all→SUCCEEDED, mixed/interrupted→PARTIAL+per-child breakdown, none→FAILED typed ตาม Step-2 convention), idempotency fold เข้า registry (TTL/cap/timebase เดิม), task.submit อยู่ใน `_LONG_HANDLERS` อยู่แล้ว → slash ordering ไม่ขยับ, R6 documented. Step 6: `complete()` persist snapshot → `_tasks.jsonl` (append byte เดียว unbuffered), `spawn_tree.list` merge dedupe-by-task_id legacy-wins (Q5). **Review fixes สำคัญ**: busy gate บน delegate batch (กัน turn ชนกันบน agent เดียว + interrupt cross-coupling), in-flight duplicate task_id → 4034, clear interrupt flag หลัง batch (พิสูจน์แล้วว่าไม่ auto-reset), session_id wiring 3 จุด (Step 6 เคยเป็น dead code!), terminal guard + bounded retention (cap 1024) + guarded interrupt + reason parity + monotonic timebase + locked snapshots ใน registry, test isolation ของ idempotency (helper เดิม poke dict ร้าง), DENIED เฉพาะ spawn-pause, live tool_count/last_tool เข้า registry. Deferred 4 ข้อบันทึกใน §"Known gaps". Suites: 620 เขียว (fail เดิม 1 = browser env). หมายเหตุ: review รอบแรกชน session limit (verifier ตายหมด) — รอบแก้ทำหลัง limit reset โดย triage จาก raw findings ใน transcript |
| 2026-06-10 | **Phase 5 Steps 1-4 landed.** Step 1: `action_runtime/task_registry.py` (`AgentTaskRecord` + `TaskStatus` แยกตาม Q3-B + registry singleton, 17 tests รวม 2 ruling pins). Step 2: delegate dual-write — child ลงทะเบียน record (weakref agent_ref) คู่ `_register_subagent` + complete ใน finally เดียวกัน, mapping: interrupted/timeout→FAILED+TRANSPORT non-retryable, guard try/except ทั้งคู่ (8 tests). Steps 3-4: `subagent.interrupt` ผ่าน registry ก่อน fallback legacy, `delegation.pause` dual-write, RPC ใหม่ 3 ตัว `task.status`/`task.cancel`/`task.list` (query miss = found:false ไม่ใช่ error), `bg_*`/`preview_*` ลงทะเบียน+complete ตามจริง — ปิด visibility gap ที่ background task หายหลัง client restart (8 protocol tests). Suites: 265+403 เขียว. เหลือ: Step 5 (intent="delegate" บน task.submit + fold idempotency), Step 6 (registry-backed `_tasks.jsonl`), Step 7 (cutover — หลัง 5-6 stable) |
| 2026-06-10 | **Phase 5 open questions ตัดสินแล้ว 5/6 (maintainer ruling).** Q1=**standalone module** (`action_runtime/task_registry.py` — dual-write ต้องทำงานบน default path ที่ flag ปิด) · Q2=**ephemeral** (persist task records ไม่ใช่ replay store; crash แล้วไม่รู้ side effects → replay = โกหก) · Q4=**`Status.PARTIAL`** + per-child breakdown (FAILED ทิ้งงานจริง+เสี่ยง side-effect ซ้ำ) · Q5=**dual-write ชั่วคราว + sunset criteria** (record ครบ field + list เสิร์ฟจาก registry 100% → deprecate, no-op compat 1 release) · Q6=**ตัดออก** (API ออกแบบให้ backing store สลับได้; state.db-backed = Phase 6 candidate). **Q3 รอ impact sweep** — maintainer ขอหลักฐาน consumer census + พิสูจน์ว่า visibility ของงานค้างไม่หายก่อนตัดสิน |
| 2026-06-10 | **BrainHost migration complete (ทุก construction site) + Phase 3 Step 4 landed.** เพิ่ม `agent/brain_host_gate.py` — `build_agent(intent, **kwargs)` helper แบบบรรทัดเดียว (module แยกเบา import แค่ `os` → คง invariant "flag off ไม่ import `agent.brain_host`" ที่ test ยึดอยู่ + lazy-import `run_agent` ตอน call เหมือน site เดิม). Migrate 14 sites ที่เหลือ + ยุบ inline gate เดิม 3 จุดเป็น `build_agent` call เดียว — รวม **20 intents / 16 ไฟล์**: tui_gateway (tui_gateway, tui-background, preview-restart), gateway (gateway-run, history-hygiene, gateway-background, compress, api-server, feishu-comment), hermes_cli (cli, cli-background, oneshot, prompt-size), agent (background-review, curator), acp, cron, batch, run-agent-cli, delegate. **ตั้งใจไม่แตะ:** `cli.py` lazy wrapper (re-export shim), `scripts/tool_search_livetest.py`, ทุกอย่างรอบ `_active_subagents`/spawn-tree/interrupt ใน delegate_tool (R2). Step 4: เพิ่ม 10 accessors (attached_images, pending_title, image_counter, show_reasoning, tool_progress_mode, tool_started_at, personality, model_override, explicit_cwd, edit_snapshots) + แปลง subscript ที่เหลือทั้งหมดใน server.py — ยกเว้น 2 จุดโดยตั้งใจ: `model_override` ใน `_apply_model_switch` (caller ส่ง plain dict บน no-session path, มี isinstance guard อยู่แล้ว) และ `_finalized` (private marker จุดเดียว). Tests: gate helper 4 + site-table parametrized (แทน grep-test เดิม) + accessor sync; suites เขียว 164 (brain_host+tui_gateway+action_runtime) + 252 server + 171 (delegate/batch/prompt_size/curator/cron); commit กลาง bisect-green. ทำด้วย workflow 5 Sonnet agents บนกลุ่มไฟล์ disjoint ใน working tree ตรง (ไม่ใช้ harness worktree — บทเรียน base ผิด) |
| 2026-06-10 | **Phase 3 design delivered (ยังไม่ implement).** Scoping fan-out 3 agents → `SessionState` dataclass (lock เป็น field) + MutableMapping-shim incremental adoption (Step1 = zero handler churn) + `BrainHost` (`agent/brain_host.py`, flag-gated `HERMES_BRAIN_HOST`, additive) + map 20 AIAgent sites (first=`_make_agent`) + documented dual-path + parity test. เก็บใน §11 "Phase 3 — design detail". **Implementation ถือไว้เป็น cycle เฉพาะ** — decompose live session-state + locks ที่ 74 handlers + concurrency model พึ่ง (gap #1, เสี่ยงสุดในแผน) ไม่ควรรีบท้าย session |
| 2026-06-10 | **BrainHost sites #2-3 + Phase 5 design (Sonnet sub-agents).** gateway/run.py `_run_agent` LRU-miss site (intent="gateway-run") + api_server `_create_agent` (intent="api-server") gate ผ่าน `HERMES_BRAIN_HOST` แบบเดียวกับ `_make_agent` (+7 gate tests; 114 api_server failures ยืนยัน pre-existing ผ่าน stash round-trip). Phase 5 design ลงเอกสารแล้ว (§11 "Phase 5 — design detail"): `AgentTaskRegistry` + dual-write migration 6 steps + risks R1-R7 (สำคัญ: R2 — ห้ามแตะ `_active_children` interrupt chain) + open questions Q1-Q6. หมายเหตุ: live install ถูกอัปเดตไป main แล้ว — งาน dev ทำใน worktrees จาก branch refs |
| 2026-06-10 | **Phase 3 Step 3 + BrainHost seam landed (branch `feat/phase3-step3-brainhost`).** Step 3: 107 subscript sites → typed attributes (กฎ: square-bracket เท่านั้น, `.get/setdefault/pop` คงเดิมเพราะ absent-key semantics ต่าง, non-session dicts ข้าม) + wrap test dicts เป็น SessionState. BrainHost: `agent/brain_host.py` (AgentSpec + singleton + parity-tested `build_agent`) gate ใน `_make_agent` ด้วย `HERMES_BRAIN_HOST=1` default-off (zero import cost พิสูจน์ผ่าน sys.modules). ทำโดย Sonnet sub-agents ใน dedicated worktrees. suites: 156 + 278 green. **คงเหลือ:** Step 4 (ทิ้ง subscript duality), migrate ~20 AIAgent sites เข้า BrainHost ทีละ site, Phase 5 multi-runtime (design ก่อน) |
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
