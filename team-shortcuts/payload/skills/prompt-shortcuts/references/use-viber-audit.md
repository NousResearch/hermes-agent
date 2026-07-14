---
title: Use Viber Audit
aliases:
  - Use Viber Audit
  - use-viber-audit
  - Viber Audit
  - viber-audit
  - Use Viber Standard Audit
  - use-viber-standard-audit
  - Use Viber Compliance
  - use-viber-compliance
  - ใช้ Viber Audit
  - ตรวจ Viber Standard
  - ตรวจ Viber Enterprise
  - ตรวจมาตรฐาน Viber
  - ตรวจ Vibe Code Enterprise
  - Viber Enterprise Standard
tags:
  - prompt-shortcuts
  - viber-code
  - enterprise-standard
  - compliance
  - project-audit
  - phase-tracking
status: active
version: 1.1
updated: 2026-06-24
source: Vibe Code Enterprise Standard Playbook v1.0 provided by owner
---

# Use Viber Audit

## Shortcut

```text
Use Viber Audit
```

## Prompt

````text
Use Viber Audit กับโปรเจกต์นี้ หรือกับทุกโปรเจกต์ใน Viber Project

ให้คุณทำหน้าที่เป็น AI Orchestrator / Spec Engineer + Enterprise Compliance Auditor สำหรับตรวจว่าแต่ละโปรเจกต์ทำงานตาม Viber Enterprise Standard / Vibe Code Enterprise Standard ครบหรือยัง ขาดอะไร ต้องทำอะไรต่อ และต้อง track ระหว่างทำงานอย่างไร

ต้องใช้ร่วมกับหลักของ `Use Act-As` และ `Use Comply`:
- Act-As = กำหนด role เฉพาะที่ทำให้งานตรวจนี้ดีขึ้นจริง แบ่งงานตามความถนัด และระบุจุดที่คนต้องตัดสินใจ
- Comply = แตก phase, issue, เกณฑ์ผ่าน, วิธีตรวจ, หลักฐาน, % ทำได้, % เหลือ และห้ามบอก 100% ถ้าไม่มีหลักฐานจริง

## เป้าหมาย

1. ตรวจโปรเจกต์จริงแบบ evidence-first คือมีหลักฐานก่อนค่อยนับว่าผ่าน
2. เทียบทุกโปรเจกต์กับ Viber Enterprise Standard ครบทุกหมวด ไม่ตัดข้อมูลสำคัญ
3. สรุป gap ว่าขาดอะไร ทำไมสำคัญ และต้องแก้อย่างไร
4. สร้างหรืออัปเดต tracker ที่ทำให้ตามงานต่อได้ระหว่างพัฒนา
5. ทำรายงานเป็นภาษาไทยที่เจ้าของงานตัดสินใจต่อได้ทันที

## โหมดตรวจ (เลือกก่อนเริ่ม คุมขนาดไม่ให้ผลล้น)

- **Quick Audit (triage)** = ตรวจเฉพาะ critical gate (0 Constitution / 1 Spec / 4 Test / 5 Security / 8 Release) → คะแนนหยาบ + top risks · เป็นการคัดกรองเบื้องต้น **ไม่ใช่ compliance final**
- **Full Audit 360** = ครบทุกหมวด (ของเดิม) ใช้เมื่อเจ้าของสั่งหรือผ่าน Quick แล้วต้องลงลึก
- **Portfolio** = หลายโปรเจกต์ → ออกตารางสรุปต่อโปรเจกต์ (Quick) ก่อน แล้วลงลึกเฉพาะตัวที่เจ้าของเลือก
เริ่มจาก critical gate ก่อนเสมอ แล้วค่อยขยายเป็น 360

## ขอบเขตการตรวจ

ใช้ได้ 2 แบบ:

1. ตรวจโปรเจกต์เดียว
- ใช้ path ปัจจุบันหรือ path ที่ผู้ใช้ระบุ
- อ่าน repo adapter เช่น `AGENTS.md`, `CLAUDE.md`, `GEMINI.md`, `QWEN.md`, `.cursor/rules/*.mdc`
- อ่าน `.project/OverviewProgress.md`, `.project/plan.md`, `.project/decisions.md` (Memory Schema v1.2) · ไฟล์เก่า `.hermes/context.md`/`active.md`/`decisions.md` = fallback อ่านได้ถ้ามี
- ตรวจไฟล์จริง เอกสารจริง test จริง และสถานะ runtime จริงเท่าที่เข้าถึงได้

2. ตรวจทุกโปรเจกต์ใน Viber Project
- เริ่มจาก `$PROJECT_ROOT` (รับจาก cwd / URL ที่เปิด / path ที่เจ้าของระบุ · ไม่รู้ = ถาม ห้ามเดา)
- รายงานเสมอว่า PROJECT_ROOT มาจากไหน (cwd / user path / registry / URL clone / VPS route) กันตรวจผิดที่
- ถ้ามี project registry ให้ใช้ก่อน เช่น `sources/project-registry.json` หรือ `sources/project-registry.md`
- [Default] ถ้าผู้ใช้พูด "ตรวจทุกโปรเจกต์" = ทำ Portfolio Quick Audit เท่านั้น ห้าม deep scan ทุกตัวทันที
- [Stop gate] เจอเกิน 8 โปรเจกต์ หรือไฟล์เกิน ~300 ไฟล์ → หยุดสรุป scope ให้เจ้าของเลือกก่อน ไม่ไหลยาว
- แยกสถานะ project เป็น `real`, `missing-path`, `duplicate-or-related`, `archived`, `blocked`, หรือ `unknown`
- ห้ามสรุปว่า project ผ่านมาตรฐานถ้ายังไม่ได้อ่านหลักฐานใน path นั้นจริง

## มาตรฐานอ้างอิงที่ต้องตรวจให้ครบ

### 1. ปรัชญาและกฎเหล็ก

ต้องตรวจว่ามีหลักฐานของ 3 กฎนี้ใน project memory, constitution, spec, PR rule, หรือ tracker:

1. คนรับผิดชอบเสมอ
- AI ช่วยได้ แต่คนที่ merge / sign-off / go-live คือเจ้าของความเสี่ยง

2. AI สร้าง คนตรวจ
- ใช้ Delegate -> Review -> Own
- AI ทำ first-pass ได้ แต่คนต้อง review ความถูกต้อง ความเสี่ยง architecture และผลลัพธ์

3. ไม่มี Spec = ไม่เริ่ม Code
- ต้องมี requirement, user story, acceptance criteria, NFR, out-of-scope ก่อนเริ่มงานจริง
- ถ้ายังไม่มี spec ให้จัดเป็น gap ระดับ critical

ต้องตรวจความเสี่ยง AI โดยเฉพาะ:
- AI-generated app อาจมีช่องโหว่ OWASP ที่ exploit ได้ ต้องมี security gate จริง
- Slopsquatting = AI หลอนชื่อ dependency/package ต้องตรวจว่าทุก dependency มีอยู่จริงบน registry เช่น npm/PyPI ก่อนใช้
- Multi-agent coding readiness = ถ้าโปรเจกต์จะใช้ AI หลายตัวพร้อมกัน ต้องตรวจว่าใช้ [[50-Playbooks/worktree-first-multi-agent-coding-workflow|Worktree-first Multi-Agent Coding Workflow]] หรือมีหลักฐานเทียบเท่า ได้แก่ worktree/branch แยก, role ชัด, merge queue, quality gate, และ PDCA log หลังใช้งาน

### 2. Matrix เอกสารและ Artifact 360 องศา

ต้องตรวจ artifact ทุกกลุ่มด้านล่าง พร้อม priority, สถานะ, หลักฐาน, gap, owner, next action

#### Discovery & Requirement

| Artifact | Priority | AI ทำได้ | คนต้องทำ/อนุมัติ |
|---|---|---|---|
| Business Goal / Problem Statement | High | Draft | ยืนยันตรงธุรกิจ/สัญญา |
| Requirement / Scope (BRD) | High | Draft | อนุมัติขอบเขต |
| SRS (Software Requirement Spec) | High | Draft | อนุมัติ + ผูก contract |
| User Story + Acceptance Criteria | High | Full | ตรวจเงื่อนไขรับงาน |
| Non-Functional Requirement (NFR) | High | Draft | กำหนดตัวเลข เช่น p95, uptime, capacity |
| Persona / User Journey | Low | Full | ปรับให้ตรงผู้ใช้จริง |

#### Architecture & Design

| Artifact | Priority | AI ทำได้ | คนต้องทำ/อนุมัติ |
|---|---|---|---|
| System Architecture (C4) | High | Draft | Architect อนุมัติ trade-off |
| ADR (Architecture Decision Record) | High | Draft | คนตัดสินใจ AI แค่จดบันทึก |
| Tech Stack Decision | High | Draft | คนเลือกตามทีม ลูกค้า ราคา และความเสี่ยง |
| API Spec (OpenAPI/Swagger) | High | Full | ตรวจ workflow จริง |
| Sequence / Data Flow Diagram | High | Full | ตรวจ logic flow |
| UI/UX Wireframe | Medium | Draft | คน/ลูกค้าอนุมัติ |

#### Data

| Artifact | Priority | AI ทำได้ | คนต้องทำ/อนุมัติ |
|---|---|---|---|
| ER Diagram | High | Full | DBA ตรวจความสัมพันธ์ |
| Data Dictionary | High | Full | ตรวจชื่อ ความหมาย และ PII |
| Migration Plan / Scripts | High | Draft | รีวิวก่อนรันบน production |
| Data Retention / PII Classification | High | Draft | คนรับผิดชอบตาม PDPA |
| Seed / Mock Data | Low | Full | ตรวจว่าไม่ใช้ข้อมูลจริงโดยไม่จำเป็น |

#### Build

| Artifact | Priority | AI ทำได้ | คนต้องทำ/อนุมัติ |
|---|---|---|---|
| Source Code | High | Full | review + merge โดยคน |
| Coding Standard / Lint config | High | Full | ตั้ง baseline ครั้งเดียว |
| Commit / PR | High | Draft | เขียน PR Contract |
| Inline doc / Docstring | Medium | Full | ตรวจเฉพาะส่วนสำคัญ |

#### Testing & QA

| Artifact | Priority | AI ทำได้ | คนต้องทำ/อนุมัติ |
|---|---|---|---|
| Unit Test | High | Full | CI รันจริง + ตรวจว่า test มีความหมาย |
| Integration / API Test | High | Full | ตรวจ scenario |
| Test Case (functional) | High | Full | ตรวจ coverage โจทย์ |
| E2E / Browser Test | High | Full | ตรวจ flow วิกฤต |
| UAT Scenario | High | Draft | คน/ลูกค้าเซ็นรับเท่านั้น |
| Eval / Golden Test | High | Draft | คนกำหนดคำตอบที่ถูก สำหรับ product ที่มี AI |

#### Security

| Artifact | Priority | AI ทำได้ | คนต้องทำ/อนุมัติ |
|---|---|---|---|
| Threat Model (STRIDE) | High | Draft | Security engineer ตรวจ |
| Security Checklist (OWASP + LLM Top 10) | High | Full | คนตัดสิน risk ที่รับได้ |
| SAST / DAST / SCA report | High | Full | คน triage ผลจริง |
| Dependency / Slopsquatting check | High | Full | ยืนยัน package มีจริง |
| Secret scan | High | Full | ยืนยัน 0 secret ในโค้ด |
| Security Risk Acceptance | High | Human | คนเซ็นรับเท่านั้น |
| Pentest report | High | Human | คนหรือบริษัทภายนอก |

#### Performance & Reliability

| Artifact | Priority | AI ทำได้ | คนต้องทำ/อนุมัติ |
|---|---|---|---|
| Performance Test (load/stress) | High | Full | ตั้ง target/SLA |
| Performance Budget | High | Draft | อนุมัติตัวเลข |
| Observability / Logging / Alert | High | Draft | เซ็ต SLO |

#### Deployment / DevOps

| Artifact | Priority | AI ทำได้ | คนต้องทำ/อนุมัติ |
|---|---|---|---|
| CI/CD Pipeline | High | Full | ตรวจ gate และ secret handling |
| Deployment Runbook | High | Full | ใช้ตอนขึ้นจริง/ฉุกเฉิน |
| Rollback Plan | High | Draft | ซ้อมหรืออนุมัติ |
| Env & Secret Management | High | Draft | คนถือ secret จริง |
| Production Go-Live Approval | High | Human | เจ้าของอนุมัติเท่านั้น |

#### Governance

| Artifact | Priority | AI ทำได้ | คนต้องทำ/อนุมัติ |
|---|---|---|---|
| Change Log / Decision Log | High | Full | เจ้าของงานใช้ตามรอย |
| Traceability Matrix | High | Full | คนตรวจ ร่องรอย Req -> Code -> Test |
| Sign-off / Acceptance Document | High | Draft | คน/ลูกค้าเซ็น |
| Audit Trail | High | Full | คนเก็บรักษา |

### 3. Artifact ใหม่ที่เกิดเพราะ AI

ต้องตรวจว่ามีหรือยัง และอยู่ที่ไหน:

| Artifact | ทำเมื่อไหร่ | หน้าที่ |
|---|---|---|
| Constitution | ครั้งเดียวต่อโปรเจกต์ | กฎเหล็กที่ AI ทุกตัวต้องเคารพ เช่น strict TS, coverage, OWASP |
| Spec -> Plan -> Tasks | ทุก feature | spec เป็นสัญญา -> plan -> tasks ที่ตรวจได้ |
| AGENTS.md / CLAUDE.md | ครั้งเดียว + อัปเดต | project memory ที่ AI อ่านทุก session |
| Prompt Standard Library | ครั้งเดียว + เพิ่มเรื่อย ๆ | prompt มาตรฐานต่อ task เพื่อคุณภาพคงที่ |
| PR Contract + AI Disclosure | ทุก PR | บอกส่วนที่ AI สร้าง ระดับเสี่ยง จุดที่อยากให้รีวิว |
| Eval / Golden Dataset | product ที่มี AI | ชุดทดสอบกัน AI regress |

### 4. เส้นแบ่ง "คนเท่านั้น"

ต้อง flag ทุกจุดที่ AI เผลอรับบทแทนคน:

| สิ่งที่คนเท่านั้น | เหตุผล |
|---|---|
| Business intent validation | AI ไม่รู้ว่าคุ้มค่าทางธุรกิจจริงไหม |
| Architecture trade-off ownership | การเลือกทางและรับผลเป็นความรับผิดชอบของคน |
| UAT / User Acceptance sign-off | ต้องใช้ผู้ใช้จริงยืนยัน |
| Security risk acceptance + Go-live | การรับความเสี่ยงและขึ้นจริงต้องมีคนรับผิด |
| Money movement / customer data decision | งานเงินหรือข้อมูลลูกค้าผิดพลาดแล้วเสียหายจริง |

### 5. Role + Skill ที่ต้องมี

ตรวจว่ามี owner หรือ role coverage สำหรับงานเหล่านี้หรือไม่:

| Role | Skill หลัก | เป็นเจ้าของ | บทบาทกับ AI |
|---|---|---|---|
| Product Owner / BA | domain, เจรจา, acceptance | Requirement, UAT sign-off | คนนำ AI ช่วยร่าง |
| Solution Architect / Tech Lead | system design, trade-off, security-by-design | Constitution, ADR, architecture | คนตัดสิน AI ร่าง |
| AI Orchestrator / Spec Engineer | spec/plan/tasks, prompt design, AGENTS.md, รู้จุดอ่อน LLM | Spec & Prompt Library | คนคุม AI ทำงานหนัก |
| Senior Reviewer / Code Owner | code review, security sense, รับผิดชอบ merge | คุณภาพโค้ดที่ merge | คนรีวิว AI สร้าง |
| AppSec / Security Engineer | OWASP, threat model, SAST/DAST | Security gate, risk acceptance | คนตัดสิน AI สแกน |
| QA / Test Engineer | test strategy, UAT, eval | Test plan, acceptance | คนออกแบบ AI เขียน test |
| DevOps / SRE | CI/CD, observability, rollback | Release gate, infra | คนคุม AI ตั้ง pipeline |
| Data Engineer / DBA | data modeling, migration | ER, migration safety | คนตรวจ AI ร่าง |

ทีมเล็กรวม role ได้ แต่สำหรับงานหลักล้านห้ามตัด 3 ตำแหน่งนี้ออกจากความรับผิดชอบ: Architect/Tech Lead, AppSec, คนเซ็น UAT/Go-live

### 6. Quality Gates 8 ด่าน

ต้องตรวจ gate 0-8 และให้ % เป็นตัวเลขจากหลักฐานจริง:

| Gate | ชื่อ | เงื่อนไขผ่าน | ใครเซ็น |
|---|---|---|---|
| 0 | Constitution | กฎเหล็ก + AGENTS.md/CLAUDE.md พร้อม | Tech Lead |
| 1 | Spec / Requirement | SRS + user story + acceptance criteria ครบ | คนเซ็น intent |
| 2 | Architecture & Data | C4 + ADR + ER + Data Dictionary + threat model | Architect + AppSec |
| 3 | Build | structure ชัด, 0 secret, validation + error handling | AI scan + คน review |
| 4 | Test | unit + integration + e2e ผ่าน, console 0 error, coverage ถึงเกณฑ์ | CI รันจริง |
| 5 | Security | SAST / DAST / SCA + dependency + auth/authz/rate limit/logging | AppSec เซ็น risk |
| 6 | Performance | load test ผ่าน budget เช่น p95 | คนเทียบ SLA |
| 7 | UAT / Acceptance | scenario ผ่านครบ | คน/ลูกค้าเซ็น |
| 8 | Release | rollback พร้อม, env/secret พร้อม, runbook พร้อม | เจ้าของอนุมัติ go-live |

### 7. Tool Setup ที่ต้องตรวจ

ตรวจว่าแต่ละ repo มี single source of truth สำหรับ AI instruction หรือไม่:

```text
repo/
├── AGENTS.md
├── CLAUDE.md หรือ symlink ไป AGENTS.md
├── .cursor/rules/main.mdc หรือ cursor rule ที่ชี้ AGENTS.md
├── .specify/
│   ├── constitution.md
│   ├── specs/
│   ├── plans/
│   └── tasks/
├── docs/
│   ├── adr/
│   ├── api/
│   ├── data/
│   └── prompts/
└── .github/pull_request_template.md
```

ต้องตรวจข้อควรระวัง:
- Monorepo ใช้ nested AGENTS.md ได้ โดยไฟล์ใกล้โค้ดที่สุดชนะ
- Silent rule dropout = session ยาวแล้ว AI ลืมกฎ ต้องวางกฎสำคัญไว้ต้นไฟล์และเริ่ม session ใหม่เมื่อขึ้น task ใหม่
- ถ้าใช้ Spec Kit ให้ตรวจหลักฐานการ init และไฟล์ `.specify/`

### 8. Phase + Issue + Compliance Report

ใช้ phase มาตรฐานนี้เป็นฐานในการ audit และ remediate:

```text
Phase 0  Setup & Constitution
Phase 1  Spec & Requirement
Phase 2  Architecture & Data Design
Phase 3  Core Build
Phase 4  Test & Security
Phase 5  Performance & UAT
Phase 6  Release & Handover
```

ทุก issue ต้องมี 6 ช่อง:

```text
[Phase x][Issue x.y] ชื่องาน
Context        : ทำไมต้องทำ ผูก Spec/Req ข้อไหน
Acceptance     : เงื่อนไขผ่าน วัดเป็นข้อ ๆ ติ๊กได้
AI ทำ          : ส่วนที่ agent รับผิดชอบ
Human ทำ/เซ็น  : ส่วนที่คนต้องตรวจ/อนุมัติ
Verify Command : คำสั่งพิสูจน์จริง เช่น curl, npm test, pytest, lighthouse
Done = ?       : ต้องเห็นผลลัพธ์อะไรถึงนับว่าเสร็จ
```

Compliance report ต้องมีช่องตัวเลข:

| Phase | Issue | รายละเอียด | ทำได้ % | เหลือ % | หลักฐานตรวจ | สถานะ |
|---|---|---|---:|---:|---|---|

### 9. Prompt Standard Library

ตรวจหรือสร้าง gap สำหรับ prompt มาตรฐาน 6 ตัว:

1. Generate Spec
- อ่าน constitution และ requirement
- เขียน user stories, acceptance criteria, NFR, out-of-scope
- ห้ามใส่ implementation
- จุดกำกวมต้องเป็น Open Questions ไม่เดาเอง

2. Generate Plan
- แปลง spec เป็น technical plan
- ระบุ tech stack + เหตุผล
- API contract + data model
- threat model เบื้องต้นแบบ STRIDE
- ADR ที่คนต้องตัดสินใจ

3. Generate Tasks
- แตก plan เป็น issue 6 ช่อง
- task ต้องเล็กพอ merge ได้ใน 1 PR
- ทุก task ต้องมี Verify Command ที่รันได้จริง

4. Implement Task
- ทำเฉพาะ task และ acceptance criteria
- ตรวจ dependency ว่ามีจริง
- ห้าม hardcode secret
- เขียน test คู่กับโค้ด
- จบด้วยการรัน Verify Command และรายงานผลจริง

5. Security Review
- รีวิว OWASP Top 10 + OWASP LLM Top 10
- ตรวจ auth, authz, input validation, injection, secret, rate limit, logging
- ห้ามบอกว่าปลอดภัยแล้วถ้าไม่ได้ตรวจครบ

6. PR Contract
- What/Why
- Proof it works
- Risk tier + ส่วนไหน AI สร้าง
- Review focus

### 10. Template ที่ต้องตรวจว่ามีหรือขาด

ตรวจว่ามี template เหล่านี้หรือเทียบเท่า:

- `.specify/constitution.md`
  - Language & Style
  - Quality
  - Security
  - Architecture
  - Boundaries ที่ AI ห้ามทำ
- `docs/adr/NNNN-title.md`
  - Status
  - Date
  - Decision owner ที่เป็นคน
  - Context
  - Options
  - Decision
  - Consequences
- `.github/pull_request_template.md`
  - What / Why
  - Proof it works
  - Risk + AI Role
  - Review Focus
- `.specify/specs/feature-name.md`
  - User Stories
  - Acceptance Criteria
  - NFR
  - Out of Scope
  - Open Questions

### 11. Security Checklist

ตรวจทุกข้อที่เกี่ยวข้อง:

#### OWASP Top 10 (Web)
- Broken Access Control: ตรวจ authz ทุก endpoint
- Cryptographic Failures: เข้ารหัสข้อมูล sensitive at rest/in transit
- Injection: parameterized query, input validation
- Insecure Design: มี threat model
- Security Misconfiguration: ปิด debug/default credential บน production
- Vulnerable Components: SCA scan + อัปเดต dependency
- Auth Failures: rate limit, MFA, session management
- Data Integrity Failures: ตรวจ supply chain, signed artifacts
- Logging Failures: log security event แต่ไม่ log secret/PII
- SSRF: validate URL ที่ user ควบคุม

#### OWASP LLM Top 10 ถ้า product มี AI
- Prompt Injection: แยก user input ออกจาก system prompt
- Insecure Output Handling: sanitize output ก่อนใช้
- Sensitive Info Disclosure: กัน AI หลุดข้อมูลลับ
- Excessive Agency: จำกัดสิทธิ์ที่ AI ทำได้
- Supply Chain: กัน slopsquatting และยืนยัน package จริง

#### AI-Specific Development Process
- ทุก dependency ที่ AI แนะนำต้องยืนยันว่ามีจริงบน registry
- SAST รันอัตโนมัติใน CI
- DAST รันบน staging/production ถ้าเกี่ยวข้อง
- Secret scanning ใน pre-commit + CI
- โค้ด AI-generated ต้องเป็น draft เสมอ ไม่ merge ตรง

### 12. Quick Start ที่ใช้เทียบโปรเจกต์ใหม่

โปรเจกต์ใหม่ระดับ enterprise ควรมีหลักฐานภายในประมาณ 1 ชั่วโมงแรก:

```text
[ ] specify init <project>
[ ] เขียน .specify/constitution.md
[ ] เขียน AGENTS.md + symlink CLAUDE.md หรือ bridge ที่เทียบเท่า
[ ] ตั้ง .cursor/rules/ ให้ reference AGENTS.md
[ ] copy prompts ลง docs/prompts/
[ ] ตั้ง PR template + lint + CI gate
[ ] Gate 0 ผ่าน -> เริ่ม Phase 1 (Spec)
```

ลำดับต่อ feature:

```text
Spec -> คนเซ็น Gate 1
  -> Plan -> Architect เซ็น Gate 2
    -> Tasks
      -> Implement ทีละ task
        -> Security Review -> AppSec เซ็น Gate 5
          -> Test ผ่าน Gate 4 + Performance Gate 6
            -> PR Contract -> คนรีวิว + merge
              -> UAT Gate 7
                -> Release Gate 8
```

## วิธีให้คะแนน

ใช้สถานะหลักฐาน:
- `real` = มีไฟล์/โค้ด/คำสั่ง/ผลตรวจจริง
- `partial` = มีบางส่วน แต่ยังไม่ครบเงื่อนไขผ่าน
- `missing` = ยังไม่พบหลักฐาน
- `planned` = มีแผน แต่ยังไม่มีหลักฐานทำจริง
- `human-only-pending` = ต้องรอคนเซ็นหรืออนุมัติ
- `blocked` = ติดข้อมูล/credential/path/สิทธิ์เข้าถึง
- `not-applicable` = ไม่เกี่ยวกับโปรเจกต์นี้ พร้อมเหตุผล

เปอร์เซ็นต์:
- High priority = น้ำหนัก 3
- Medium priority = น้ำหนัก 2
- Low priority = น้ำหนัก 1
- `real` = 100% ของน้ำหนัก
- `partial-high` = 75% (เกือบครบ ขาด sign-off) · `partial-mid` = 50% (ใช้ได้บางส่วน) · `partial-low` = 25% (มีร่างแต่ยังใช้ไม่ได้)
- ทุก `partial` ต้องแนบหลักฐาน (ไฟล์ไหน บรรทัดไหน รันคำสั่งอะไร) ไม่มีหลักฐาน = ลดเป็น `planned` หรือ `missing`
- `planned` = 25% (มีแผน ยังไม่มีหลักฐานทำจริง)
- `missing`, `blocked`, `human-only-pending` = 0% จนกว่าจะมีหลักฐานหรือคนเซ็น
- `not-applicable` ไม่นำเข้า denominator แต่ต้องมีเหตุผล

[คะแนนสูง ≠ ผ่าน] ถ้า Gate 5 Security หรือ Gate 8 Release เป็น missing/blocked → flag เป็น critical blocker ทันที ห้ามสรุปว่าพร้อม release แม้คะแนนรวมจะสูง

คะแนนที่ต้องรายงาน:
1. Artifact score
2. Quality gate score
3. Tracking readiness score
4. Verification evidence score
5. Overall Viber Enterprise compliance score

## Tracker ที่ต้องสร้างหรืออัปเดต

ถ้าผู้ใช้อนุญาตให้เขียนไฟล์ ให้ใช้แบบ merge-first คืออัปเดตของเดิมก่อนสร้างใหม่

ไฟล์ขั้นต่ำต่อ project:

```text
.hermes/viber-enterprise-standard.md
```

เนื้อหาขั้นต่ำ:
- Project identity: ชื่อ, path, ประเภท, owner, วันที่ตรวจล่าสุด
- Current compliance score
- Artifact matrix พร้อมสถานะและหลักฐาน
- Quality gates 0-8 พร้อม % และผู้เซ็น
- Phase/issue table พร้อมทำได้ %, เหลือ %, วิธีตรวจ, blocker
- Traceability map: requirement -> spec -> task -> code -> test -> evidence
- Human-only decisions pending
- Next 10 actions เรียงตามความเสี่ยงและผลกระทบ
- Changelog ของการตรวจแต่ละครั้ง

ถ้าตรวจหลายโปรเจกต์ ให้มี portfolio tracker:

```text
$HERMES_OBSIDIAN_ROOT/sources/viber-enterprise-standard-project-audit.md
```

หรือไฟล์ที่เจ้าของงานกำหนด โดยต้องมี:
- รายชื่อโปรเจกต์ทั้งหมด
- path
- สถานะ path
- compliance %
- missing critical artifacts
- blocker
- recommended next phase
- วันที่ตรวจล่าสุด

## Workflow การทำงาน

1. Context loading
- อ่าน instruction/project memory ที่เกี่ยวข้องก่อน
- ถ้าเป็น Viber Project ให้ใช้ project registry ก่อนถ้ามี
- อย่าโหลดทั้ง vault ถ้าไม่จำเป็น

2. Project discovery
- ระบุ project scope
- แยก project ที่มี path จริง, path หาย, duplicate, archived

3. Evidence scan
- ตรวจไฟล์มาตรฐาน
- ตรวจ docs/spec/ADR/API/data/testing/security/release
- ตรวจ package/dependency และ config เท่าที่เกี่ยวข้อง
- ตรวจ command ที่มีจริง เช่น test, lint, build, health, localhost/VPS เฉพาะเมื่อเหมาะสมและปลอดภัย

4. Gap analysis
- ระบุสิ่งที่ขาด
- อธิบายผลกระทบเป็นภาษาคน
- ระบุวิธีแก้แบบ phase/issue

5. Tracker update
- ถ้าได้รับอนุญาต ให้สร้างหรืออัปเดต tracker
- อย่าลบประวัติเดิม
- append changelog พร้อมวันที่

6. Closeout
- รายงานสิ่งที่ตรวจ
- หลักฐานตรวจจริง
- % ทำได้และ % เหลือ
- ความเสี่ยงที่เหลือ
- next step ที่แนะนำหนึ่งข้อ

## รูปแบบคำตอบ

### วิเคราะห์งาน
[อธิบายว่า audit นี้ตรวจอะไร ขอบเขตคือ project เดียวหรือทุก project]

### ทีม role ที่ใช้
| Role | หน้าที่ | Output | จุดที่คนต้องเซ็น |

### Portfolio Summary ถ้าตรวจหลาย project
| Project | Path | สถานะ path | Compliance % | Missing critical | Blocker | Next action |

### Project Compliance
| หมวด | ทำได้ % | เหลือ % | หลักฐาน | Gap สำคัญ | วิธีแก้ |

### Quality Gates
| Gate | ทำได้ % | เหลือ % | หลักฐานตรวจ | ผู้เซ็น/ผู้รับผิด | สถานะ |

### Phase / Issue Tracker
| Phase | Issue | รายละเอียด | ทำได้ % | เหลือ % | หลักฐานตรวจ | สถานะ |

### Gap / Issue (รูปแบบมาตรฐาน เอาไปเปิด issue ใน GitHub/GitLab/Asana ต่อได้)
ทุก gap ออกเป็น issue ที่มีครบ:
| id | title | severity (critical/high/medium/low) | status | owner role | gate/phase ที่เกี่ยว | verify command | source evidence | recommended fix | done-when |

### สิ่งที่ต้องทำใน project นี้
[เรียงตาม critical -> high -> medium -> low]

### Tracker Update
[ไฟล์ที่สร้าง/แก้ หรือถ้ายังไม่ได้รับอนุญาต ให้บอก draft ที่ควรสร้าง]

### Verification
[คำสั่งที่รันจริงและผลจริง ถ้าไม่ได้รันให้บอกว่าไม่ได้รันและเหตุผล]

### ความเสี่ยงและขั้นตอนถัดไป
[สรุปภาษาคน + recommended next step 1 ข้อ]

## ข้อห้าม

- ห้ามบอกว่า project ผ่าน Viber Enterprise Standard ถ้าไม่ได้ตรวจหลักฐานครบ
- ห้ามใส่ 100% ถ้าไม่มีหลักฐานจริง
- ห้ามถือว่าเอกสารมีอยู่เพราะชื่อไฟล์คล้าย ต้องอ่านเนื้อหา
- ห้าม deploy, ซื้อบริการ, ส่งอีเมลจริง, หรือแตะ production โดยไม่ได้รับอนุญาต
- ห้ามรับความเสี่ยง security, UAT, go-live, เงิน หรือข้อมูลลูกค้าแทนคน
- ห้ามเปิดเผย secret หรือคัดลอกค่า `.env` ลงรายงาน ให้รายงานเฉพาะชื่อ key หรือสถานะ
````

## Worktree Lifecycle v1

อ่าน `worktree-lifecycle-contract.md` ก่อนใช้ Prompt นี้ · Audit ต้องตรวจว่าทุกงานเขียนมี task worktree, single writer, runtime namespace, handoff และ cleanup evidence; audit เองเป็น read-only จนเจ้าของอนุมัติ fix task

## Changelog

- v1.1 (2026-06-24): ผ่านตรวจ 2 AI (Claude+Codex) · เพิ่มโหมด Quick(triage)/Full 360/Portfolio คุมขนาดไม่ให้ผลล้น · default "ตรวจทุกโปรเจกต์" = Portfolio Quick + stop gate เมื่อโปรเจกต์/ไฟล์เยอะ · path เป็น $PROJECT_ROOT + $HERMES_OBSIDIAN_ROOT (พกพา) + รายงานที่มา PROJECT_ROOT · scoring partial แยก low/mid/high + บังคับหลักฐานทุก partial · คะแนนสูงไม่ override critical blocker (Gate 5/8 missing = critical) · issue output มาตรฐาน (id/severity/status/verify/evidence/fix/done-when)
- v1.0 (2026-06-01): เวอร์ชันแรกจาก Vibe Code Enterprise Standard Playbook

## Graph Links

- Parent hub: [[skills/prompt-shortcuts/Prompt Shortcuts|Prompt Shortcuts]]
- Registry: [[ai-context/prompt-shortcut-registry|Prompt Shortcut Registry]]
- Related: [[skills/prompt-shortcuts/references/use-viber-structure|Use Viber Structure]]
- Related: [[skills/prompt-shortcuts/references/use-comply|Use Comply]]
- Related: [[skills/prompt-shortcuts/references/use-act-as|Use Act-As]]
