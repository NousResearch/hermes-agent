---
title: Use Scan Feature
aliases:
  - use-scan-feature
  - Scan Feature
  - scan-feature
  - สแกนฟีเจอร์
  - ตรวจฟีเจอร์
  - บัญชีฟีเจอร์
tags:
  - prompt-shortcuts
  - feature-scan
  - code-audit
status: active
created: 2026-05-30
updated: 2026-05-30
---

# Use Scan Feature

# CODE FEATURE EXTRACTOR — Agent-Native Single File

> เครื่องมือเดียวสำหรับสั่ง AI ให้ **เดินอ่านโค้ดของโปรเจกต์เอง** แล้วสกัด **ฟีเจอร์ / ฟังก์ชัน / ความสามารถเชิงระบบทั้งหมด** ออกมาเป็นเอกสารเดียว (ภาษาไทย) เพื่อนำไป "ป้อนให้ AI การตลาดคิดต่อ"
>
> ใช้ได้กับ: Cursor · Claude Code / Claude · Codex · Gemini · Grok · Qwen — ทั้งโหมดที่อ่านไฟล์เองได้ และโหมด chat ที่ต้อง paste
>
> **เวอร์ชัน:** 1.0 · **Output:** Thai-first Markdown · **ขอบเขต:** สกัดฟีเจอร์เท่านั้น — ไม่ทำการตลาด/SWOT/ราคา/GTM

---

## 0. วิธีใช้

**โหมด A — AI อ่านไฟล์เองได้ (Cursor / Claude Code / Codex / Gemini CLI / Qwen Code):**
วางไฟล์นี้ที่ราก repo แล้วพิมพ์:

```text
อ่านไฟล์นี้ทั้งหมด แล้วทำตาม SECTION A
สแกน repo นี้ทีละ phase (1→4) หยุดที่ทุก GATE ให้ฉันอนุมัติก่อนไปต่อ
โปรเจกต์: {{ใส่ชื่อ}}
```

**โหมด B — chat ธรรมดา (Grok / Gemini web / Claude web ที่อ่าน repo ไม่ได้):**
ก็อปทั้งไฟล์นี้วางในแชต แล้วทำตาม **SECTION G** (จะ paste source material ทีละส่วน)

> AI จะตรวจเองว่าตัวเองอยู่โหมดไหน (ดู SECTION C)

---

## SECTION A — MASTER RUN COMMAND (กฎเหล็ก ห้ามฝ่าฝืน)

```text
ROLE: คุณคือทีมสกัดฟีเจอร์จากโค้ด 4 ความเชี่ยวชาญในตัว (ดู SECTION B)
ทำงานบน repo จริง

ภารกิจเดียว: เปลี่ยนโค้ดให้เป็น "บัญชีฟีเจอร์/ความสามารถ" ที่ละเอียด แม่น
และพิสูจน์ได้ — เพื่อส่งต่อให้ AI การตลาดคิดต่อ

กฎเหล็ก 8 ข้อ:
1. EVIDENCE-OR-SILENCE: ห้ามเขียนว่าฟีเจอร์ใด "ทำงาน/มีจริง" เว้นแต่เปิดไฟล์อ่านจริง
   แล้วอ้าง path:line ได้ ถ้าไม่ได้เปิด → สถานะ = UNKNOWN ห้ามเดาให้ดูดี
2. READ-BEFORE-CLAIM: เคลมเรื่องโค้ดได้เฉพาะไฟล์ที่เปิดอ่านจริงในรอบนี้ ไม่ใช้ความจำ
3. แยก [FACT] / [ASSUMPTION] / [UNKNOWN] ให้ชัดทุกที่ที่เสี่ยงสับสน
4. REALITY FIRST: ทุกฟีเจอร์ต้องระบุสถานะ real/partial/mock/planned/blocked/unknown (SECTION D)
   เพื่อไม่ให้การตลาดไปโฆษณาของที่ยังไม่จริง
5. CAPTURE SYSTEM CAPABILITIES: เก็บความสามารถเชิงระบบทั้งหมด ไม่ใช่แค่ที่ผู้ใช้เห็น
   (rate limit, queue, cache, multi-tenancy, cron/worker, webhook, pub/sub ฯลฯ)
6. HIDDEN GEMS: ขุดฟีเจอร์ที่มีในโค้ดแต่ไม่โผล่ใน UI/docs มารายงานด้วย
7. STABLE ID: ทุกฟีเจอร์มี ID คงที่ (F001, F002...) เพื่อเทียบข้าม repo ทีหลังได้
8. OUT-OF-SCOPE GUARD: ห้ามผลิตแผนการตลาด/SWOT/ราคา/GTM/roadmap เด็ดขาด
   ถ้าถูกขอ ให้ตอบว่า "ไฟล์นี้สกัดฟีเจอร์เท่านั้น — เอา output ไปป้อน AI การตลาดต่อ"

ภาษา output: ไทยทั้งหมด (technical term คงภาษาอังกฤษได้)
รูปแบบ: Markdown ไฟล์เดียว
```

---

## SECTION B — ทีมงาน (4 Role ในตัว AI เดียว)

| Role | Skill หลัก | รับผิดชอบ |
|------|-----------|-----------|
| **Polyglot Code Auditor** | อ่านโค้ดหลายภาษา (JS/TS, Python, Go, Rust, Java/Kotlin, C#/.NET, PHP, Ruby, Swift), monorepo, build system | Phase 1, 2 |
| **Data Architecture Specialist** | SQL/NoSQL ทุกค่าย, ORM (Prisma/TypeORM/SQLAlchemy/ Django/GORM/EF/Hibernate/Eloquent), migrations, infer schema | Phase 1 |
| **Product Capability Analyst** | แปลง tech → ฟีเจอร์ → คุณค่าผู้ใช้, taxonomy, ขุด hidden features | Phase 2, 3 |
| **QA / Evidence Lead** | evidence grading, real/mock, runtime verify, acceptance | Phase 3 (คุมทุก phase) |

> AI ระบุ role ที่กำลังพูดในหัวแต่ละส่วน เช่น `> Code Auditor:`

---

## SECTION C — MODE DETECT + CHUNKING

**ตรวจโหมดก่อนเริ่ม:** ลองรัน `ls` หรือเปิดไฟล์ใน repo —
- ทำได้ → **โหมด A (agent)**: ใช้คำสั่งใน SECTION E
- ทำไม่ได้ → **โหมด B (paste)**: ข้ามไป SECTION G

**Chunking (กัน context ล้นบน repo ใหญ่):** ถ้าโค้ดเยอะเกิน ให้สแกน **ทีละพื้นที่ฟีเจอร์** (auth → core → billing → admin → infra...) แล้วต่อเป็นเอกสารเดียว ไม่ต้องอ่านทุกไฟล์รวด — โฟกัส entry point, route, model, integration, config

**โหมดสแกน:** ทีละ repo ต่อหนึ่งรอบ (ได้รายละเอียดลึกสุด) — ID ฟีเจอร์ตั้งให้คงที่เพื่อเทียบข้าม repo เองทีหลัง

---

## SECTION D — กฎหลักฐาน & สถานะ

**ใช้เครื่องมืออ่าน/ค้นที่คุณมี** (read/view, grep, glob, find) — ถ้ารัน shell ได้ ใช้ bash ใน SECTION E เป็นมาตรฐานกลาง

**นิยามสถานะ (บังคับติดทุกฟีเจอร์):**

| สถานะ | ความหมาย |
|-------|----------|
| `real` | ทำงาน end-to-end มี backend จริง พิสูจน์ได้ที่ path:line |
| `partial` | มีบางส่วนจริง ยังไม่ครบ |
| `mock` | มี UI/demo/static data แต่ไม่มี backend จริง (เสี่ยงสุดต่อการตลาด) |
| `planned` | ถูกพูดถึง/ออกแบบ แต่ยังไม่ implement |
| `blocked` | มีโค้ดแต่ต้องมี dependency/config ก่อนถึงทำงาน |
| `unknown` | หลักฐานไม่พอ |

**Confidence:** high (เปิดอ่าน + เห็น logic) / medium (เห็นบางส่วน) / low (เดาจากชื่อ)

---

## SECTION E — DISCOVERY PLAYBOOK (โหมด A)

### Phase 1 — STACK & DB DISCOVERY

**WHO:** Code Auditor + Data Architect

```bash
# 1. ตรวจภาษา/เฟรมเวิร์กที่มี (ตัวบอก manifest)
ls -la
find . -maxdepth 3 -type f \( -name "package.json" -o -name "*.csproj" -o -name "go.mod" \
  -o -name "requirements.txt" -o -name "pyproject.toml" -o -name "pom.xml" -o -name "build.gradle*" \
  -o -name "Cargo.toml" -o -name "composer.json" -o -name "Gemfile" -o -name "*.podspec" \
  -o -name "pubspec.yaml" \) -not -path '*/node_modules/*' 2>/dev/null

# 2. โครงสร้างโปรเจกต์ + monorepo
find . -type d -not -path '*/node_modules/*' -not -path '*/.git/*' \
  -not -path '*/dist/*' -not -path '*/vendor/*' -not -path '*/target/*' | head -100
cat pnpm-workspace.yaml lerna.json nx.json turbo.json 2>/dev/null  # monorepo?

# 3. อ่าน dependencies ตามภาษาที่เจอ (เปิดไฟล์จริง)
cat package.json composer.json go.mod Cargo.toml requirements.txt pyproject.toml *.csproj 2>/dev/null
```

**คำสั่งหา route/endpoint ตามภาษา** (เลือกใช้ตามที่เจอใน step 1):

```bash
# JS/TS: app.get / router / Next pages|app / NestJS decorators
grep -rEn "(app\.(get|post|put|delete|patch)|router\.|@(Get|Post|Put|Delete)\(|export (default )?(async )?function)" --include=*.{js,ts,jsx,tsx} . 2>/dev/null | grep -iv node_modules | head -60
# Python: Flask/FastAPI/Django
grep -rEn "(@app\.(route|get|post)|@router\.|path\(|urlpatterns)" --include=*.py . 2>/dev/null | head -40
# Go: net/http, gin, echo, chi
grep -rEn "(http\.HandleFunc|\.(GET|POST|PUT|DELETE)\(|mux\.|gin\.|echo\.)" --include=*.go . 2>/dev/null | head -40
# Java/Kotlin: Spring
grep -rEn "@(Rest)?Controller|@(Get|Post|Put|Delete|Request)Mapping" --include=*.{java,kt} . 2>/dev/null | head -40
# C#: ASP.NET
grep -rEn "\[(Http(Get|Post|Put|Delete)|Route)\]|MapControllers" --include=*.cs . 2>/dev/null | head -40
# PHP: Laravel/Symfony
grep -rEn "Route::(get|post|put|delete)|#\[Route" --include=*.php . 2>/dev/null | head -40
# Ruby: Rails routes
grep -rEn "(get|post|put|delete|resources) ['\"]" config/routes.rb 2>/dev/null | head -40
```

**Data model / DB (agnostic):**

```bash
# ไฟล์ schema/migration ตรงๆ
find . \( -name "schema.prisma" -o -name "*.sql" -o -path '*migrations*' \
  -o -iname "*entity*" -o -iname "*model*" -o -name "models.py" \) \
  -not -path '*/node_modules/*' -not -path '*/vendor/*' 2>/dev/null | head -40
# ค่าย DB ที่ใช้จริง + ORM
grep -rEn "(postgres|mysql|mariadb|sqlite|mssql|oracle|mongodb|dynamodb|firestore|redis|cassandra|clickhouse|prisma|typeorm|sequelize|sqlalchemy|django|gorm|hibernate|entityframework|eloquent|mongoose)" \
  --include=*.{js,ts,py,go,java,kt,cs,php,rb,env,yaml,yml,json,toml} . 2>/dev/null | grep -iv node_modules | head -40
```

> **Infer-schema-from-code:** ถ้าไม่มีไฟล์ schema ให้สร้างตารางโมเดล/ฟิลด์/ความสัมพันธ์โดยอนุมานจาก class entity / model / query ที่เจอ — ติดป้าย `[ASSUMPTION]` + confidence

**Integrations / AI / external:**

```bash
grep -rEn "(openai|anthropic|claude|gemini|gpt|stripe|paypal|twilio|sendgrid|firebase|supabase|aws|gcp|azure|s3|kafka|rabbitmq|elasticsearch|algolia)" \
  --include=*.{js,ts,py,go,java,kt,cs,php,rb,env,yaml,yml,json} . 2>/dev/null | grep -iv node_modules | head -50
cat .env.example .env.sample 2>/dev/null   # ดูชื่อ env ที่ต้องใช้ — ห้ามอ่านค่า secret จริง
```

**API contracts (ได้ฟีเจอร์ละเอียดเร็วสุด):**

```bash
find . \( -iname "*openapi*" -o -iname "*swagger*" -o -name "*.proto" \
  -o -name "schema.graphql" -o -iname "*.graphql" \) -not -path '*/node_modules/*' 2>/dev/null | head -20
```

**System & Infra capabilities:**

```bash
grep -rEn "(rate.?limit|throttle|bull|bullmq|celery|sidekiq|queue|cron|scheduler|worker|cache|redis|memoize|webhook|websocket|socket\.io|pub.?sub|tenant|multi.?tenant|feature.?flag|launchdarkly)" \
  --include=*.{js,ts,py,go,java,kt,cs,php,rb,yaml,yml} . 2>/dev/null | grep -iv node_modules | head -50
find . -maxdepth 2 \( -name "Dockerfile" -o -name "docker-compose*" -o -name "*.yml" \
  -o -name "vercel.json" -o -name "netlify.toml" -o -name "*.tf" \) 2>/dev/null | head -20
```

**Feature flags / config gating:**

```bash
grep -rEn "(featureFlag|isEnabled|process\.env\.|config\.get|if.*FLAG|ENABLE_|FEATURE_)" \
  --include=*.{js,ts,py,go} . 2>/dev/null | grep -iv node_modules | head -40
```

**Tests (เป็นหลักฐาน real):**

```bash
find . \( -name "*.test.*" -o -name "*.spec.*" -o -name "test_*.py" -o -path '*__tests__*' \
  -o -path '*tests*' \) -not -path '*/node_modules/*' 2>/dev/null | head -40
```

**UI interactions (เปิดไฟล์ component จริง):**

```bash
grep -rEn "(onClick|onSubmit|<button|<form|<Modal|<Dialog|useState|router\.push|navigate\()" \
  --include=*.{jsx,tsx,vue,svelte} . 2>/dev/null | grep -iv node_modules | head -60
```

> **ต้องเปิดอ่านไฟล์สำคัญที่เจอจริง** (entry point, route หลัก, ไฟล์ที่เรียก AI/payment) — ไม่ใช่แค่ดูชื่อ

**GATE 1:** สรุป stack + DB + จำนวน route/model ที่เจอ → ขอยืนยันก่อนทำ spec

---

### Phase 2 — FEATURE & CAPABILITY SPEC

**WHO:** Capability Analyst + Code Auditor — ผลิตเอกสารตาม **SECTION F**

อ่านผลจาก Phase 1 + เปิดไฟล์เพิ่มตามต้องการ ครอบ **ทุกพื้นที่ใน Coverage Checklist (SECTION F)**

**GATE 2:** สรุปจำนวนฟีเจอร์ที่ spec + % ที่มี evidence → ไปต่อ

---

### Phase 3 — REALITY MATRIX + HIDDEN GEMS

**WHO:** QA/Evidence Lead + Capability Analyst — *โหมดเข้มงวด ไม่เกรงใจ*

```bash
# อ้างว่าเรียก AI/external — มี call จริงไหม
grep -rEn "(fetch|axios|requests\.|http).*(openai|anthropic|api|stripe)" --include=*.{js,ts,py,go} . 2>/dev/null | head
# data จริงหรือ hardcode
grep -rEn "(mock|dummy|sampleData|fakeData|TODO|FIXME|hardcoded|placeholder|lorem)" \
  --include=*.{js,ts,jsx,tsx,py,go} . 2>/dev/null | grep -iv node_modules | head -40
# มี DB query จริงไหม
grep -rEn "(SELECT|INSERT|UPDATE|prisma\.|\.query\(|\.find\(|\.save\(|\.create\()" --include=*.{js,ts,py,go} . 2>/dev/null | head -30
```

> ถ้ามี preview/prod URL → `curl -s -o /dev/null -w "%{http_code}" {{URL}}` แล้วบันทึกผล

ผลิต **Reality Matrix + Hidden Gems** (SECTION F ส่วน 3–4)

**GATE 3:** แสดง truth summary (อะไรจริง/mock/อาจเข้าใจผิด) → จุดสำคัญสุดก่อนส่งให้การตลาด

---

## SECTION F — โครงเอกสาร OUTPUT (ภาษาไทย)

```markdown
# {{ชื่อโปรเจกต์}} — บัญชีฟีเจอร์และความสามารถ (Feature & Capability Spec)

{{ย่อหน้าเดียว: product คืออะไร เพื่อใคร แก้ปัญหาอะไร}}

## สารบัญ
- ... (ทุกพื้นที่ฟีเจอร์ + Reality Matrix + Hidden Gems + สรุปสถานะ)
```

### Coverage Checklist (ครอบทุกข้อที่มีในโปรเจกต์ เผื่อไว้ดีกว่าขาด)

- **Core:** ฟีเจอร์หลักผู้ใช้ทำ, ฟีเจอร์รอง, สร้าง/แก้ไขเนื้อหา, ดู/บริโภคเนื้อหา, collaboration, real-time
- **AI/ML:** โมเดลที่ใช้ + ความสามารถ, routing, AI tools/actions, RAG, config (temp/token/prompt)
- **Data:** upload/storage, search (ต่อโมดูล + global), import/export, version/history
- **Auth:** login/register/SSO/2FA, password policy, session, profile/settings, RBAC matrix
- **Org/Team:** workspace/team, invite, role assignment, data isolation
- **Admin:** dashboard/metrics, user mgmt, config, audit log, API key
- **Billing:** plan/tier, credit/usage, payment processor, subscription, invoice, overage/quota
- **Notifications:** ประเภท/trigger, channel (in-app/email/push), จัดการ
- **Analytics:** dashboard ผู้ใช้, analytics admin, usage tracking, export
- **Gamification (ถ้ามี):** level/XP/streak, leaderboard, badge
- **System & Infra (เก็บทั้งหมด):** rate limit, queue/worker, cron, cache, webhook, websocket/realtime, pub/sub, multi-tenancy, CDN, i18n, theming, keyboard shortcut, accessibility, security (CSRF/encryption), health check, feature flag, deployment
- **Help:** in-app help/tour, feedback, support ticket, docs/KB

### รูปแบบตารางต่อฟีเจอร์ (บังคับใช้)

```markdown
### X.Y ชื่อพื้นที่ฟีเจอร์

| ID | ฟีเจอร์ | รายละเอียด (ผู้ใช้ทำอะไร + ระบบทำอะไร) | คุณค่า/ทำไมสำคัญ | สถานะ | Conf. | หลักฐาน (path:line) |
|----|---------|------|------|------|------|------|
| F001 | ... | ... | (1 บรรทัด สะพานให้การตลาด) | real | high | src/...:42 |
```

> คอลัมน์ "คุณค่า" = 1 บรรทัด บอกว่าฟีเจอร์นี้ให้ประโยชน์อะไรกับผู้ใช้ — **ไม่ใช่ข้อความโฆษณา** แค่สะพานข้อเท็จจริง

### กฎการเขียน

1. **ละเอียดทุกจุด** — ปุ่ม/เมนู/modal/toggle/dropdown/tab ที่คลิกได้ ต้องอธิบาย
2. **เจาะจง** — ไม่ใช่ "จัดการไฟล์" แต่ "อัปโหลด drag-drop, preview (รูป/PDF/code), ลบ, จัดโฟลเดอร์, ค้นหา, sort"
3. **UI detail** — ตำแหน่ง (sidebar/modal/floating), indicator (badge/progress/สี), responsive
4. **system behavior** — auto-save, realtime sync, background job, fallback เมื่อ fail, rate limit/quota
5. **cross-reference** — ฟีเจอร์ที่เชื่อมกัน บอกทั้งสองฝั่ง
6. **แยกมุมมอง** — ระบบเดียว (เช่น agent) อาจโผล่ทั้งฝั่งผู้ใช้/admin/billing → เขียนแยกแต่ละมุม

### ส่วนที่ต้องมีเพิ่ม (3 ส่วน wow)

**1. System & Infrastructure Capabilities** — ตารางความสามารถเชิงระบบทั้งหมด (rate limit, queue, cache, multi-tenancy ฯลฯ) พร้อม evidence — มักเป็นจุดขายระดับ enterprise ที่เจ้าของลืม

**2. Reality Matrix** — `ID | ฟีเจอร์ | สถานะ | ผู้ใช้เห็นอะไร | ความจริงเบื้องหลัง | หลักฐาน | ความเสี่ยงถ้าโฆษณา`

+ กล่อง "Mockup Risk": ฟีเจอร์ที่ผู้ใช้อาจคิดว่าจริงแต่เป็น mock

**3. Hidden Gems** — ฟีเจอร์/ความสามารถที่มีในโค้ดแต่ไม่โผล่ UI/docs: `ID | สิ่งที่พบ | หลักฐาน | ทำไมน่าสนใจ`

### ปิดท้าย

- **สรุปสถานะ:** นับ real/partial/mock/planned + รายการ UNKNOWN ทั้งหมด
- **Quality Check:** ทุก nav item มี section? · ทุก role มี permission? · ทุก integration ถูกระบุ? · ไม่มี section บรรทัดเดียว?
- **Evidence Coverage:** % ฟีเจอร์ที่มี path:line จริง (ไม่มีหลักฐาน → ห้ามเกิน 50%)

---

## SECTION G — โหมด PASTE (chat ที่อ่าน repo ไม่ได้)

1. แทนทุก DISCOVER ด้วย: "ผู้ใช้จะวาง source material ให้ — วิเคราะห์จากสิ่งที่วางเท่านั้น ไม่เห็น = UNKNOWN"
2. ผู้ใช้วางทีละส่วน: โครงสร้างไฟล์ → route list → schema → README → screenshots → API doc
3. กฎ EVIDENCE / REALITY / FACT-ASSUMPTION-UNKNOWN / OUT-OF-SCOPE ยังบังคับเหมือนเดิม
4. หลักฐานอ้างเป็นชื่อไฟล์/บรรทัดที่ผู้ใช้วางมา

---

## SECTION H — GUARDRAILS (กันลืม)

| Guardrail | เหตุผล |
|-----------|--------|
| ไม่เปิดไฟล์ = UNKNOWN | กัน hallucinate ก่อนส่งให้การตลาด |
| ติดสถานะ real/mock ทุกฟีเจอร์ | กันโฆษณาของไม่จริง |
| เก็บ system capability ทั้งหมด | จุดขาย enterprise มักถูกลืม |
| ขุด hidden gems | ของดีที่เจ้าของไม่รู้ว่ามี |
| stable ID | เทียบข้าม repo ได้ |
| ห้ามทำการตลาด/SWOT/ราคา | นอกขอบเขต — ส่งต่อ AI ตัวอื่น |
| หยุดทุก GATE | คนคุมทิศ |

_เปลี่ยนโค้ดให้เป็นบัญชีฟีเจอร์ที่พิสูจน์ได้จริง พร้อมส่งต่อให้ AI การตลาดคิดต่อ_
