---
name: prompt-shortcuts
description: Use this skill when the user invokes any reusable prompt shortcut from HermesAgent, including "Use Act-As", "Use Comply", "Use Summary", "Use Scan Feature", "Use AI Relay", "Use Viber Structure", "Use Viber Audit", "Use Impeccable", "Use Blog Auto", "Use WOW Resource", "Use Flow Guardian", "Use New Chat", "Use Close Chat", "Use Save Git", "Use Merge to Production", "Use Continue", "Use Move Folder", "Review Chat", "Use AI Pair", "Use Business Plan", "Use SaaS Opus Master Prompt", "Use BusinessPlan", "Use OverviewProgress", "Use FeatureSpec", "Use DesignSystem", "Use Create Design System", "Use Hermes Structure", "Use Create Content", "Use SonarQube", or Thai/alias variants such as "ใช้ AI Relay", "ปิดแชท", "เซฟ Git", "ทำต่อ", "รีวิวแชท", "สร้างคอนเทนต์จากแชท", "สร้างไฟล์ภาพรวมงาน", and "สแกน SonarQube".
metadata:
  short-description: Reusable prompt shortcut loader
---

# Prompt Shortcuts

This skill loads standard reusable prompts from HermesAgent. The v2 prompt files in `references/` are the source of truth; do not paraphrase them when the user asks to use a shortcut.

## Shortcut Map

| Shortcut | Aliases | Prompt File |
| --- | --- | --- |
| `Use Act-As` | `use-act-as`, `Use Act As`, `Act-As`, `act-as`, `ใช้ Act-As`, `กำหนดบทบาท`, `เรียกทีมผู้เชี่ยวชาญ` | `references/use-act-as.md` |
| `Use Comply` | `use-comply`, `Comply`, `comply`, `ใช้ Comply`, `ทำ Comply`, `แตกเฟส`, `ทำตารางเปอร์เซ็นต์` | `references/use-comply.md` |
| `Use Summary` | `use-summary`, `Summary`, `summary`, `ใช้ Summary`, `สรุป`, `สรุปลิงก์`, `วิเคราะห์บทความ`, `สรุปข้อมูล` | `references/use-summary.md` |
| `Use Scan Feature` | `use-scan-feature`, `Scan Feature`, `scan-feature`, `สแกนฟีเจอร์`, `ตรวจฟีเจอร์`, `บัญชีฟีเจอร์` | `references/use-scan-feature.md` |
| `Use AI Relay` | `use-ai-relay`, `AI Relay`, `ai-relay`, `ใช้ AI Relay`, `สายพาน AI`, `สายพานส่งต่องาน AI`, `Claude วางแผน Grok โค้ด`, `ให้ AI ตัวอื่นโค้ดแล้ว Claude ตรวจ` | `references/use-ai-relay.md` |
| `Use Viber Structure` | `use-viber-structure`, `Viber Structure`, `viber-structure`, `ใช้ Viber Structure`, `โครงสร้าง Viber`, `วางโครงสร้าง Viber Code`, `วางแผน Viber Code`, `Vibe Code Enterprise` | `references/use-viber-structure.md` |
| `Use Viber Audit` | `use-viber-audit`, `Viber Audit`, `viber-audit`, `Use Viber Standard Audit`, `Use Viber Compliance`, `ใช้ Viber Audit`, `ตรวจ Viber Standard`, `ตรวจ Viber Enterprise`, `ตรวจมาตรฐาน Viber`, `Viber Enterprise Standard` | `references/use-viber-audit.md` |
| `Use Impeccable` | `use-impeccable`, `Impeccable`, `ใช้ Impeccable`, `ตรวจ UI Slop`, `แก้ AI Slop` | `references/use-impeccable.md` |
| `Use Blog Auto` | `use-blog-auto`, `Blog Auto`, `blog-auto`, `ใช้ Blog Auto`, `เขียนบล็อกอัตโนมัติ`, `ทำบล็อกจากงานนี้`, `ส่งเข้า One Man Fleet` | `references/use-blog-auto.md` |
| `Use WOW Resource` | `use-wow-resource`, `WOW Resource`, `wow-resource`, `ใช้ WOW Resource`, `ใช้ WOW`, `WOW Layout`, `WOW Menu`, `WOW Script`, `WOW Design`, `WOW Web Engine` | `references/use-wow-resource.md` |
| `Use Flow Guardian` | `use-flow-guardian`, `Flow Guardian`, `Safe Flow`, `New Chat Gate`, `ใช้ Flow Guardian`, `ใช้ Safe Flow`, `เปิด Flow Guardian`, `ตรวจ worktree`, `กัน AI แก้งานทับกัน` | `references/use-flow-guardian.md` |
| `Use New Chat` | `use-new-chat`, `Start New Chat`, `New Chat Startup`, `Initialize Hermes Agent chat`, `เริ่ม New Chat`, `เปิด New Chat`, `เริ่มแชทใหม่`, `เปิดแชทใหม่` | `references/use-new-chat.md` |
| `Use Close Chat` | `use-close-chat`, `Close Chat`, `close-chat`, `ใช้ Close Chat`, `ปิดแชท`, `ปิดงานแชท`, `จบแชท` | `references/use-close-chat.md` |
| `Use Save Git` | `use-save-git`, `Save Git`, `save-git`, `ใช้ Save Git`, `เซฟ Git`, `ก่อน push`, `ก่อน merge`, `ก่อน deploy`, `Git Safe Flow`, `GitLab Deploy Safe Flow`, `Use GitLab Deploy Safe Flow`, `Use Ship Gate` | `references/use-save-git.md` |
| `Use Merge to Production` | `use-merge-to-production`, `Merge to Production`, `merge-to-production`, `ใช้ Merge to Production`, `ขึ้น production`, `deploy production`, `Ship to Production` | `references/use-merge-to-production.md` |
| `Use Continue` | `use-continue`, `Continue`, `continue`, `ทำต่อ`, `ทำต่อเอง`, `ทำงานต่อ`, `ทำต่ออัตโนมัติ`, `ไม่ต้องรอผม`, legacy: `Go to Sleep`, `go-to-sleep`, `Sleep Mode`, `sleep-mode`, `เข้าโหมดนอน`, `โหมดนอน` | `references/use-continue.md` |
| `Use Move Folder` | `use-move-folder`, `Move Folder`, `move-folder`, `movefolder`, `ใช้ Move Folder`, `ย้ายโฟลเดอร์`, `จัดเรียง Folder`, `จัดเรียงโฟลเดอร์` | `references/use-move-folder.md` |
| `Review Chat` | `review-chat`, `Chat Review`, `chat-review`, `รีวิวแชท`, `ตรวจแชท`, `สรุปส่งต่อ`, `สรุปเปิดแชทใหม่` | `references/review-chat.md` |
| `Use AI Pair` | `use-ai-pair`, `AI Pair`, `ai-pair`, `Use Pair AI`, `Pair AI`, `pair-ai`, `ใช้ AI Pair`, `ใช้ Pair AI`, `จับคู่ AI เขียนตรวจ`, `ทีม AI สามตัว` | `references/use-ai-pair.md` |
| `Use Business Plan` | `use-business-plan`, `Business Plan`, `business-plan`, `ใช้ Business Plan`, `รีวิวโจทย์ธุรกิจ`, `วางแผนธุรกิจ`, `วางแผนการตลาด`, `วางแผน Pitch`, `งานประมูล` | `references/use-business-plan.md` |
| `Use SaaS Opus Master Prompt` | `use-saas-opus-master-prompt`, `SaaS Opus Prompt`, `Opus SaaS Plan`, `Opus SaaS Master Prompt`, `ส่ง prompt SaaS Opus`, `prompt วางแผน SaaS`, `prompt ธุรกิจ SaaS`, `prompt pitch SaaS`, `prompt SaaS แบบละเอียดที่สุด` | `references/use-saas-opus-master-prompt.md` |
| `Use BusinessPlan` | `use-businessplan`, `Use BusinessPlan File`, `Use Project BusinessPlan`, `BusinessPlan File`, `ใช้ BusinessPlan`, `สร้างไฟล์แผนธุรกิจ`, `สแกนแผนธุรกิจ project`, `อัปเดตแผนธุรกิจ project` | `references/use-businessplan.md` |
| `Use OverviewProgress` | `use-overviewprogress`, `Use Overview Progress`, `ใช้ OverviewProgress`, `สร้างไฟล์ภาพรวมงาน`, `อัปเดตภาพรวม project`, `ภาพรวมความคืบหน้า` | `references/use-overviewprogress.md` |
| `Use FeatureSpec` | `use-featurespec`, `Use Feature Spec`, `ใช้ FeatureSpec`, `สแกนฟีเจอร์ project`, `บัญชีฟีเจอร์`, `อัปเดตรายการฟีเจอร์` | `references/use-featurespec.md` |
| `Use DesignSystem` | `use-designsystem`, `Use Design System File`, `ใช้ DesignSystem`, `สร้างไฟล์ Design System`, `อัปเดต Design System project`, `ตรวจดีไซน์ตามมาตรฐาน` | `references/use-designsystem.md` |
| `Use Create Design System` / `Use Design System` | `use-create-design-system`, `Create Design System`, `create-design-system`, `Use Design System`, `use-design-system`, `ใช้ Design System`, `ใช้ Create Design System`, `สร้าง Design System`, `ทำ Design System มาตรฐาน`, `วาง Design System ให้โปรเจกต์` | `references/use-create-design-system.md` |
| `Use Hermes Structure` | `use-hermes-structure`, `Hermes Structure`, `ใช้ Hermes Structure`, `มาตรฐานกลาง Hermes` | `references/use-hermes-structure.md` |
| `Use Create Content` | `use-create-content`, `Create Content`, `create-content`, `ใช้ Create Content`, `สร้างคอนเทนต์จากแชท`, `แปลงแชทเป็นคอนเทนต์`, `ทำ Content Master` | `references/use-create-content.md` |
| `Use QA QC` / `Use QC QA` | `use-qa-qc`, `use-qc-qa`, `Use QAQC`, `Use QCQA`, `QA QC`, `QC QA`, `ใช้ QA QC`, `ใช้ QC QA`, `ตรวจคุณภาพงาน`, `สแกนคุณภาพโปรเจกต์`, `สแกน QA`, `ตรวจงานก่อนส่งมอบ` | `references/use-qa-qc.md` |
| `Use SonarQube` | `use-sonarqube`, `SonarQube`, `ใช้ SonarQube`, `สแกน SonarQube`, `ตรวจโค้ดด้วย SonarQube` | `references/use-sonarqube.md` |

## How To Use

When the user invokes a shortcut:

1. Read the mapped prompt file in full.
2. Apply the prompt to the user's current task or the task text that follows the shortcut.
3. If the shortcut is invoked without a target task, ask what task the user wants to apply it to.
4. Follow any safety or approval constraints inside the loaded prompt exactly.

## Important Behavior

For `Use Act-As`, the loaded prompt requires deep role definition and work decomposition, and explicitly says not to create files until the user approves. Respect that constraint even if the surrounding task sounds implementation-ready.

For `Use Comply`, build phase-level plans with detailed issue checklists, numeric completion percentages, and localhost/VPS verification before delivery.

For `Use Summary`, summarize and analyze user-provided links plus content, present routing options first, and do not write to memory, KITS, registry, or files until the owner approves unless the owner explicitly says to choose and proceed.

For `Use Scan Feature`, scan the real repository phase by phase, refuse to claim any feature without reading evidence, label every capability as real/partial/mock/planned/blocked/unknown, stop at every required gate, and produce only a Thai feature/capability extraction document. Do not create marketing, SWOT, pricing, GTM, or roadmap output under this shortcut.


For `Use AI Pair`, default to the 3-AI pilot when context is sufficient: Claude plans/final-reviews, Codex writes, and Qwen reviews read-only. Do not stop by asking whether to create a brief or whether to proceed when the next safe step is obvious; create the coder brief, reviewer packet, and handoff immediately when file writes are allowed, or print them in chat when they are not. Ask only when the target repo/task/branch is unknowable, a risky write/deploy lacks approval, or a required secret/account/runtime is unavailable. Keep the reviewer read-only by default, route review through controlled diff/brief/evidence, and use GitLab Merge Request/CI as the final gate.

For `Use AI Relay`, load `references/use-ai-relay.md` and `references/ai-relay-catalog.md`. Honor the owner's mode from Use New Chat without asking twice: mode 1 assigns separate AIs to study/plan, production, and review; mode 2 uses a primary AI to produce the study/analysis output and a second AI to review it before acceptance. If no mode was supplied, ask once. Every code call also requires a fresh task-scoped Write Permit from Use New Chat; never reuse one for a new request or expanded path set. Fable/Faber/Fiber 5 is removed from the active path. Use `relay-call --role review` for AI reviews so Codex is read-only, silence alone does not stop it, one compact retry stays under the same issue, retry suffixes cannot reset counters, and concurrent duplicate work returns `already_running`. The same reviewer plus review method may fail at most twice per root issue; after that split the findings and switch to deterministic gates or a different-vendor reviewer, never a third identical review. Use `gate-run` for real verification; never treat an AI claim or partial timeout output as verified without a gate row.

For `Use Business Plan`, review the owner's raw business/marketing/pitch/tender/website question before execution, choose the right business modules and expert roles, build phase and issue checklists, ask for missing inputs first, and do not create files or durable writes until approved.

For `Use SaaS Opus Master Prompt`, send the owner-approved detailed one-file Opus 4.8 master prompt for SaaS business, product, marketing, pricing, pitch, WOW proof, and portfolio decision work. Do not replace it with a short summary.

For `Use Viber Structure`, turn the Viber Code / Vibe Code Enterprise playbook into a project structure, artifact matrix, phase/issue tracker, and quality-gate plan. Require spec before code, numeric compliance, and real verification evidence before claiming completion.

For `Use Viber Audit`, inspect one or many real Viber Project repos against the full Viber Enterprise Standard, score artifact/gate/tracking/verification coverage from evidence only, identify missing critical work, and create or update per-project tracking when authorized.

For `Use Impeccable`, use exactly one owner-facing shortcut for Impeccable UI quality work. Read the mapped prompt, infer the target from context when possible, ask one short target question only when needed, and let the AI choose whether to install, scan, explain, fix blocking UI issues, or plan UI-debt cleanup. Do not expose multiple Impeccable sub-shortcuts to the owner.

For `Use Blog Auto`, extract useful work knowledge into a One Man Fleet blog route, run privacy review first, decide whether to create a new post or update an existing one, use English public plus Thai internal summary by default, create only drafts until owner approval, record Obsidian index/traceability, and hand off platform drafts to Content Factory without auto-posting.

For `Use WOW Resource`, read the mapped prompt, route through WOW System and Web Design Intelligence, select resources based on the project goal, reject mismatched/generic options, and transform the selected patterns into project-specific layout/design/script direction. Do not copy scripts or visual patterns directly.

For `Use Flow Guardian`, apply Home OS Agent safe workflow before project work: resolve and report the registered staff+project folder, branch, and dirty status; for new writable work propose only a branch inside that existing folder. Never propose or create a new worktree. Require no-write audit, approval gates, verification, tracking, and handoff when applicable.

For `Use New Chat`, run the startup checklist plus AI Relay Startup and `hermes-hook-doctor` from a single invocation before any readiness response. The doctor must prove the plain-language, independent-review, and prompt-evidence gates block their bad fixtures; file presence alone is insufficient. Inspect current folder, Git root, registered staff+project folder, folder match, branch, dirty status, local/remote/VPS equality, service/endpoint, and Relay readiness when applicable. After the report, ask the owner to choose mode 1 (separate AIs for study/plan, production, and review) or mode 2 (a primary AI creates the study/analysis output and a second AI reviews it), then pass that choice to AI Relay without asking twice. Before every distinct writable request in the same chat, re-check branch/status/claim and issue a task+scope+paths Write Permit; approval never carries to another task. Load `references/use-new-chat-conditional-gates.md` only when its legacy-memory, optional-check, or team-routing condition is present. Never propose or create a new worktree; new writable work may create only a branch inside the clean registered folder after approval. Return a New Chat Startup Report, not only "ready for commands".

For `Use Close Chat`, close the chat by checking real work status, quality gate evidence, commit/push/merge handoff, and memory writes before returning CLOSED_CLEAN, CLOSED_WITH_PENDING, or NEED_OWNER_ACTION_BEFORE_CLOSE. It does not push, merge, or deploy by itself.

For `Use Save Git`, enforce the safe Git/GitLab/VPS shipping gate before push, merge, deploy, or final readiness claims: inspect project path, branch, dirty files, worktrees, remote, Local/GitLab/VPS SHA, health endpoint, and return a SAFE/STOP decision. Do not read or reveal `.env` or secrets. Stop on dirty worktree, wrong branch, wrong remote, missing registry, SHA mismatch, deploy-not-allowed, route missing, secret risk, or unknown state.

For `Use Merge to Production`, treat it as a merger-only production path. Confirm the caller and target are allowed, run the Save Git merge gate and ship gate, deploy only from the approved remote/branch, and stop on any unknown state.

For `Use Continue`, continue autonomously through phases, make best-judgment choices when selection is needed, require each phase to reach 100%, and provide a final phase percentage table for review. Treat `Go to Sleep` and sleep-related names only as legacy aliases for this same behavior.

For `Use Move Folder`, load `references/use-move-folder.md`, then read the live VPS registry under `/home/linux-nat/.codex/use-move-folder/project-registry` before doing any cleanup, folder move, retention review, or disk-space work. Do not claim the shortcut is missing just because it is stored in Codex runtime state. Do not scan protected/no-touch roots or mutate anything unless the owner gives exact approval.

For `Review Chat`, review the current conversation for pending work, update the relevant HermesNous status files when appropriate, provide a clean new-chat starter message, and warn that exact context-window percentage may be unavailable unless the UI exposes it.

For `Use BusinessPlan`, create or update the per-project business memory files under `.project/` after reading the real repo and the existing business files. Keep it separate from `Use Business Plan`, which is the raw business-planning prompt.

For `Use OverviewProgress`, create or update `.project/OverviewProgress.md` using the Memory Schema v1.2 top sections and prove the file is not hidden by git ignore rules.

For `Use FeatureSpec`, scan the real code and record feature status as real, partial, mock, planned, blocked, or unknown with path evidence.

For `Use DesignSystem`, create or update the per-project design-system memory file. Do not confuse it with `Use Create Design System`, which builds or migrates a project-wide design system.

For `Use Create Design System`, read the project first, then apply the approved design-system standard with owner color approval and measured adoption gates before code changes.

For `Use Hermes Structure`, route the owner to the Hermes standard workflow and use the safe apply tools from the central standard set. Do not edit VPS/global files directly unless the owner approves that exact action.

For `Use Create Content`, convert the current chat or source material into a privacy-reviewed Content Master draft, then hand off to Blog Auto or the content factory without publishing.

For `Use QA QC` (or `Use QC QA`), open a two-axis quality-scan menu (project progress 25/50/75/100% × 16 check categories Q01-Q16, multi-select, Scan All last behind a confirm gate), run a cross-vendor scan pipeline (primary scanner + counter-scanner from a different vendor, fixer = third AI, reviewer ≠ fixer, verified = gate-run rows only), produce a severity table, then write `.project/qaqc-scan.md` before any fixes.

For `Use SonarQube`, analyze an existing project with the owner's already-installed SonarQube instance. Read the project rules, detect its build system, verify server status and credentials without revealing secrets, run the matching scanner, confirm the server-side analysis through the API, and return a Thai report. Never install or upgrade the SonarQube server through this shortcut, and never change source code without separate owner approval.

## Source Files

- `Prompt Shortcuts.md`: Obsidian index note for all shortcuts.
- `ai-context/prompt-shortcut-registry.md`: shared registry for non-Codex adapters.
- `references/use-act-as.md`: full prompt for `Use Act-As`.
- `references/use-comply.md`: full prompt for `Use Comply`.
- `references/use-summary.md`: full prompt for `Use Summary`.
- `references/use-scan-feature.md`: full prompt for `Use Scan Feature`.
- `references/use-ai-pair.md`: full prompt for `Use AI Pair`.
- `references/use-ai-relay.md`: full prompt for `Use AI Relay`.
- `references/ai-relay-catalog.md`: AI Relay catalog and routing rules.
- `references/use-business-plan.md`: full prompt for `Use Business Plan`.
- `references/use-saas-opus-master-prompt.md`: full prompt for `Use SaaS Opus Master Prompt`.
- `references/use-viber-structure.md`: full prompt for `Use Viber Structure`.
- `references/use-viber-audit.md`: full prompt for `Use Viber Audit`.
- `references/use-impeccable.md`: full prompt for `Use Impeccable`.
- `references/use-blog-auto.md`: full prompt for `Use Blog Auto`.
- `references/use-wow-resource.md`: full prompt for `Use WOW Resource`.
- `references/use-flow-guardian.md`: full prompt for `Use Flow Guardian`.
- `references/use-new-chat.md`: full prompt for `Use New Chat`.
- `references/use-close-chat.md`: full prompt for `Use Close Chat`.
- `references/use-save-git.md`: full prompt for `Use Save Git`.
- `references/use-merge-to-production.md`: full prompt for `Use Merge to Production`.
- `references/use-continue.md`: full prompt for `Use Continue`.
- `references/use-move-folder.md`: full prompt for `Use Move Folder`.
- `references/go-to-sleep.md`: legacy alias note for old `Go to Sleep` invocations.
- `references/review-chat.md`: full prompt for `Review Chat`.
- `references/use-businessplan.md`: full prompt for `Use BusinessPlan`.
- `references/use-overviewprogress.md`: full prompt for `Use OverviewProgress`.
- `references/use-featurespec.md`: full prompt for `Use FeatureSpec`.
- `references/use-designsystem.md`: full prompt for `Use DesignSystem`.
- `references/use-create-design-system.md`: full prompt for `Use Create Design System`.
- `references/use-hermes-structure.md`: full prompt for `Use Hermes Structure`.
- `references/use-create-content.md`: full prompt for `Use Create Content`.
- `references/use-qa-qc.md`: full prompt for `Use QA QC` / `Use QC QA`.
- `references/use-sonarqube.md`: full prompt for recurring `Use SonarQube` project analysis.
- `references/sonarqube-vps-install-for-cursor.md`: one-time Cursor prompt for installing SonarQube Community Build on the owner's VPS; this is not a shortcut.

## Graph Links

- Parent hub: [[skills/README|skills]]
- Router: [[00-Center/docs/AI_SKILL_ROUTER|AI Skill Router]]
- Graph: [[00-Center/docs/SKILL_GRAPH|Skill Graph]]
