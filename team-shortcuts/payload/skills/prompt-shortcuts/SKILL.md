---
name: prompt-shortcuts
description: Use this skill when the user invokes a reusable prompt shortcut, especially "Use Act-As", "Use Comply", "Use Summary", "Use Scan Feature", "Use Opus Plan", "Use AI Pair", "Use Pair AI", "Use Business Plan", "Use SaaS Opus Master Prompt", "Use Viber Structure", "Use Viber Audit", "Use Impeccable", "Use Blog Auto", "Use WOW Resource", "Use Flow Guardian", "Use New Chat", "Use Save Git", "Use Continue", "Use Move Folder", "Review Chat", "use-act-as", "use-comply", "use-summary", "use-scan-feature", "use-opus-plan", "use-ai-pair", "pair-ai", "use-business-plan", "use-saas-opus-master-prompt", "use-viber-structure", "use-viber-audit", "use-impeccable", "use-blog-auto", "use-wow-resource", "use-flow-guardian", "use-new-chat", "use-save-git", "save-git", "use-continue", "use-move-folder", "go-to-sleep", "review-chat", "Act-As", "Comply", "Summary", "Scan Feature", "Opus Plan", "AI Pair", "Pair AI", "Business Plan", "SaaS Opus Prompt", "Opus SaaS Plan", "Opus SaaS Master Prompt", "ส่ง prompt SaaS Opus", "prompt วางแผน SaaS", "prompt ธุรกิจ SaaS", "prompt pitch SaaS", "prompt SaaS แบบละเอียดที่สุด", "Viber Structure", "Viber Audit", "Viber Enterprise Standard", "Impeccable", "ใช้ Impeccable", "ตรวจ UI Slop", "แก้ AI Slop", "Blog Auto", "ใช้ Blog Auto", "เขียนบล็อกอัตโนมัติ", "ทำบล็อกจากงานนี้", "ส่งเข้า Hi Logic Labs", "WOW Resource", "WOW Layout", "WOW Menu", "WOW Script", "WOW Design", "WOW Web Engine", "Flow Guardian", "Safe Flow", "New Chat Gate", "Start New Chat", "Initialize Hermes Agent chat", "เริ่ม New Chat", "เปิด New Chat", "เริ่มแชทใหม่", "เปิดแชทใหม่", "Save Git", "เซฟ Git", "ก่อน push", "ก่อน merge", "ก่อน deploy", "Git Safe Flow", "GitLab Deploy Safe Flow", "Use Ship Gate", "ตรวจ worktree", "กัน AI แก้งานทับกัน", "ใช้ Pair AI", "จับคู่ AI เขียนตรวจ", "Continue", "ทำต่อ", "ทำต่อเอง", "ทำงานต่อ", "ไม่ต้องรอผม", "Move Folder", "move-folder", "movefolder", "ใช้ Move Folder", "ย้ายโฟลเดอร์", "จัดเรียง Folder", "จัดเรียงโฟลเดอร์", "Go to Sleep", "Sleep Mode", "Chat Review", or asks to load a standard prompt from HermesNous instead of pasting it manually.
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
| `Use AI Pair` | `use-ai-pair`, `AI Pair`, `ai-pair`, `Use Pair AI`, `Pair AI`, `pair-ai`, `ใช้ AI Pair`, `ใช้ Pair AI`, `จับคู่ AI เขียนตรวจ` | `references/use-ai-pair.md` |
| `Use Business Plan` | `use-business-plan`, `Business Plan`, `business-plan`, `ใช้ Business Plan`, `รีวิวโจทย์ธุรกิจ`, `วางแผนธุรกิจ`, `วางแผนการตลาด`, `วางแผน Pitch`, `งานประมูล` | `references/use-business-plan.md` |
| `Use SaaS Opus Master Prompt` | `use-saas-opus-master-prompt`, `SaaS Opus Prompt`, `Opus SaaS Plan`, `Opus SaaS Master Prompt`, `ส่ง prompt SaaS Opus`, `prompt วางแผน SaaS`, `prompt ธุรกิจ SaaS`, `prompt pitch SaaS`, `prompt SaaS แบบละเอียดที่สุด` | `references/use-saas-opus-master-prompt.md` |
| `Use Viber Structure` | `use-viber-structure`, `Viber Structure`, `viber-structure`, `ใช้ Viber Structure`, `โครงสร้าง Viber`, `วางโครงสร้าง Viber Code`, `วางแผน Viber Code`, `Vibe Code Enterprise` | `references/use-viber-structure.md` |
| `Use Viber Audit` | `use-viber-audit`, `Viber Audit`, `viber-audit`, `Use Viber Standard Audit`, `Use Viber Compliance`, `ใช้ Viber Audit`, `ตรวจ Viber Standard`, `ตรวจ Viber Enterprise`, `ตรวจมาตรฐาน Viber`, `Viber Enterprise Standard` | `references/use-viber-audit.md` |
| `Use Impeccable` | `use-impeccable`, `Impeccable`, `ใช้ Impeccable`, `ตรวจ UI Slop`, `แก้ AI Slop` | `references/use-impeccable.md` |
| `Use Blog Auto` | `use-blog-auto`, `Blog Auto`, `blog-auto`, `ใช้ Blog Auto`, `เขียนบล็อกอัตโนมัติ`, `ทำบล็อกจากงานนี้`, `ส่งเข้า Hi Logic Labs` | `references/use-blog-auto.md` |
| `Use WOW Resource` | `use-wow-resource`, `WOW Resource`, `wow-resource`, `ใช้ WOW Resource`, `ใช้ WOW`, `WOW Layout`, `WOW Menu`, `WOW Script`, `WOW Design`, `WOW Web Engine` | `references/use-wow-resource.md` |
| `Use Flow Guardian` | `use-flow-guardian`, `Flow Guardian`, `Safe Flow`, `New Chat Gate`, `ใช้ Flow Guardian`, `ใช้ Safe Flow`, `เปิด Flow Guardian`, `ตรวจ worktree`, `กัน AI แก้งานทับกัน` | `references/use-flow-guardian.md` |
| `Use New Chat` | `use-new-chat`, `Start New Chat`, `New Chat Startup`, `Initialize Hermes Agent chat`, `เริ่ม New Chat`, `เปิด New Chat`, `เริ่มแชทใหม่`, `เปิดแชทใหม่` | `references/use-new-chat.md` |
| `Use Save Git` | `use-save-git`, `Save Git`, `save-git`, `ใช้ Save Git`, `เซฟ Git`, `ก่อน push`, `ก่อน merge`, `ก่อน deploy`, `Git Safe Flow`, `GitLab Deploy Safe Flow`, `Use GitLab Deploy Safe Flow`, `Use Ship Gate` | `references/use-save-git.md` |
| `Use Continue` | `use-continue`, `Continue`, `continue`, `ทำต่อ`, `ทำต่อเอง`, `ทำงานต่อ`, `ทำต่ออัตโนมัติ`, `ไม่ต้องรอผม`, legacy: `Go to Sleep`, `go-to-sleep`, `Sleep Mode`, `sleep-mode`, `เข้าโหมดนอน`, `โหมดนอน` | `references/use-continue.md` |
| `Use Move Folder` | `use-move-folder`, `Move Folder`, `move-folder`, `movefolder`, `ใช้ Move Folder`, `ย้ายโฟลเดอร์`, `จัดเรียง Folder`, `จัดเรียงโฟลเดอร์` | `references/use-move-folder.md` |
| `Review Chat` | `review-chat`, `Chat Review`, `chat-review`, `รีวิวแชท`, `ตรวจแชท`, `สรุปส่งต่อ`, `สรุปเปิดแชทใหม่` | `references/review-chat.md` |

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

For `Use Business Plan`, review the owner's raw business/marketing/pitch/tender/website question before execution, choose the right business modules and expert roles, build phase and issue checklists, ask for missing inputs first, and do not create files or durable writes until approved.

For `Use SaaS Opus Master Prompt`, send the owner-approved detailed one-file Opus 4.8 master prompt for SaaS business, product, marketing, pricing, pitch, WOW proof, and portfolio decision work. Do not replace it with a short summary.

For `Use Viber Structure`, turn the Viber Code / Vibe Code Enterprise playbook into a project structure, artifact matrix, phase/issue tracker, and quality-gate plan. Require spec before code, numeric compliance, and real verification evidence before claiming completion.

For `Use Viber Audit`, inspect one or many real Viber Project repos against the full Viber Enterprise Standard, score artifact/gate/tracking/verification coverage from evidence only, identify missing critical work, and create or update per-project tracking when authorized.

For `Use Impeccable`, use exactly one owner-facing shortcut for Impeccable UI quality work. Read the mapped prompt, infer the target from context when possible, ask one short target question only when needed, and let the AI choose whether to install, scan, explain, fix blocking UI issues, or plan UI-debt cleanup. Do not expose multiple Impeccable sub-shortcuts to the owner.

For `Use Blog Auto`, extract useful work knowledge into a Hi Logic Labs blog route, run privacy review first, decide whether to create a new post or update an existing one, use English public plus Thai internal summary by default, create only drafts until owner approval, record Obsidian index/traceability, and hand off platform drafts to Content Factory without auto-posting.

For `Use WOW Resource`, read the mapped prompt, route through WOW System and Web Design Intelligence, select resources based on the project goal, reject mismatched/generic options, and transform the selected patterns into project-specific layout/design/script direction. Do not copy scripts or visual patterns directly.

For `Use Flow Guardian`, apply Home OS Agent safe workflow before project work: report current worktree, branch, and dirty status; ask branch/worktree choice for new features; require no-write audit, approval gates, verification, tracking, and handoff when applicable.

For `Use New Chat`, run the startup checklist before any readiness response: inspect project path, worktree, branch, dirty status, local/remote/VPS equality, service/endpoint when applicable, and return a New Chat Startup Report. Do not answer only "ready for commands".

For `Use Save Git`, enforce the safe Git/GitLab/VPS shipping gate before push, merge, deploy, or final readiness claims: inspect project path, branch, dirty files, worktrees, remote, Local/GitLab/VPS SHA, health endpoint, and return a SAFE/STOP decision. Do not read or reveal `.env` or secrets. Stop on dirty worktree, wrong branch, wrong remote, missing registry, SHA mismatch, deploy-not-allowed, route missing, secret risk, or unknown state.

For `Use Continue`, continue autonomously through phases, make best-judgment choices when selection is needed, require each phase to reach 100%, and provide a final phase percentage table for review. Treat `Go to Sleep` and sleep-related names only as legacy aliases for this same behavior.

For `Use Move Folder`, load `references/use-move-folder.md`, then read the live VPS registry under `/home/linux-nat/.codex/use-move-folder/project-registry` before doing any cleanup, folder move, retention review, or disk-space work. Do not claim the shortcut is missing just because it is stored in Codex runtime state. Do not scan protected/no-touch roots or mutate anything unless the owner gives exact approval.

For `Review Chat`, review the current conversation for pending work, update the relevant HermesNous status files when appropriate, provide a clean new-chat starter message, and warn that exact context-window percentage may be unavailable unless the UI exposes it.

## Source Files

- `Prompt Shortcuts.md`: Obsidian index note for all shortcuts.
- `ai-context/prompt-shortcut-registry.md`: shared registry for non-Codex adapters.
- `references/use-act-as.md`: full prompt for `Use Act-As`.
- `references/use-comply.md`: full prompt for `Use Comply`.
- `references/use-summary.md`: full prompt for `Use Summary`.
- `references/use-scan-feature.md`: full prompt for `Use Scan Feature`.
- `references/use-ai-pair.md`: full prompt for `Use AI Pair`.
- `references/use-business-plan.md`: full prompt for `Use Business Plan`.
- `references/use-saas-opus-master-prompt.md`: full prompt for `Use SaaS Opus Master Prompt`.
- `references/use-viber-structure.md`: full prompt for `Use Viber Structure`.
- `references/use-viber-audit.md`: full prompt for `Use Viber Audit`.
- `references/use-impeccable.md`: full prompt for `Use Impeccable`.
- `references/use-blog-auto.md`: full prompt for `Use Blog Auto`.
- `references/use-wow-resource.md`: full prompt for `Use WOW Resource`.
- `references/use-flow-guardian.md`: full prompt for `Use Flow Guardian`.
- `references/use-new-chat.md`: full prompt for `Use New Chat`.
- `references/use-save-git.md`: full prompt for `Use Save Git`.
- `references/use-continue.md`: full prompt for `Use Continue`.
- `references/use-move-folder.md`: full prompt for `Use Move Folder`.
- `references/go-to-sleep.md`: legacy alias note for old `Go to Sleep` invocations.
- `references/review-chat.md`: full prompt for `Review Chat`.

## Graph Links

- Parent hub: [[skills/README|skills]]
- Router: [[00-Center/docs/AI_SKILL_ROUTER|AI Skill Router]]
- Graph: [[00-Center/docs/SKILL_GRAPH|Skill Graph]]
