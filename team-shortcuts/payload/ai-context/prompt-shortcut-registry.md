---
title: Prompt Shortcut Registry
aliases:
  - Multi-AI Prompt Shortcuts
  - Shortcut Registry
  - Prompt Registry
tags:
  - ai-context
  - prompt-shortcuts
  - multi-ai
  - adapters
status: active
created: 2026-05-24
updated: 2026-06-22
source_of_truth: skills/prompt-shortcuts
codex_runtime_path: /Users/rattanasak/.codex/skills/prompt-shortcuts
---

# Prompt Shortcut Registry

This is the shared prompt shortcut contract for Codex, Qwen, Claude Code, Gemini, Cursor, and future AI tools.

## Runtime Rule

When the user invokes a shortcut, the AI must read the mapped prompt file and apply it to the current task. Do not paraphrase, invent, or partially apply the shortcut.

## Shortcut Map

| Shortcut | Aliases | Prompt File | Purpose |
|---|---|---|---|
| `Use Act-As` | `use-act-as`, `Use Act As`, `Act-As`, `act-as`, `ใช้ Act-As`, `กำหนดบทบาท`, `เรียกทีมผู้เชี่ยวชาญ` | [[skills/prompt-shortcuts/references/use-act-as|use-act-as]] | Define expert roles, split work by role, and avoid file creation until approval when required. |
| `Use Comply` | `use-comply`, `Comply`, `comply`, `ใช้ Comply`, `ทำ Comply`, `แตกเฟส`, `ทำตารางเปอร์เซ็นต์` | [[skills/prompt-shortcuts/references/use-comply|use-comply]] | Build phase plans, detailed issue checklists, numeric compliance, and verification. |
| `Use Summary` | `use-summary`, `Summary`, `summary`, `ใช้ Summary`, `สรุป`, `สรุปลิงก์`, `วิเคราะห์บทความ`, `สรุปข้อมูล` | [[skills/prompt-shortcuts/references/use-summary|use-summary]] | Summarize and analyze user-provided links plus content, propose routing options, and wait for owner approval before writing to durable systems. |
| `Use Scan Feature` | `use-scan-feature`, `Scan Feature`, `scan-feature`, `สแกนฟีเจอร์`, `ตรวจฟีเจอร์`, `บัญชีฟีเจอร์` | [[skills/prompt-shortcuts/references/use-scan-feature|use-scan-feature]] | Scan a real repo phase by phase and extract a Thai feature/capability spec with evidence, real/mock status, Reality Matrix, and Hidden Gems. |
| `Use AI Relay` | `use-ai-relay`, `AI Relay`, `ai-relay`, `ใช้ AI Relay`, `สายพาน AI`, `สายพานส่งต่องาน AI`, `Claude วางแผน Grok โค้ด`, `ให้ AI ตัวอื่นโค้ดแล้ว Claude ตรวจ` | [[skills/prompt-shortcuts/references/use-ai-relay|use-ai-relay]] | Token-saving relay: Claude only plans/picks-coder/reviews/decides while a cheaper non-Claude coder (Grok default, Codex for hard backend, Gemini for big UI) writes code on a branch; loop fix-review until 100% before the phase gate opens. Includes per-project permission/bridge setup checklist and proven connection notes. |
| `Use Viber Structure` | `use-viber-structure`, `Viber Structure`, `viber-structure`, `ใช้ Viber Structure`, `โครงสร้าง Viber`, `วางโครงสร้าง Viber Code`, `วางแผน Viber Code`, `Vibe Code Enterprise` | [[skills/prompt-shortcuts/references/use-viber-structure|use-viber-structure]] | Turn the Viber Code / Vibe Code Enterprise playbook into project structure, artifact matrices, phase/issue trackers, quality gates, and verification evidence rules. |
| `Use Viber Audit` | `use-viber-audit`, `Viber Audit`, `viber-audit`, `Use Viber Standard Audit`, `Use Viber Compliance`, `ใช้ Viber Audit`, `ตรวจ Viber Standard`, `ตรวจ Viber Enterprise`, `ตรวจมาตรฐาน Viber`, `Viber Enterprise Standard` | [[skills/prompt-shortcuts/references/use-viber-audit|use-viber-audit]] | Audit one or many Viber Project repos against the full Viber Enterprise Standard with evidence-based scoring, missing artifact/gate analysis, remediation issues, and updateable tracking. |
| `Use Impeccable` | `use-impeccable`, `Impeccable`, `ใช้ Impeccable`, `ตรวจ UI Slop`, `แก้ AI Slop` | [[skills/prompt-shortcuts/references/use-impeccable|use-impeccable]] | Use one simple owner-facing shortcut for Impeccable UI quality work; AI chooses whether to install, scan, ask for target, fix blocking issues, or plan UI debt cleanup. |
| `Use Blog Auto` | `use-blog-auto`, `Blog Auto`, `blog-auto`, `ใช้ Blog Auto`, `เขียนบล็อกอัตโนมัติ`, `ทำบล็อกจากงานนี้`, `ส่งเข้า Hi Logic Labs` | [[skills/prompt-shortcuts/references/use-blog-auto|use-blog-auto]] | Extract work knowledge into Hi Logic Labs draft-first blog planning, privacy review, Obsidian index, and Content Factory handoff without publishing before owner approval. |
| `Use WOW Resource` | `use-wow-resource`, `WOW Resource`, `wow-resource`, `ใช้ WOW Resource`, `ใช้ WOW`, `WOW Layout`, `WOW Menu`, `WOW Script`, `WOW Design`, `WOW Web Engine` | [[skills/prompt-shortcuts/references/use-wow-resource|use-wow-resource]] | Read WOW System and Web Design Intelligence resources from Obsidian, select suitable layout/design/script references for the project goal, reject mismatched options, and transform choices into project-specific direction without copying scripts directly. |
| `Use Flow Guardian` | `use-flow-guardian`, `Flow Guardian`, `Safe Flow`, `New Chat Gate`, `ใช้ Flow Guardian`, `ใช้ Safe Flow`, `เปิด Flow Guardian`, `ตรวจ worktree`, `กัน AI แก้งานทับกัน` | [[skills/prompt-shortcuts/references/use-flow-guardian|use-flow-guardian]] | Enforce Home OS Agent safe workflow: report worktree/branch/status, ask branch/worktree choice for new features, require no-write audit, approval, verification, tracking, and handoff. |
| `Use New Chat` | `use-new-chat`, `Start New Chat`, `New Chat Startup`, `Initialize Hermes Agent chat`, `เริ่ม New Chat`, `เปิด New Chat`, `เริ่มแชทใหม่`, `เปิดแชทใหม่` | [[skills/prompt-shortcuts/references/use-new-chat|use-new-chat]] | Start a new chat with real startup checks: project path, worktree, branch, dirty status, local/remote/VPS equality, service/endpoint, and a Thai readiness report before accepting new work. On open it must read the latest handoff + session log (pairs with Use Close Chat) so the AI does not forget prior work. |
| `Use Close Chat` | `use-close-chat`, `Close Chat`, `close-chat`, `ใช้ Close Chat`, `ปิดแชท`, `ปิดงานแชท`, `จบแชท` | [[skills/prompt-shortcuts/references/use-close-chat|use-close-chat]] | Close a chat safely and write durable session memory the next chat reads on open (fixes AI forgetting prior work). Pre-close gate asks: committed yet, push/merge/notify-merger, each task really verified vs claimed, anything the AI forgot, release claim. Writes short handoff + detailed session log (changed-files, decisions, pending, next owner). Does not push/merge itself; returns CLOSED_CLEAN / CLOSED_WITH_PENDING / NEED_OWNER_ACTION_BEFORE_CLOSE. |
| `Use Save Git` | `use-save-git`, `Save Git`, `save-git`, `ใช้ Save Git`, `เซฟ Git`, `ก่อน push`, `ก่อน merge`, `ก่อน deploy`, `Git Safe Flow`, `GitLab Deploy Safe Flow`, `Use GitLab Deploy Safe Flow`, `Use Ship Gate`, `Merge Gate`, `Pre-Merge Gate`, `Pre-Merge Production Gate` | [[skills/prompt-shortcuts/references/use-save-git|use-save-git]] | v2 · run one real 5-stage gate (local, mr, ci, vps-dryrun, production) via `save-git --stage merge-gate`/`ship-gate` reading per-project `.savegit.json`; return one decision token (SAFE_TO_MERGE / BLOCKED_DO_NOT_MERGE / OWNER_DECISION_REQUIRED / SAFE_TO_DEPLOY / PRODUCTION_VERIFIED / PRODUCTION_NOT_VERIFIED) plus a single Grid showing the real blocking layer, before the owner clicks merge. |
| `Use Merge to Production` | `use-merge-to-production`, `Merge to Production`, `merge-to-production`, `ใช้ Merge to Production`, `ขึ้น production`, `deploy production`, `Ship to Production` | [[skills/prompt-shortcuts/references/use-merge-to-production|use-merge-to-production]] | Merger-only shortcut to merge into the production branch then deploy to the real VPS: verify the caller is an allowlisted merger (default nat, namton, nam), run save-git merge-gate then ship-gate, deploy from origin/target only, and return one decision token plus a 6-stage Grid. Builds on Use Save Git; GitLab protected branch is the authoritative server-side enforcement. |
| `Use Continue` | `use-continue`, `Continue`, `continue`, `ทำต่อ`, `ทำต่อเอง`, `ทำงานต่อ`, `ทำต่ออัตโนมัติ`, `ไม่ต้องรอผม`, legacy: `Go to Sleep`, `go-to-sleep`, `Sleep Mode`, `sleep-mode`, `เข้าโหมดนอน`, `โหมดนอน` | [[skills/prompt-shortcuts/references/use-continue|use-continue]] | Continue autonomously, choose best options, and close every phase at 100%. |
| `Use Move Folder` | `use-move-folder`, `Move Folder`, `move-folder`, `movefolder`, `ใช้ Move Folder`, `ย้ายโฟลเดอร์`, `จัดเรียง Folder`, `จัดเรียงโฟลเดอร์` | [[skills/prompt-shortcuts/references/use-move-folder|use-move-folder]] | Route to the existing VPS folder cleanup and move workflow under `/home/linux-nat/.codex/use-move-folder/project-registry`; read its live checkpoint, no-touch policy, report rules, and latest scope evidence before any action. |
| `Review Chat` | `review-chat`, `Chat Review`, `chat-review`, `รีวิวแชท`, `ตรวจแชท`, `สรุปส่งต่อ`, `สรุปเปิดแชทใหม่` | [[skills/prompt-shortcuts/references/review-chat|review-chat]] | Review the current chat, identify pending work, update handoff, and prepare a clean continuation prompt. |

## Tool-Specific Loading

| Tool | Native route | Required behavior |
|---|---|---|
| Codex | `$prompt-shortcuts` skill via `/Users/rattanasak/.codex/skills/prompt-shortcuts` | Load the skill and then the mapped reference prompt. |
| Claude Code | `CLAUDE.md` project/global memory | Read this registry, then open the mapped prompt file from Obsidian. |
| Gemini CLI | `GEMINI.md` project/global memory | Read this registry, then open the mapped prompt file from Obsidian. |
| Qwen | `QWEN.md` or `AGENTS.md` project memory | Read this registry, then open the mapped prompt file from Obsidian. |
| Cursor | `.cursor/rules/obsidian-context.mdc` | Read this registry, then open the mapped prompt file from Obsidian. |

## Verification Standard

The setup is considered ready only when:

1. Codex skill symlink points to `skills/prompt-shortcuts`.
2. Every repo adapter file references this registry.
3. Adapter templates include this registry for future projects.
4. The health check fails if any required adapter cannot see prompt shortcuts.

## Project-Wide Status

As of 2026-05-28, the project-wide owner rules rollout points every real Viber Project AI instruction file to this registry. Agents must treat Thai aliases and English aliases as equivalent shortcut invocations.

As of 2026-05-28, the original four mapped prompt files are upgraded to detailed v2 prompts. As of 2026-05-29, `Use Summary` is added for link-plus-content intake review before durable writes. As of 2026-05-30, `Use Scan Feature` is added for evidence-first repo feature extraction. As of 2026-05-31, `Use Business Plan` is added for reusable business, marketing, pitch, tender, website, and strategy question review, and `Use WOW Resource` is added for WOW System plus Web Design Intelligence resource selection without direct script copying. As of 2026-06-01, `Use Viber Structure` is added for Viber Code / Vibe Code Enterprise project structure, artifact, tracker, and quality-gate generation, `Use Viber Audit` is added for evidence-based Viber Enterprise Standard checks across one or many projects, and `Use Impeccable` is added as the single owner-facing shortcut for Impeccable UI slop detection, installation, scanning, fixing, and review workflow. As of 2026-06-06, `Use Flow Guardian` is added for Home OS Agent startup discipline, worktree/branch safety, no-write audit, approval gates, verification, tracking, and handoff, and `Use New Chat` is added to force real startup checks before any “ready for commands” response. As of 2026-06-07, `Use Blog Auto` is added for Hi Logic Labs blog orchestration, privacy gate, Obsidian index, Content Factory handoff, and draft-first owner approval. As of 2026-06-07, `Go to Sleep` is renamed to `Use Continue`; the old sleep wording remains only as a legacy alias so AI tools continue work instead of interpreting the shortcut as a stop/sleep instruction. As of 2026-06-07, `Use Save Git` is added as the owner-facing safe Git/GitLab/VPS shipping gate before push, merge, deploy, and final SHA/health claims. As of 2026-06-08, `Use Save Git` v1.3 removes copyable bad examples and pairs the prompt with runtime gate output that returns `REMEDIATE_DIRTY_WORKTREE_NOW` plus required AI follow-up instead of stop-at-audit wording. As of 2026-06-08 (later same day), `Use Save Git` is rebuilt as v2 after reviewing 11 real project failures of merge/push-to-production loops: v1.3 is retired and the old action-based flow must not be used. v2 makes the gate a single runnable 5-stage check (`save_git_gate.py --stage merge-gate|ship-gate`) driven by a per-project `.savegit.json` adapter, verified to block on secret files, scope-bloat/wrong-target MRs, and failed/old-commit health endpoints, returning one decision token plus a Grid that names the real blocking layer. Redesign evidence: `95-Inbox-Lab/review/save-git-redesign/evidence-and-review.md`. As of 2026-06-22, `Use Move Folder` is added to route Hermes and other AI tools to the existing VPS cleanup/move workflow under `/home/linux-nat/.codex/use-move-folder/project-registry` instead of falsely treating the shortcut as missing. As of 2026-06-25, `Use Move Folder` is upgraded to v1.1 after a two-AI review (Claude draft plus GPT-5.5 Codex cross-check): adds an `Operational Safety Gate` (live `NO_TOUCH_POLICY.md` first, `realpath` source/dest, no-touch plus `DEC-039` duplicate-canonical-folder check, filesystem-boundary `rsync --dry-run`, exact command plus rollback/verify plan, exact-scope approval), three end-of-phase decision tokens (`MOVE_SAFE_BATCH_PROPOSED`, `MOVE_OWNER_DECISION_REQUIRED`, `MOVE_BLOCKED_NO_TOUCH`), a no-fabricate rule when the VPS is unreadable, and marks the in-file protected-roots list as a non-authoritative snapshot. The mapped prompt files remain the source of truth.

As of 2026-06-09, `Use AI Relay` is added as a proven, token-saving variant of `Use AI Pair`: Claude only plans, picks the coder, reviews, and decides, while a cheaper non-Claude coder writes the code (Grok as default via headless terminal, Codex for hard backend via cross-check `ask_gpt5`, Gemini for large UI). It loops fix->review until 100% before the phase gate opens, and carries a per-project setup checklist (owner must add `Bash(grok:*)`/`Bash(gemini:*)` via `/permissions` since AI cannot self-grant) plus connection-reality notes. Proof case: master-webengine 2026-06-09 — Grok headless file write and Claude review verified end-to-end.

## Graph Links

- [[skills/prompt-shortcuts/Prompt Shortcuts|Prompt Shortcuts]]
- [[skills/prompt-shortcuts/SKILL|Prompt Shortcuts Skill]]
- [[ai-context/session-start-contract|Session Start Contract]]
- [[ai-context/adapter-contract|Adapter Contract]]
