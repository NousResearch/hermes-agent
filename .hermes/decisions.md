# Implementation Decisions

Updated: 2026-06-01

## Vault Reorganization (2026-06-01)

Decision: reorganize the Obsidian vault from 28 mixed-name folders into 13 human-readable top folders, keep external-facing anchors in place.

Reason:

- folder names were AI/system jargon; owner could not tell what each stored
- `ai-context/`, `memory/`, `AI_MEMORY.md`, `skills/`, `projects/` are referenced by ~250 instruction files across 30+ projects, so they stay put (moving them breaks every project)
- moved 791 files, rewrote 2,352 internal wikilinks, patched 33 external instruction files; verified 0 stale links, 0 file loss
- 3 timestamped backups kept under `ObsidianVault/_backups/`

Decision: add a PDCA knowledge gate. AI must not cite `95-Inbox-Lab/` (inbox + review) as verified knowledge until promoted.

Reason: prevents unproven/raw material from being answered as fact; wired into `ai-context/session-start-contract.md` and `ai-context/knowledge-stage-gate.md`.

Decision: extend `obsidian_safe_bridge` — write target moved `review-queue/` → `95-Inbox-Lab/review/`; deny reads of `90-Owner-Private/` and any `owner-private/` zone unless `HERMES_OWNER=1`.

Reason: the reorg moved the old review-queue (broke the bridge write path), and owner asked for owner-only private zones. 9 plugin tests pass. Per-person grants await owner spec in `99-System/access-policy/`.

## Project Index + Agent Registry + CLAUDE.md Governance (2026-06-01)

Decision: build a Hermes "control tower" in Obsidian — deep-read all projects into capsule cards, mirror all agents into a registry, and standardize bloated CLAUDE.md files.

What was done:

- Deep-read 35 projects from real code (multi-agent workflow, 1.79M tokens) → standard capsule cards in `projects/_index/` + index README with health scores.
- Generated agent registry: 29 agents (9 system + 20 business) from `~/.claude/agents/` → `40-Agents/` cards + index (rerun: `agent_registry_gen.py`).
- CLAUDE.md governance: health-checker (`claudemd_health_check.py`, scans 53 files), standard doc (`00-Center/standards/claudemd-standard.md`), and thinned the 5 most bloated CLAUDE.md (MQ5 631→153, WebEngine 392→94, SynerryEoffice 323→122, ViberQC 244→75, JigsawWebChat 193→81) — all backed up, project-specific rules preserved, central rules replaced with a pointer.

Decision: keep central rules in ONE place (`~/.claude/` + vault `ai-context/`); project CLAUDE.md must be thin and point to it. Rationale: central rules were copy-pasted across CLAUDE/AGENTS/GEMINI/QWEN × 40 projects (~160 spots), causing drift.

---

Updated: 2026-05-26

## Trend Discovery Center

Decision: implement as a bundled standalone plugin, not by rewriting Hermes core.

Reason:

- keeps this business-trend system isolated and maintainable
- uses existing Hermes plugin CLI surface
- avoids hardcoding a one-off project into `run_agent.py`, `cli.py`, or gateway core

Decision: use SQLite under `~/.hermes/trend-discovery/`.

Reason:

- local-first and profile-scoped
- no VPS/Postgres dependency required for the initial reliable system
- easy for future AI agents to inspect and back up

Decision: use macOS `launchd` on this machine.

Reason:

- user asked for the system to actually run
- this host is macOS
- LaunchAgents continue running on schedule without keeping the chat open

Decision: configure macOS notifications with local fallback.

Reason:

- provides real user-visible notification on this Mac
- local receipt logs still capture delivery attempts if notification fails

Decision: keep Open Crawl and n8n optional.

Reason:

- the original failure mode was over-reliance on Open Crawl and n8n
- source adapters should fail independently without taking down the pipeline
- empty optional adapter URLs must be marked skipped, not success

Decision: expose explicit operator modes and source administration.

Reason:

- future AI agents and the user need to see how the system is controlled
- launchd schedule alone is not enough; operators need mode/status/source/log commands
- `trend-discovery ops` is the top-level operational summary

---

## 2026-07-04 · AI Relay บน VPS · แก้ทางโค้ด ไม่แตะ token gateway (nat)

Decision: แก้ Fable/Opus เรียกไม่ได้บน VPS ด้วยการให้ relay-call ตัด `CLAUDE_CODE_OAUTH_TOKEN` (token org ที่ปิดสิทธิ์ Claude Code) เฉพาะตอนเรียก claude (fable/opus) → ใช้ login เครื่องแทน · ไม่แก้ `~/.hermes/.env` ของ gateway

Reason:
- ทางโค้ดไม่แตะไฟล์ลับที่ Hermes gateway (พนักงานใช้ร่วม) ใช้อยู่ → gateway ไม่เสี่ยงล่ม
- ต้นเหตุจริงที่ Codex เรียก relay ไม่ได้ = สำเนาเก่า `/usr/local/bin/relay-call` (root) บังตัวใหม่ `~/.local/bin` (symlink→repo) · แก้ด้วย sudo symlink ชี้ repo (เจ้าของรันเอง)
- verified tier 5: เจ้าของรัน `relay-call --tool fable` บน VPS ได้ opus (rotated_from=fable)
- commit หลัก `06aff79d8` · main=VPS=`816cdf2e0` · pytest 11/11
