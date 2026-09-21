Communication: Direct, casual, sharp. No AI-speak, em-dashes, or sycophancy. Profanity OK. Pushback wanted. Flag costs before recommending. When explaining technical issues, use simple clear explanations first then offer to go deeper. Prefers lane-by-lane systematic review of task lists rather than dumping everything at once. Trusts agent recommendations once explained clearly.
§
Expects proactive skill updates after productive sessions. 'Nothing to save' only when true.
§
Architecture preference: plugin/overlay on Hermes tool APIs, never fork upstream. Merge risk is primary. Prefers efficient role-based profile architecture: persistent profiles only for ongoing lanes, lightweight worker prompt templates for ephemeral specialists.
§
CoachOS renamed to CoachSense. Access self-hosted services via SSH tunnel (localhost:4007). One X dev app covers multiple accounts — don't create separate OAuth apps per account. Will do dev portal setups (X, LinkedIn) when walked through step-by-step. Asks about cost implications before committing to anything.
§
User's "just execute" signal: capital-letter approvals, "finish the remaining work," or clear root cause + low-risk fix. Act without asking, report summary after. Only confirm expensive, destructive, or external-facing actions. For bigger decisions (upstream merges, infra changes, multi-step ops with risk), requires a pre-output summary first — what's in it, relevance to his setup, pre-risks flagged, then a go/no-go gate before proceeding.
§
Cost-pragmatic, not cost-minimising. OK with higher-token models for constructive agent work. Default to most useful, not cheapest, for internal crons. But will defer API-dependent features until ROI is proven — asked about multi-account cost implications before committing to tweet ripeline. Free-first until validated, then pays for what works.
§
Content approval: Telegram reply (A:all, A:1-3, R:4,5) to Topic 22. Content must be deep story-driven narrative posts — taglines and 1-liners unacceptable. Engine imports llm_drafts.py not drafts.py.
§
User cares about Telegram format quality and kanban output readability, calling current output "not reader friendly." Wants clean, scannable, one-glance messages. Next time: propose delta-only digests, collapse >5 items, max 2 sections / 5 bullets visible, verbose detail in <blockquote expandable>.
§
User expects the heartbeat audit cron to be fully functional with KANBAN integration — executing every hour and linking with the audit-engine skill, kanban-orchestrator skill, and kanban task filing. The kensei-heartbeat-audit should not be script-only; it needs to be LLM-driven with rotation targets and kanban task creation. Previous sessions rewired the backlog processor and kanban instigator to be rewritten and wired up correctly into the KENSEI autonomous loop.
§
Sahil prefers technical problem-solving that follows: 1) Root cause analysis (not symptoms), 2) Implement permanent fix (survives container recreates), 3) Verify via direct checks (API/logs). He values verification steps and dislikes temporary workarounds that require reapplication.
§
Algorithm-informed content approach: proactively analyses platform algorithms (x-algorithm repo, engagement weights, banger classifiers, slop detection) and adjusts content strategy accordingly. Wants content engineered for distribution mechanics, not just quality writing — evaluates templates through the lens of what the algorithm rewards. Prefers longer story-driven content over punchy tweets. Likes styles: educational lists (Ole Lehmann), data-driven myth-busting (SwedishRumble), narrative how-to essays (shannholmberg), nuanced football analysis (pythaginboots). For MMaestro brand voice, wants semi-professional + down to earth + friendly banterish but unsure of exact direction — needs research first.