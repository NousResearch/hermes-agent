---
name: skill-resolver
description: "Use when this is your first task or the task is ambiguous — load this resolver skill first, read the routing tree, then dispatch to the matching leaf skill. Always loaded in every session by default."
version: 1.0.0
author: Hermes Agent
license: MIT
metadata:
  hermes:
    tags: [resolver, routing, meta-skill, thin-harness, dispatcher]
    related_skills: [hermes-agent-skill-authoring, check-resolvable]
---

# Skill Resolver — Thin Harness Routing Skill

THIS IS THE FIRST SKILL THAT MUST LOAD IN EVERY SESSION. It is the thin routing table — the "harness" in Garry Tan's "Thin Harness, Fat Skills" pattern.

## Behavior Contract

1. When you receive a task, load this skill (if not already loaded).
2. Read the matching branch in the decision tree below.
3. Load the leaf skill(s) it points to.
4. Execute using the leaf skill's instructions.
5. If no match exists in the tree, load `skill-resolver` again and walk the tree more carefully before falling back to `skills_list`.

## Decision Tree (Compact)

| Task signal | Load skill(s) |
|---|---|
| "setup hermes", "config", "gateway", "profile", "model", "tools", "cron", "webhook", "mcp", "auth", "doctor" | `hermes-agent` |
| "author skill", "create skill", "edit skill", "skill frontmatter", "SKILL.md" | `hermes-agent-skill-authoring` |
| "delegate", "subagent", "spawn worker" | `subagent-driven-development` |
| "clone repo", "create repo", "new repo", "fork", "release" | `github-repo-management` |
| "new PR", "open PR", "push branch", "CI failed" | `github-pr-workflow` |
| "review PR", "code review", "review changes" | `github-code-review` |
| "GitHub issue", "create issue", "triage issue" | `github-issues` |
| "auth github", "github token", "ssh key github" | `github-auth` |
| "codebase stats", "lines of code", "LOC" | `codebase-inspection` |
| "debug python", "pdb", "debugpy" | `python-debugpy` |
| "debug node", "inspect node", "chrome devtools" | `node-inspect-debugger` |
| "rust tui", "ratatui", "terminal UI" | `rust-tui-development` |
| "research paper", "arxiv", "academic" | `arxiv` |
| "blog monitor", "rss", "feed" | `blogwatcher` |
| "polymarket", "prediction market", "betting" | `polymarket` |
| "youtube transcript", "youtube summary" | `youtube-content` |
| "wiki", "knowledge base", "llm wiki" | `llm-wiki` |
| "benchmark llm", "evaluate model", "mmlu", "gsm8k" | `evaluating-llms-harness` |
| "serve llm", "vllm", "deploy model" | `serving-llms-vllm` |
| "llama.cpp", "GGUF", "local inference" | `llama-cpp` |
| "huggingface", "hf download", "model search" | `huggingface-hub` |
| "abliterate", "refusal", "uncensor" | `obliteratus` |
| "dspy", "prompt optimizer", "rag program" | `dspy` |
| "wandb", "weights and biases", "experiment track" | `weights-and-biases` |
| "musicgen", "audiocraft", "audio gen" | `audiocraft-audio-generation` |
| "segment anything", "sam", "image segmentation" | `segment-anything-model` |
| "diagram", "architecture diagram", "infra diagram" | `architecture-diagram` |
| "ascii art", "figlet", "cowsay" | `ascii-art` |
| "ascii video", "terminal video" | `ascii-video` |
| "comfyui", "workflow", "image gen pipeline" | `comfyui` |
| "excalidraw", "hand-drawn diagram", "whiteboard" | `excalidraw` |
| "prototype", "mockup", "html mockup" | `sketch` |
| "landing page", "design artifact", "HTML design" | `claude-design` |
| "manim", "animation", "3b1b" | `manim-video` |
| "p5js", "creative coding", "processing" | `p5js` |
| "pixel art", "sprite", "retro graphics" | `pixel-art` |
| "songwriting", "music prompt", "suno" | `songwriting-and-ai-music` |
| "touchdesigner", "visuals", "realtime" | `touchdesigner-mcp` |
| "social post", "brand post", "content production" | `brand-social-post-production` |
| "article illustration", "illustration" | `baoyu-article-illustrator` |
| "comic", "knowledge comic", "educational comic" | `baoyu-comic` |
| "infographic", "info graphic", "visualization" | `baoyu-infographic` |
| "ideation", "brainstorm", "idea generation" | `ideation` |
| "humanize", "remove ai", "natural text" | `humanizer` |
| "design token", "DESIGN.md" | `design-md` |
| "jupyter", "notebook", "data analysis" | `jupyter-live-kernel` |
| "email", "himalaya", "send mail", "read mail" | `himalaya` |
| "twitter", "x.com", "post tweet", "xurl" | `xurl` |
| "minecraft server", "modpack" | `minecraft-modpack-server` |
| "pokemon", "emulator", "gameboy" | `pokemon-player` |
| "google workspace", "gmail", "google docs", "google sheets", "google calendar", "google drive" | `google-workspace` |
| "notion", "notion api" | `notion` |
| "obsidian", "obsidian vault", "note" | `obsidian` |
| "airtable", "base", "records" | `airtable` |
| "powerpoint", "pptx", "slide deck" | `powerpoint` |
| "edit pdf", "pdf text", "nano-pdf" | `nano-pdf` |
| "ocr", "extract text", "scan pdf" | `ocr-and-documents` |
| "linear", "linear issue", "linear ticket" | `linear` |
| "maps", "geocode", "directions", "route" | `maps` |
| "teams meeting", "meeting summary" | `teams-meeting-pipeline` |
| "petdex", "mascot", "pet" | `petdex` |
| "apple notes", "memo" | `apple-notes` |
| "apple reminders", "reminders" | `apple-reminders` |
| "findmy", "airtag", "find device" | `findmy` |
| "imessage", "sms", "text message" | `imessage` |
| "computer use", "desktop automation", "click", "macos" | `computer-use` + `macos-computer-use` |
| "cua driver", "computer use setup" | `macos-cua-driver-setup` |
| "philips hue", "smart light", "openhue" | `openhue` |
| "shannon", "pentest", "security audit", "vuln scan" | `shannon` |
| "jailbreak", "godmode", "red team" | `godmode` |
| "dogfood", "qa test", "bug hunt", "exploratory" | `dogfood` |
| "spotify", "playlist", "music" | `spotify` |
| "gif", "tenor", "animation" | `gif-search` |
| "write plan", "implementation plan" | `writing-plans` |
| "plan mode", ".hermes/plans" | `plan` |
| "spike", "experiment", "prototype research" | `spike` |
| "check resolvable", "reachability audit", "dark skills" | `check-resolvable` |
| "gateway ops", "hermes gateway", "container health" | `hermes-gateway-ops` |
| "fleet memory", "memory audit" | `fleet-memory-audit` |
| "mcp server", "mcp setup" | `native-mcp` |
| "webhook", "webhook subscribe" | `webhook-subscriptions` |
| "iris design", "iris agent", "designer agent" | `iris` |
| "provision vision", "vision debug", "multimodal debug" | `provider-vision-debugging` |
| "pop culture design", "web design reference", "stripe design", "linear design", "vercel design" | `popular-web-designs` |
| "pretext", "text layout", "text geometry" | `pretext` |
| "hermes pet", "pet mode", "mascot" | `hermes-pets` |
| "can't find match" | Load `check-resolvable` to audit, then re-scan via `skills_list` |

## Category Resolvers

For deep dives into a category, load the category resolver file:

- **Apple ecosystem:** `skills/apple/RESOLVER.md`
- **Creative work:** `skills/creative/RESOLVER.md`
- **ML/AI Ops:** `skills/mlops/RESOLVER.md`
- **DevOps:** `skills/devops/RESOLVER.md`
- **GitHub:** `skills/github/RESOLVER.md`
- **Productivity:** `skills/productivity/RESOLVER.md`
- **Autonomous AI Agents:** `skills/autonomous-ai-agents/RESOLVER.md`
- **Software development:** `skills/software-development/RESOLVER.md`

## Invocation Priority

1. If the task is precise (single domain), load only 1-2 leaf skills from the table above. No need to load the resolver if you already know which skill to use — the description match IS the resolver.
2. If the task is broad or ambiguous, load this `skill-resolver` first and walk the tree.
3. If you've loaded this resolver, read the matching row, load the target skill, and execute from there.
4. If the target skill also has a resolver, recurse.

## Filing Rules

| Artifact | Destination |
|---|---|
| New skill | `skill_manage(action='create')` |
| User preference learned | `memory(target='user', action='add', content=...)` |
| Working procedure | `memory(target='memory', action='add', content=...)` |
| Project rule | `.hermes.md` or `AGENTS.md` in project root |
| Task decomposed | Kanban task via `kanban-create` tool |
| Code reference | Script or reference file in the relevant skill directory |
| Resolver gap found | Create a Kanban task assigned to this profile to add the missing resolver entry |

## Completion Criteria

- [ ] You loaded the correct leaf skill for the task
- [ ] You followed the leaf skill's instructions, not the resolver's
- [ ] If the resolver had no match, you either found the right skill or created a Kanban task to fill the gap
- [ ] Any newly created skill was added to the resolver chain
