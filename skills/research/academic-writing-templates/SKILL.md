---
name: academic-writing-templates
description: "LaTeX APA7 templates, AI writing tools, and article writing guides — curated from GitHub Topics: academic-writing."
version: 1.0.0
author: Fuad Al Fajri
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [LaTeX, APA7, Templates, Academic-Writing, Paper, Overleaf]
    category: research
    related_skills: [phd-resources, research-paper-writing]
---

# Academic Writing Templates & Tools

Collection of LaTeX templates, AI writing tools, and paper structure guides to accelerate journal article writing.

## When to Use

- User asks for LaTeX templates for APA 7, journal articles, or theses
- User asks for AI-assisted academic writing tools or prompts
- User asks how to structure an IMRaD paper or which writing workflow to follow
- User needs proofreading/grammar tools for academic prose

Don't use for: writing an actual paper end-to-end (see `research-paper-writing` for the full pipeline). This skill curates tools and templates.

## 🧠 AI Writing Tools (Priority)

| Repo | ⭐ | Function |
|------|----|--------|
| **academic-research-skills** | 37k+ | Full Claude Code pipeline: research → write → review → finalize |
| **opendraft** | 317 | 19-agent paper writer, verified citations via CrossRef/arXiv |
| **Feynman** (companion-inc) | 8.3k | Standalone AI research agent + CLI |
| **PaperOrchestra** (Google Research) | 608 | Multi-agent → submission-ready LaTeX |
| **chatgpt-prompts-academic** | 4.8k+ | Ready-made prompts per article section |

### Using with Hermes:

Simply say:

> "Help me write [paper section] using IMRaD template, formal academic style."

Or:

> "Review my paper draft for structure, argumentation, and language."

## 📐 LaTeX Templates (APA 7)

For journal articles and theses following APA 7:

| Template | Features |
|----------|-------|
| **apa7-latex-template** | Zero-config Tectonic, strict APA 7 compliance |
| **LaTeX-APA_Template** | Guided structure + sample PDF |
| **apa-latex** | Student paper + thesis templates |
| **LaTeX-templates** (diverse) | Collection for various needs |

### How to use:
1. Open Overleaf (overleaf.com)
2. Pick a template or upload a .zip template
3. Start writing

## 🏗️ General Paper Templates

| Template | Features |
|----------|-------|
| **latex-paper** | Minimalist design, arXiv-ready |
| **latex-templates** (martinhelso) | Collection for conferences, theses, etc. |
| **awesome-scientific-writing** | Curated toolchain: Markdown → Pandoc → LaTeX |

## ✍️ Proofreading & Grammar Tools

| Tool | Function | Cost |
|------|--------|------|
| **proselint** | Linter for academic prose | **Open source** |
| **write-good** | Weak sentence detection | **Open source** |
| **Rousseau** | Auto-fix common mistakes | **Open source** |
| **Grammarly** | Full grammar & style | Freemium |

## 🔄 Writing Workflow

### From Zero to Submission:

1. **Structure** — Decide IMRaD/other outline
2. **Draft** — Write per section, don't be a perfectionist yet
3. **Self-review** — Re-read, check logical flow
4. **AI assist** — Ask Hermes to review structure & language
5. **Proofreading** — Use proselint / Grammarly
6. **Template** — Format per target journal template
7. **Submit** — Prepare cover letter, metadata, files

### Prompt template for Hermes:

```
I want to write a journal article about [topic].
Target journal: [journal name, IMRaD template].
Help me:
1. Create a complete outline
2. Write a draft [section]
3. Review consistency and flow
```

## Common Pitfalls

- Don't copy AI writing tools blindly — verify citations against CrossRef/arXiv (tools like opendraft do this; manual prompts don't)
- Don't ignore journal-specific formatting — a general APA template may not match the target journal's house style
- Don't skip proofreading tools just because an LLM drafted the text — proselint/write-good catch different issues

## Verification Checklist

- [ ] Recommended template matches the target output (journal article vs thesis vs conference)
- [ ] AI tool recommendation includes citation-verification caveat
- [ ] Writing workflow steps are complete from structure to submission

## References

- [GitHub Topics: academic-writing](https://github.com/topics/academic-writing) — 766 repositories
- [academic-research-skills](https://github.com/Imbad0202/academic-research-skills) — 37k⭐
- [PaperOrchestra](https://github.com/google-research/PaperOrchestra) — multi-agent paper writer
