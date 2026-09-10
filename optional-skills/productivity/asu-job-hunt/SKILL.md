---
name: asu-job-hunt
description: Chinese job-hunt pack — resume, interview, offers.
version: 1.0.0
author: Hisn00w (ASu-skills), ported for Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [job-hunt, resume, interview, chinese, career]
    category: productivity
    related_skills: [humanizer, docx, pdf]
---

# ASu Job-Hunt Workflow (中文求职工作流)

Use when a user working in Chinese asks to polish job-hunt materials: reframe
their real experience for a target role (经历"酥化"), build or replicate an
HTML/PDF resume, prepare open-source contribution evidence, rehearse
interviews with follow-up chains, or track applications through 秋招 season.

Ported from [Hisn00w/ASu-skills](https://github.com/Hisn00w/ASu-skills)
(MIT, 2.1k★). The six upstream skills are preserved verbatim in Chinese under
`references/<stage>/` — load the one matching the user's stage and follow it.
This hub only adds routing and Hermes-specific adaptations.

## Stage routing

| User intent | Load |
|---|---|
| Reframe experience for a target role, HR opener, self-intro (酥化/包装) | `references/asu/asu.md` |
| Build an editable HTML resume, pick from 18 templates, export PDF | `references/resume/resume.md` |
| Replicate the "ASu-style" tech resume look from a screenshot | `references/asu-resume/asu-resume.md` |
| Build verifiable open-source contribution evidence cards | `references/contributor/contributor.md` |
| Interview prediction + chained follow-up drills | `references/interview/interview.md` |
| 秋招 application/assessment/offer tracking, recruiting-mailbox triage | `references/offer/offer.md` |

Per-stage sub-references live next to each stage file (claim–evidence ledger,
business-analysis evidence, page-balance QA, daily contributor routine,
email monitoring). Follow the stage file's own links — paths are relative and
resolve inside this skill directory.

## Hermes adaptations (read before following the zh bodies)

1. **Asset paths.** Upstream texts reference `../../assets/` (repo-plugin
   layout) or `../../assets/asu/` (Claude Code layout). In Hermes, ALL assets
   live in THIS skill's `assets/` directory:
   `assets/templates-html/` (18 zh resume templates),
   `assets/resume-template-editable.html`, `assets/resume-template-two-page.html`,
   `assets/asu-resume-template.html`, `assets/resume-data-template.json`,
   `assets/application-tracker.html` + `application-tracker-overview.svg`,
   `assets/career-claim-ledger-template.json`, `assets/icons/`, `assets/logos/`.
   The upstream "candidate directory must contain all 18 templates" detection
   dance is unnecessary here — use this skill's `assets/` directly.
2. **PDF export.** `scripts/export-resume-pdf.mjs` is zero-dependency (drives
   installed Chrome/Edge/Chromium headless). Run
   `node scripts/export-resume-pdf.mjs <html> --out <pdf>` from the skill dir
   (the upstream `npm run export:pdf` form does NOT work here — no
   package.json ships with this port).
   Verify the output with `pdftotext` or read_file before delivering. If no
   Chromium is installed, fall back to
   `chromium --headless --print-to-pdf=<pdf> <html>` or ask the user to print
   to PDF from a browser.
3. **CJK read trap.** `read_file` can misdetect dense-CJK markdown as binary.
   If that happens, read the file via `execute_code` with Python `open()`.
4. **Slash commands.** Upstream triggers like `/asu`, `/interview`, `/offer`
   are Claude Code plugin commands — in Hermes just route by intent using the
   table above.
5. **Truthfulness boundary is load-bearing.** The whole pack's premise is
   evidence-backed reframing, NOT fabrication: never invent titles, employers,
   projects, tech stacks, or numbers. Keep the claim–evidence ledger
   (`assets/career-claim-ledger-template.json`) when the user makes strong
   claims across stages.
6. **Demo photo.** `assets/fictional-resume-photo.png` is a fictional persona
   image for template demos only — never present it as a real person. When
   filling a COPY of a template outside `assets/`, also copy the photo next to
   it (or inline it as a data URL) — templates reference it by relative path.

## Verification

- Resume flow: filled template renders in a browser, exports to PDF at the
  right page size, and every strong claim maps to a ledger entry.
- Interview flow: each predicted question carries a follow-up chain and an
  evidence anchor from the user's real materials.
- Tracker: `application-tracker.html` opens standalone and edits persist via
  its built-in localStorage save.

## Credit

Original work by [Hisn00w and contributors](https://github.com/Hisn00w/ASu-skills)
(MIT — see `LICENSE.upstream`), distilling public sharing by 阿酥在coding and
Hi Mr Lonely (credited upstream). Brand icons from LobeHub lobe-icons.
