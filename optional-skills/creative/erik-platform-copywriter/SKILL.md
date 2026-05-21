---
name: erik-platform-copywriter
description: Adapt Erik ideas into platform-native copy.
version: 0.1.0
author: Erik Johnson, with Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [writing, copywriting, social, brand-voice, creator, platform]
    category: creative
    related_skills: [humanizer, youtube-content]
---

# Erik Platform Copywriter Skill

Turn a core Erik or ASON idea into publishable, platform-native copy. This skill does not simply repost one caption everywhere; it translates the idea into the behavior, format, and audience expectation of each surface while preserving Erik's operator voice.

Use it as the bridge between strategy notes, transcript-derived voice material, Notion draft rows, and actual publishing copy.

## When to Use

Use this skill when the user asks to:

- Write, rewrite, or adapt Erik-facing copy for LinkedIn, Substack, Instagram, YouTube, X, Threads, Facebook, TikTok, or a website archive.
- Turn a Notion draft, campaign outline, cultural read, podcast extraction, transcript insight, report thesis, or ASON proof object into platform-ready posts.
- Produce multiple channel versions from one idea without flattening the idea into generic reposts.
- Make writing sound more like Erik: practical, specific, taste-driven, culturally fluent, dry when appropriate, and useful to operators.
- Review whether a draft feels too generic, too corporate, too ASON-sales-heavy, or too far from Erik's actual voice.

Do not use this skill for generic brand copy unless the user explicitly wants Erik's personal/operator voice as the controlling style.

## Prerequisites

Preferred source materials:

- Erik likeness model: `/Users/imnfst/Documents/Obsidian Vault/Creator Intelligence/Erik Brand OS/Erik Brand OS Likeness Model.md`
- Erik platform playbook: `/Users/imnfst/Documents/Obsidian Vault/Creator Intelligence/Erik Brand OS/Erik Brand OS Platform Playbook.md`
- Erik source protocol: `/Users/imnfst/Documents/Obsidian Vault/Creator Intelligence/Erik Brand OS/Erik Source Intake and Scoring Protocol.md`
- UNCULTURABLE extraction notes: `/Users/imnfst/Documents/Obsidian Vault/Creator Intelligence/Content Library/YouTube/`
- Raw transcript notes: `/Users/imnfst/Documents/Obsidian Vault/Creator Intelligence/Raw Transcripts/YouTube/`
- Current Notion publishing surface: `Publishing Drafts` database and the relevant series hub or draft row.

Use `read_file` for local Markdown sources, `search_files` when the exact source file is unknown, and Notion MCP tools such as `notion-search`, `notion-fetch`, and `notion-update-page` when the working draft lives in Notion.

If source attribution is uncertain because the podcast transcript is automated or speaker labels are unreliable, mark claims as `provisional` and avoid turning them into hard biography.

## How to Run

Start with one of these inputs:

- A core idea: "AI interpretation is replacing attention as the real brand battleground."
- A Notion draft row or page.
- A transcript/extraction note.
- A report or campaign thesis.
- A rough caption.
- A platform request: "Make this a LinkedIn post, Substack essay, Instagram carousel, and YouTube Short."

If the user does not specify platforms, default to:

1. LinkedIn operator post.
2. Substack essay outline plus opening section.
3. Instagram carousel or Reel concept.
4. YouTube Short script.
5. X/Threads sharp take.

For ASON/company voice, make the copy more polished and less personal than Erik's direct voice, but keep it plainspoken. For Erik personal voice, prioritize a specific receipt, operator principle, and practical move.

## Quick Reference

| Platform | Native Job | Default Output |
| --- | --- | --- |
| LinkedIn | Business authority and relationship surface | 120-300 word operator post |
| Substack | Owned POV and deeper trust | Signal essay or newsletter section |
| Instagram | Taste, human proof, and social texture | Story sequence, carousel, or Reel caption |
| YouTube Short | Searchable spoken POV | 30-90 second script |
| X / Threads | Fast signal testing | Sharp take or short thread |
| Facebook | Warm legacy trust graph | Reflective personal version |
| TikTok | Discovery and hook testing | Direct-to-camera hook script |
| Website | Canonical archive | Durable thesis summary |

## Procedure

### 1. Identify the Source Unit

Reduce the input to one core idea. Do not start drafting until the idea can be stated in one sentence.

Capture:

- The claim.
- The evidence or receipt.
- The audience.
- The platform goal.
- The desired action or response.
- What must not be claimed.

If the idea is only an outline, say that clearly before writing copy.

### 2. Select the Voice Mode

Choose one primary mode:

- `Erik personal`: first-person operator voice; more specific, warmer, drier, and more conversational.
- `ASON company`: sharper institutional POV; less biography; more proof-methodology and market framing.
- `Hybrid`: Erik byline with ASON gravity; personal read connected to the business thesis.

For Erik personal voice, use the likeness pattern:

1. Specific receipt or signal.
2. Pattern recognition.
3. Operator principle.
4. Practical move.
5. Dry closing line or serious question.

### 3. Apply Erik Voice Rules

Do:

- Start from a concrete receipt when possible.
- Make the insight useful to founders, operators, creators, brand-side decision-makers, or buyers.
- Let taste and timing show through details.
- Use plain language and short paragraphs.
- Admit contradiction before making the point.
- Keep ASON as strategic gravity, not constant sales copy.

Avoid:

- Generic AI futurist language.
- Generic agency founder language.
- Motivational speaker phrasing.
- Vague "culture is changing" claims when a specific signal exists.
- Attributing Pablo's wife/kids/Texas/Mexico/fatherhood material to Erik.
- Making every post about ASON.

### 4. Translate by Platform

Do not paste the same copy into every channel. Translate the idea into native behavior.

#### LinkedIn

Output:

- Hook.
- Full post.
- Optional alternate hook.
- Comment prompt.
- Suggested first comment if useful.

Structure:

```markdown
[Specific observation or receipt]

[What most people misunderstand]

[The operator read]

[The move / principle]

[Question or closing line]
```

Keep it useful, not performative. Prefer relationship-building comments over engagement bait.

#### Substack

Output:

- Title options.
- Dek or preview text.
- Essay outline.
- Draft opening section.
- Closing question.

Structure:

```markdown
# [Signal-driven title]

## The signal

## The behavior shift

## The operator read

## The move

## What I'm watching next
```

Use Substack when the idea needs depth, receipts, and an owned archive.

#### Instagram

Choose the most natural format:

- Story sequence for human context or fast signal.
- Carousel for explainers or proof structures.
- Reel for direct POV, taste, or personality.
- Feed caption only when there is a strong visual.

Output:

- Format recommendation.
- Slide/frame breakdown.
- Caption.
- Story sticker or reply prompt.
- Visual direction.

Default Story sequence:

```markdown
Frame 1: What I'm seeing
Frame 2: Why it matters
Frame 3: The human/taste/context layer
Frame 4: Question/poll/reply prompt
```

Avoid LinkedIn screenshots as the default.

#### YouTube Short

Output:

- Hook.
- 30-90 second spoken script.
- On-screen text beats.
- Caption.
- Visual direction.

Structure:

```markdown
Hook: [Specific belief or misconception]
Point: [What is actually happening]
Example: [Concrete reference]
Move: [What operators/brands/creators should do]
Close: [Memorable line or question]
```

Write for spoken cadence. The script should sound natural when read aloud.

#### X / Threads

Output:

- One sharp take.
- Optional 3-7 post thread.
- Reply bait only if it invites useful discussion.

Default structure:

```markdown
Everyone is treating [topic] like [surface read].

The real signal is [behavior/timing/trust/taste shift].
```

Use this as the testing layer, not the final archive.

### 5. Add Copy QA

Before handing off, run a visible QA pass:

- Does this have a specific receipt or signal?
- Does the copy move from observation to useful principle?
- Is the operator move clear?
- Could a generic AI agency founder have written this?
- Is Erik's biography safe and correctly attributed?
- Is the platform format native?
- Does the CTA fit the platform?

If any answer fails, revise once before presenting.

### 6. Handoff Format

Return outputs grouped by platform. For each platform include:

- `Status`: ready, needs source, needs Erik approval, or internal only.
- `Copy`: publishable text or script.
- `Why this format`: one sentence.
- `Watch for`: performance signal or risk.

When updating Notion, preserve original draft rows and add platform-specific variants rather than overwriting the source idea unless the user explicitly asks for replacement.

## Pitfalls

- Treating the hub outline as publishable copy. It is routing; the real copy lives in draft rows or generated platform variants.
- Making ASON posts sound like Erik's personal diary.
- Making Erik posts sound like ASON brochure copy.
- Using transcript-derived claims as biography without speaker verification.
- Turning every insight into "AI will change everything."
- Reusing one LinkedIn caption across Instagram, Substack, and YouTube.
- Writing without a visible receipt, example, or operator move.
- Over-polishing away the plainspoken cadence.

## Verification

Verify the output before publishing:

- Read it aloud for Erik personal voice.
- Confirm the platform has the correct native format.
- Confirm any Notion row or page link still points to the active source.
- Confirm sensitive or uncertain source claims are marked or removed.
- Confirm the copy has one clear idea, not a bundle of adjacent ideas.
- Confirm the closing line invites the desired behavior: reply, save, DM, subscribe, watch, or think.

For Notion workflows, fetch the updated page or row after writing and confirm the new platform variant appears where expected.
