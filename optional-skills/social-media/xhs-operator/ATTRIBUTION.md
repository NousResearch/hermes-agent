# Attribution

This skill is adapted from a third-party MIT-licensed skill. All reuse is
credited here and the original copyright notice is reproduced in full below.

## xiaohongshu-account-launch-expert (agent-skills-launch-pack)

- Source: https://github.com/chenjin-cmd/agent-skills-launch-pack_
- Skill: `skills/xiaohongshu-account-launch-expert/`
- License: MIT
- Copyright: © 2026 Chen

### What was adapted (SOP skeleton — reused)

The compliance-first operating skeleton is reused, restructured for Hermes
Agent's SKILL.md conventions:

- 起号模式判定 (launch-mode routing: IP-trust / traffic / repair / scale)
- 定位句 (positioning sentence) + 主页地基 (profile foundation)
- 简报先行 (brief-before-draft) two-stage note workflow (文案 + 封面)
- 内容日历 (content calendar) + 周复盘 (weekly review) loop
- 风险边界 (risk boundaries) — no fake engagement, no rewrite-to-evade,
  no copied images/cover, no 广告法 极限词, no promised outcomes.
  Reproduced with intent.

### What was rewritten (real-data workflow — the ApexNodes delta)

Two sections were rewritten from prompt-only to **real-data** workflows that
call the platform social-data tools (`social_search` / `social_posts` /
`creator_top_posts` / `social_content` / `social_comments` / `social_profile`
with `platform="xiaohongshu"`) instead of asking the model to invent competitor
benchmarks, viral covers, or trending tags:

- **对标拆解 (competitor teardown)** — now pulls real 小红书 notes via the
  tools (including `creator_top_posts`, which xiaohongshu supports) and
  abstracts title hooks / cover patterns / structure / tags from live data.
- **选题库 (topic bank)** — now a field table seeded from real search results,
  with tag suggestions traced to real notes/comments and an honest note on
  which signals are external-unavailable.

Cover-image generation is wired to the runtime's own image-generation
capability with routing chosen by the runtime tier (带字封面 default; a
free cover line — platform base image + typographic overlay, hc-433 — also
selectable within the tier). The skill does not hard-code a provider. Skipped
honestly when no image-generation capability is available.

### Honest scope notes (ApexNodes environment)

- 小红书 has **no public trending-board data source** at this time; the topic
  bank and weekly review approximate hot topics via time-scoped
  `social_search` and leave an inline note to wire the trending board when it
  ships (hc-426).
- Creator-backend metrics (exposure, CTR, follower growth, collect rate, paid
  promotion) are external-unavailable; the skill asks the user to screenshot
  them rather than inventing numbers.

### License compatibility

The source ships under the MIT License, which permits redistribution and
modification with attribution. This skill preserves the original copyright
notice (below) and credits the source in the SKILL.md metadata. No code was
relicensed.

---

## Original MIT License (agent-skills-launch-pack)

```
MIT License

Copyright (c) 2026 Chen

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## xhs-operator skill itself

- License: MIT (inherits from hermes-agent repo)
- Original SOP author: Chen (agent-skills-launch-pack, MIT)
- Real-data rewrite + Hermes/ApexNodes tooling integration: ApexNodes / Hermes Agent contributors
