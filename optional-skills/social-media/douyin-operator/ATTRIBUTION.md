# Attribution

This skill is adapted from a third-party MIT-licensed skill. All reuse is
credited here and the original copyright notice is reproduced in full below.

## douyin-account-launch-expert (agent-skills-launch-pack)

- Source: https://github.com/chenjin-cmd/agent-skills-launch-pack_
- Skill: `skills/douyin-account-launch-expert/`
- License: MIT
- Copyright: © 2026 Chen

### What was adapted (SOP skeleton — reused)

The compliance-first operating skeleton is reused, restructured for Hermes
Agent's SKILL.md conventions:

- 起号模式判定 (launch-mode routing: IP-trust / knowledge / traffic / repair / scale)
- 定位句 (positioning sentence) + 主页地基 (profile foundation)
- 简报先行 (brief-before-draft) two-stage note workflow (口播脚本 + 封面)
- 内容日历 (content calendar) + 周复盘 (weekly review) loop
- 风险边界 (risk boundaries) — no fake engagement, no rewrite-to-evade,
  no copied video/BGM/cover, no 广告法 极限词, no promised outcomes.
  Reproduced with intent.

### What was rewritten (real-data + transcription workflow — the ApexNodes delta)

Several sections were rewritten from prompt-only to **real-data** workflows that
call the platform social-data tools (`social_search` / `social_trending` /
`social_posts` / `creator_top_posts` / `social_content` / `social_comments` /
`social_profile` with `platform="douyin"`) instead of asking the model to invent
competitor benchmarks, viral hooks, or trending topics:

- **对标拆解 (competitor teardown)** — now pulls real 抖音 videos via the tools
  (heat-board first via `social_trending`, then `social_search` /
  `creator_top_posts`, which douyin supports) and abstracts 3-second hooks /
  cover patterns / structure from live data.
- **选题库 (topic bank)** — now a field table seeded from real trending/search
  results, with hooks traced to real videos/comments and an honest note on which
  signals are external-unavailable.

**The douyin-specific advantage that the WeChat-MP and 小红书 skills do not have:**

- **爆款视频转写拆解 (viral-video transcription teardown)** — a new workflow step
  that instructs the agent to use the platform's **video download + transcription**
  tools (`social_download` + `media_transcribe`, or one-shot
  `media_transcribe url=…`, and `social_batch_submit` / `social_batch_status` for
  batches) to pull the spoken-word transcript of a viral short video and dissect
  its hook / pacing / structure / closing. Short-form video hooks live in the
  audio, not the title — so this is the real edge for a video platform. The skill
  hard-forbids self-installing a local whisper and honors the
  `terminal_fallback_allowed=false` honesty signal.
- **9-video experiment matrix** (3 topics × 3 formats) with post-publish re-scrape
  via `social_posts` / `social_content` and an explicit "small sample, signal ≠
  conclusion" caveat.

Cover-image generation is wired to the runtime's own image-generation
capability with routing chosen by the runtime tier (带字封面 default; a
free cover line — platform base image + typographic overlay, hc-433 — also
selectable within the tier). The skill does not hard-code a provider. Skipped
honestly when no image-generation capability is available.

### Honest scope notes (ApexNodes environment)

- 抖音 **does** have a public trending board (`social_trending` with
  `platform="douyin"`), unlike 小红书/公众号 — the topic bank uses it directly.
- **Cross-platform topic sourcing via weibo (微博热搜)**: the cloud added the
  weibo platform (hc-426), but this desktop fork's gateway whitelist
  (`plugins/apexnodes_gateway.py::SOCIAL_PLATFORMS`, 10 platforms) does **not**
  include weibo yet. The skill leaves an inline note to wire
  `social_trending platform=weibo` once weibo reaches the fork whitelist, and
  meanwhile covers douyin-native hot topics with douyin `social_trending` +
  `social_search` (it explicitly warns against calling weibo, which would hit
  `platform_not_allowed`).
- Creator-backend metrics (play count, completion rate, 5s completion, average
  watch time, follower growth, DOU+/qianchuan promotion, 星图 quote) are
  external-unavailable; the skill asks the user to screenshot them rather than
  inventing numbers.

### Compliance red line (hc-075)

The skill does **not** design comment auto-reply or private-message automation.
Comment-section analysis is used only to produce **human** reply strategy and a
pinned-comment hook — content-strategy layer only, no automation.

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

## douyin-operator skill itself

- License: MIT (inherits from hermes-agent repo)
- Original SOP author: Chen (agent-skills-launch-pack, MIT)
- Real-data + transcription rewrite + Hermes/ApexNodes tooling integration: ApexNodes / Hermes Agent contributors
