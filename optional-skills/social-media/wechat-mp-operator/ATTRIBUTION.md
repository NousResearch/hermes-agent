# Attribution

This skill is adapted from a third-party MIT-licensed skill. All reuse is
credited here and the original copyright notice is reproduced in full below.

## wechat-account-launch-expert (agent-skills-launch-pack)

- Source: https://github.com/chenjin-cmd/agent-skills-launch-pack_
- Skill: `skills/wechat-account-launch-expert/`
- License: MIT
- Copyright: © 2026 Chen

### What was adapted (SOP skeleton — reused)

The compliance-first operating skeleton is reused, restructured for Hermes
Agent's SKILL.md conventions:

- 起号模式判定 (launch-mode routing: IP-trust / traffic / repair / scale)
- 定位句 (positioning sentence) + 账号地基 (account foundation)
- 简报先行 (brief-before-draft) two-stage article workflow
- 发布节奏 (publish cadence) + 周复盘 (weekly review) loop
- 风险边界 (risk boundaries) — no fake engagement, no rewrite-to-evade,
  no copied images/text, no promised outcomes. Reproduced with intent.

### What was rewritten (real-data workflow — the ApexNodes delta)

Two sections were rewritten from prompt-only to **real-data** workflows that
call the platform social-data tools (`social_search` / `social_posts` /
`social_content` / `social_comments` / `social_profile` with
`platform="wechat_mp"`) instead of asking the model to invent competitor
benchmarks:

- **对标拆解 (competitor teardown)** — now pulls real 公众号 articles via the
  tools and abstracts title patterns / structure / hooks from live data.
- **选题库 (topic bank)** — now a field table seeded from real search results,
  with an honest note on which signals are external-unavailable.

Cover-image generation is wired to the runtime's own image-generation
capability (default caption-cover model), skipped honestly when unavailable.

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

## wechat-mp-operator skill itself

- License: MIT (inherits from hermes-agent repo)
- Original SOP author: Chen (agent-skills-launch-pack, MIT)
- Real-data rewrite + Hermes/ApexNodes tooling integration: ApexNodes / Hermes Agent contributors
