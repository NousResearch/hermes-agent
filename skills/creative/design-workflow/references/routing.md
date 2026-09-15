# Task Routing

Start every design task by routing it. If the route is obvious, proceed without asking. If platform or goal materially changes the work, ask one clarifying question.

## Route A — New Page or Screen

Trigger examples:
- "做一个官网首页"
- "设计一个 dashboard"
- "帮我做 onboarding"
- "做一个 iOS 设置页"

Flow:

```text
Understand → Shape → Direction → Craft → Verify → Polish → Audit
```

Required output:
- Design Brief
- structure / section plan / screen zones
- visual direction and style dials
- implementation or spec guidance
- verification notes

## Route B — Redesign Existing UI

Trigger examples:
- "这个页面太 AI 了"
- "不好看，帮我高级一点"
- "帮我重设计这个页面"
- "这个移动端体验很差"

Flow:

```text
Critique → Distill → Repair → Style Tune → Verify → Audit
```

Required output:
- what works
- what fails
- P0/P1/P2 issues
- concrete fix order
- before/after verification if implementation is available

## Route C — Style Exploration

Trigger examples:
- "给我几个方向"
- "参考 Apple/Linear/Stripe 的感觉"
- "这个产品应该走什么视觉方向？"

Flow:

```text
Style Brief → 2-3 Concepts → Compare → Pick → Craft
```

Use local skills when useful:
- `sketch` for quick variants
- `claude-design` for polished HTML concepts
- `popular-web-designs` for reference vocabularies

## Route D — Final Audit

Trigger examples:
- "上线前检查"
- "帮我 audit"
- "看看还有什么 UX 问题"

Flow:

```text
Audit → Fix Plan → Recheck
```

Output must be prioritized:
- P0 must fix
- P1 should fix
- P2 polish
- Pass
- Not checked

## Route E — Native Mobile Product Screen

Trigger examples:
- "设计一个 iOS 页面"
- "这个 Android app screen 怎么改"
- "移动端原生页面也兼容"

Flow:

```text
Platform Context → Shape → Native Direction → Interaction/State Review → Audit
```

Check:
- iOS vs Android conventions
- navigation/back behavior
- safe areas/status bars
- thumb reach
- keyboard and input behavior
- Dynamic Type / text scaling
- VoiceOver / TalkBack
- permission, loading, empty, error states
