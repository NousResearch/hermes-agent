---
description: Automatically process theme materials submitted to the theme material ingestion group (oc_a19b4f58f14f7bea48a67610eb0bcb33). Receives, classifies, and stages materials for review.
---

# Theme Material Ingestion Skill

You are processing messages from the **主题资料录入群** (Theme Material Ingestion Group).

## Your Task
When a message arrives in this group and meets the trigger conditions, you must:

1. **Identify the message source** - Record chat_id, sender name, message_id, and timestamp
2. **Extract content** - Handle text, links, images, and documents appropriately
3. **Classify the topic** - Match against known themes with confidence level
4. **Stage the material** - Write to staging directory, never directly to main wiki
5. **Respond in group** - Confirm receipt or ask clarifying questions

## Trigger Conditions
A message triggers this workflow when it:
- Is from chat_id `oc_a19b4f58f14f7bea48a67610eb0bcb33`
- AND contains `@` mention of the bot **OR** trigger keywords **OR** attachments

**Trigger Keywords**: `录入`, `入库`, `归档`, `转 Markdown`, `整理资料`, `新资料`

## Existing Path Mapping

| topic | existing_path | duplicate_policy |
|-------|-------------|-----------------|
| `competition_aild` | `projects/competition-consulting-qa/aild/` | reuse_existing |
| `competition_emergency_safety` | `projects/competition-consulting-qa/emergency-safety/` | reuse_existing |
| `chuangqingchun` | `TODO: confirm path` | reuse_existing |
| `unknown` | N/A (requires human confirmation) | N/A |

**Duplicate policy**: `reuse_existing` — do NOT create new directories; reuse the existing path for each known topic.

## Built-in Topic Definitions

### 1. `competition_aild`
- **Name**: AILD 智能设计大赛
- **HIGH confidence**: `AILD`, `aild.caa.org.cn`, `智能设计大赛`
- **MEDIUM confidence**: `aild`
- **Existing path**: `projects/competition-consulting-qa/aild/`

### 2. `competition_emergency_safety`
- **Name**: 全国青少年应急与安全科普创新大赛
- **HIGH confidence**: `nyseic.cn`, `全国青少年应急与安全科普创新大赛`, `应急安全`
- **MEDIUM confidence**: `应急与安全`
- **Existing path**: `projects/competition-consulting-qa/emergency-safety/`

### 3. `chuangqingchun`
- **Name**: 创青春大赛
- **HIGH confidence**: `创青春`, `中银杯`
- **MEDIUM confidence**: `创业大赛`, `中国青年创青春`, `天津青年创青春`
- **Existing path**: `TODO: confirm path`

### 4. `unknown`
- Cannot determine topic → requires user confirmation before staging
- **Never** write directly to official wiki for `unknown` topics

## Confidence Levels
- **HIGH**: Explicit match on topic name or official domain
- **MEDIUM**: Multiple weak keyword matches
- **LOW**: Only generic terms (competition, notification, plan, etc.)
- **UNKNOWN**: No match possible → ask for clarification

## Step-by-Step Processing

### Step 1: Extract Message
```
- chat_id: oc_a19b4f58f14f7bea48a67610eb0bcb33
- message_id: <from event>
- sender: <from event>
- timestamp: <from event>
- has_attachment: true/false
- text_content: <raw text or "" >
```

### Step 2: Classify Topic
Check text against each topic's keywords in order:
1. Try HIGH confidence matches first
2. Fall back to MEDIUM confidence
3. If no match → `unknown`

### Step 3: Stage Material
- Stage to: `projects/_staging/materials/<topic>/<timestamp>_<sender>_<message_id>/`
- Never write directly to `projects/competition-consulting-qa/*/official/`
- Create a `manifest.md` inside the staging dir listing all received files

### Step 4: Respond in Group
**For known topics (HIGH/MEDIUM confidence)**:
```
已收到资料，主题：[topic_name]
资料已暂存至待审区，请等待人工审核后入库。
```

**For `unknown`**:
```
您好，系统无法自动识别此资料主题。
请确认资料属于哪个主题（AILD / 应急安全 / 创青春），或联系管理员确认后再录入。
```

**For attachments** (images/PDF/Word):
```
已收到附件，正在处理中...
```
→ Download and stage to the same staging directory.

## Staging Directory Structure
```
projects/_staging/materials/<topic>/<timestamp>_<sender>_<message_id>/
├── manifest.md          # List of all received items
├── original/           # Original files (images, PDFs, Word docs)
└── converted/          # Converted Markdown (if applicable)
```

## Rules
1. **Never write directly to official wiki** (`projects/competition-consulting-qa/*/official/`)
2. **Always stage first** → GPT/人工审核 → 才能 move to official
3. **For `unknown` topics**: always ask in group, never guess
4. **For `chuangqingchun`**: flag as needing path confirmation until `existing_path` is set
5. **Duplicate files**: `duplicate_policy: reuse_existing` — do not overwrite; append with timestamp suffix if needed
