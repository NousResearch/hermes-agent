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

---

## Trigger Conditions
A message triggers this workflow when it:
- Is from chat_id `oc_a19b4f58f14f7bea48a67610eb0bcb33`
- AND contains `@` mention of the bot OR trigger keywords OR attachments

**Trigger Keywords**: 录入, 入库, 归档, 转 Markdown, 整理资料, 新资料

---

## Built-in Topic Definitions

### 1. `competition_aild`
- **Name**: AILD 智能设计大赛
- **HIGH confidence keywords**: `AILD`, `aild.caa.org.cn`, `智能设计大赛`
- **MEDIUM confidence keywords**: `aild`

### 2. `competition_emergency_safety`
- **Name**: 全国青少年应急与安全科普创新大赛
- **HIGH confidence keywords**: `nyseic.cn`, `全国青少年应急与安全科普创新大赛`, `应急安全`
- **MEDIUM confidence keywords**: `应急与安全`

### 3. `chuangqingchun`
- **Name**: 创青春大赛
- **HIGH confidence keywords**: `创青春`, `中银杯`
- **MEDIUM confidence keywords**: `创业大赛`, `中国青年创青春`, `天津青年创青春`

### 4. `unknown`
- Cannot determine topic → requires user confirmation before staging

---

## Confidence Levels
- **HIGH**: Explicit match on topic name or official domain (e.g., `aild.caa.org.cn` matches `competition_aild`)
- **MEDIUM**: Multiple weak keyword matches
- **LOW**: Only generic terms (competition, notification, plan, etc.)
- **UNKNOWN**: No match possible

---

## Step-by-Step Processing

### Step 1: Record Message Source
Extract and record:
- `source_chat_id`: `oc_a19b4f58f14f7bea48a67610eb0bcb33`
- `source_message_id`: The message ID from the event
- `source_sender`: The sender's display name
- `captured_at`: Current time in UTC+8 format (YYYY-MM-DD HH:mm:ss +0800)

### Step 2: Extract Content
- **Text**: Extract directly from message body
- **Links**: Preserve URL. If possible, attempt to fetch title and description. If fetch fails, mark as `pending_manual_review`
- **Images**: If vision/OCR capability is available, attempt text extraction. Otherwise, save image path as attachment
- **Documents (Word/PDF)**: If document conversion is available, convert to Markdown. Otherwise, save file metadata

### Step 3: Classify Topic
Match the extracted content against topic definitions:

1. Check for HIGH confidence keywords first
2. Check for MEDIUM confidence keywords
3. If multiple topics match with equal confidence, mark as `unknown` and ask for clarification
4. If no keywords match, the topic is `unknown`

### Step 4: Handle Uncertainty
Ask clarifying questions in the group if:
- `topic_id = unknown`
- `confidence = low`
- Source/origin is unclear
- Date/time is unclear
- Content appears to be a paraphrase
- Content could belong to multiple topics

**Clarification Questions Format**:
```
这份资料暂不能直接入库，需要确认：
1. 归属主题是哪个？
2. 是否为官方原文或已确认材料？
3. 是否允许作为候选 source 进入知识库？
```

### Step 5: Stage the Material
Write to the staging directory structure:
```
projects/_staging/materials/theme-ingestion/{topic_id}/YYYY-MM-DD/{message_id}/
```

Create these files:
- `raw_message.md` - Original message content
- `candidate_source.md` - Processed content with front matter
- `attachments_manifest.json` - List of attachments with metadata
- `ingestion_report.md` - Processing log

### Step 6: Front Matter
Every `candidate_source.md` must include this front matter:
```yaml
---
source_type: feishu_group_material
source_chat_id: oc_a19b4f58f14f7bea48a67610eb0bcb33
source_message_id: "<message_id>"
source_sender: "<sender_display_name>"
captured_at: "<YYYY-MM-DD HH:mm:ss +0800>"
topic_id: "<topic_id>"
topic_confidence: "<high|medium|low|unknown>"
review_status: pending_user_review
target_project: ""
source_url: ""
original_filename: ""
sensitive_review_required: true
---
```

### Step 7: Respond in Group
- **For identifiable materials**: 
  ```
  已收到资料，已生成候选文档，请在审核确认后继续操作。
  ```
- **For uncertain materials**: Use the clarification format from Step 4

---

## Important Rules
1. **Never write directly to main wiki** - Only write to `projects/_staging/`
2. **All materials require review** - Set `review_status: pending_user_review`
3. **Sensitive materials require extra review** - Set `sensitive_review_required: true`
4. **Media files are cached locally** - Reference by path in `attachments_manifest.json`
5. **Do not delete any existing materials** - This skill only creates new staging entries

---

## Staging Directory Structure
```
projects/_staging/materials/theme-ingestion/
├── competition_aild/
│   └── YYYY-MM-DD/
│       └── {message_id}/
│           ├── raw_message.md
│           ├── candidate_source.md
│           ├── attachments_manifest.json
│           └── ingestion_report.md
├── competition_emergency_safety/
│   └── ...
├── chuangqingchun/
│   └── ...
└── unknown/
    └── ...
```

## Example Processing Flow
1. Message arrives with text "请帮忙录入 AILD 智能设计大赛的通知文件"
2. Extract sender, message_id, timestamp
3. Match "AILD" and "智能设计大赛" → HIGH confidence for `competition_aild`
4. Create staging directory: `projects/_staging/materials/theme-ingestion/competition_aild/2026-05-29/{message_id}/`
5. Write files: `raw_message.md`, `candidate_source.md` (with front matter), `ingestion_report.md`
6. Respond: "已收到资料，已生成候选文档，请在审核确认后继续操作。"
