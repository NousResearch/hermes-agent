---
name: feishu-card-builder
description: Build and send Feishu interactive card messages — metric cards, table cards, list cards  
version: 1.0.0
author: Minksgo
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [feishu, lark, card, interactive, messaging, notification]
    category: communication
---

# Feishu Card Builder

Compose rich **interactive card messages** for Feishu/Lark without manually constructing JSON.

## When to Use

- User asks for a **status dashboard**, **metrics overview**, or **KPI report** → use `MetricCard`
- User wants **structured data** shown as a table → use `TableCard`  
- User asks for a **task list**, **checklist**, or **itemized content** → use `ListCard`
- User says "send me a card" or "send this to Feishu nicely formatted"

## How to Use

The Python module `scripts/feishu_card_builder.py` provides builder classes.
Run it via `terminal` to generate card JSON, then send it through the Feishu gateway.

### MetricCard — for KPIs and stats

```python
python3 ~/.hermes/skills/communication/feishu-card-builder/scripts/feishu_card_builder.py metric \
  --header "📊 System Status" \
  --items "Uptime:99.9%" "Errors:0" "Users:1,247"
```

### TableCard — for structured data

```python
python3 ~/.hermes/skills/communication/feishu-card-builder/scripts/feishu_card_builder.py table \
  --header "📋 Rankings" \
  --columns "Rank,Name,Score" \
  --rows "1,Alice,98" "2,Bob,87" "3,Carol,92"
```

### ListCard — for checklists and task lists

```python
python3 ~/.hermes/skills/communication/feishu-card-builder/scripts/feishu_card_builder.py list \
  --header "📌 Pending Tasks" \
  --items "✅ Review PR:Needs approval" "⏳ Run CI:Waiting" "📝 Write docs:In progress"
```

## Card Colors

Available header colors: `blue`, `green`, `red`, `yellow`, `orange`, `purple`, `grey`

Add `--color green` to any command.

## Notes

- Cards are sent as `msg_type=interactive` via the Feishu gateway
- Wide screen mode is disabled by default (`wide_screen_mode: false`) for mobile readability
- Table cards render as markdown grids which Feishu's post API renders as bordered tables
- No additional API keys or dependencies required
