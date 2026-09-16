---
name: notion-backup
description: Backup and classify content to Notion database.
version: 1.0.0
author: Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [Notion, backup, classify, content]
prerequisites:
  commands: [python3]
---

# Notion Backup Skill

This skill allows Hermes to backup content to a Notion database and classify it automatically.

## Prerequisites

- Notion API key and database ID must be set as environment variables:
  - `NOTION_API_KEY`
  - `NOTION_DATABASE_ID`

## Usage

1. Provide the content and title to backup.
2. The skill will classify the content and upload it to the Notion database.

## Example

```bash
hermes run notion-backup --title "My Title" --content "This is the content."
```