---
name: hermesclawzero-memory
description: Manages long-term semantic memory using HermesClawZero.
---
# HermesClawZero Memory Manager

This skill manages long-term semantic memory using the HermesClawZero API. It enables persistent, searchable storage for information, task history, and knowledge base items.

## API Configuration
- Base URL: from MEM_PUBLIC_URL (example: http://localhost:8000)
- API Key: from API_KEY

## Tools
1. **capture(text: str)**: Saves text to memory.
2. **search(query: str, limit: int = 5)**: Retrieves relevant memories.
3. **autosave(content: str, filename: str)**: Writes content to a file in the sync folder for automatic indexing.

## Usage
Whenever you have important information, use the `capture` tool. Hermes automatically uses `autosave` to ensure your session history is persisted.
