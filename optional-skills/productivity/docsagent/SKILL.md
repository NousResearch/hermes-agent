---
name: docsagent
description: Search and manage private, local document collections (PDF, PPTX, DOCX) offline. Use when you need to find information within your private files, not for web research.
author: awadb.vincent@gmail.com
version: 0.1.0
---

# DocsAgent Skill

DocsAgent is a local-first documentation engine designed to perform lightning-fast **hybrid search** (text and semantic search) over your private local files.

## Key Features
- **Privacy-First**: All indexing and searching happen 100% locally on your machine. No documents or data ever leave your system.
- **Hybrid Search**: Combines semantic and keyword-based text search for high accuracy.
- **Supported Formats**: Optimized for **PDF, PPTX, and DOCX** files.
- **No Web Dependency**: This tool does not perform web searches. It relies on a local index you create.

## Privacy Guarantee
*   **100% Local**: All processing, indexing, and search operations are performed entirely on your machine.
*   **Zero Data Leakage**: No data is sent to external servers. Your documents remain private.

## When to use
- Use this when searching for information within your own private documentation, research papers, reports, or presentations.
- Do not use this for general internet queries, news, or public website information.

## Installation
DocsAgent is a CLI toolkit, when you install the global `docsagent` package, You can directly use 'docsagent' command.
```bash
npm install -g @docsagent/docsagent
```
For more information, please see https://github.com/docsagent/docsagent.


## Usage Guidelines

### 1. Add Documentation
Adds directory or file paths (PDF, DOCX, PPTX) to the index. You can pass multiple paths at once.
```bash
# Add a single directory
docsagent add ./documents/manuals

# Add multiple files and directories at once
docsagent add ./reports/q1.pdf ./presentation.pptx ./notes/meeting-files
```
*   **Use case**: Use to index folders or specific private files for searchability.

### 2. Search Documents
Performs a hybrid search (text + semantic) across your indexed private documents.
```bash
# Search for a specific topic
docsagent search "how to configure local encryption"

# Search for technical concepts
docsagent search "API authentication process"
```
The tool returns the most relevant snippets with their source paths and scores.
*   **Important**: Ensure your documents have been indexed using the `add` operation before searching.

### 3. List Indexed Sources
Lists all files currently being monitored or indexed.
```bash
docsagent list
```

### 4. Check Status
Shows the indexing progress and service health.
```bash
docsagent status
```

### 5. Stop Agent
Stops the indexing service.
```bash
docsagent stop
```

## Workflow: Answering Questions from Local Docs

1. **Check Index**: If unsure if the documents are indexed, you can try a `search` first.
2. **Search**: Execute `search` with a specific query derived from the user's request.
3. **Analyze**: Review the context returned. If multiple snippets are found, synthesize them into a coherent answer.
4. **Cite Sources**: Always mention the file path and page number (if available) in your response.

## Error Handling

- If `search` returns "No results found", inform the user and ask if they want to index a specific directory using `add`.
- If a path is not found during `add`, verify the path exists before retrying.

