---
name: papersgpt-for-zotero
description: A privacy-first, local-first search assistant and MCP server for your Zotero library, enabling AI agents to search and analyze your research papers securely.
version: 0.0.1
author: awadb.vincent@gmail.com
---

# PapersGPT for Zotero Skill

PapersGPT for Zotero is a specialized research assistant designed to turn your local Zotero library into a searchable, intelligent knowledge base. It helps researchers find relevant materials, gain AI-driven insights, rapidly parse research papers, and streamline the literature review process.

## Key Research Capabilities
- **Effortless Discovery**: Instantly search across your entire collection of research papers to find specific concepts or data points.
- **AI-Powered Insights**: Get synthesized insights from your personal library to accelerate your understanding of complex topics.
- **Rapid Literature Review**: Quickly identify key arguments, methodologies, or findings across multiple documents to build your literature reviews.

## Installation
You can install PapersGPT for Zotero globally using npm. This will make the `pz` and `pgz` commands available in your terminal.

```bash
npm install -g papersgpt-for-zotero
```

After installation, ensure that you have your Zotero storage directory accessible and you are ready to use the `pz` command. More information please see https://github.com/papersgpt/papersgpt-for-zotero.

## When to use
- Use PapersGPT for Zotero when you need to search, analyze, or synthesize information from your personal collection of research papers, PDFs, or academic notes stored in Zotero.
- Use it to accelerate literature reviews, find specific research findings, or quickly look up concepts across your local research database.
- Do not use this for searching the public internet or answering general knowledge questions outside your own document collection.


## How to use
1. **Initialize**: Run `pz init` once to link your Zotero storage and start the background indexing process.
   ```bash
   # Initialize with default Zotero path
   pz init

   # Initialize with a custom path
   pz init "/Users/name/Documents/Zotero/storage"
   ```
2. **Search**: Use `pz search "your research query"` to perform a targeted search across your local files.
   ```bash
   # Find papers on a specific methodology
   pz search "Bayesian inference in clinical trials"

   # Look for specific findings or data points
   pz search "What is the baseline accuracy reported in the 2023 study?"
   ```
3. **Analyze**: Review the context snippets provided by the tool to synthesize answers or findings for your research work.
4. **Manage**: Use `pz stop` when you are finished to shut down the background service.
   ```bash
   pz stop
   ```
## Workflow: Answering Research Questions

1. **Initialize**: Run `pz init` to ensure your library is indexed.
2. **Search**: Use `pz search` with specific queries related to your current research focus.
3. **Synthesize**: Review the relevant snippets to gain insights or build arguments for your literature review.
4. **Cite**: Always verify and cite your sources directly from your Zotero library.

## Troubleshooting

- If `pz search` returns no results, ensure you have initialized your library with `pz init`.
- If performance seems slow initially, it may be due to the background indexing process. The system will become faster as more files are indexed.
