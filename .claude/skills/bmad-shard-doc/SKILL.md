---
name: bmad-shard-doc
description: 'Splits large markdown documents into smaller, organized files based on level 2 (default) sections. Use if the user says perform shard document'
---

# Shard Document

**Goal:** Split large markdown documents into smaller, organized files based on level 2 sections using `npx @kayvan/markdown-tree-parser`.

## CRITICAL RULES

- MANDATORY: Execute ALL steps in the EXECUTION section IN EXACT ORDER
- DO NOT skip steps or change the sequence
- HALT immediately when halt-conditions are met
- Each action within a step is a REQUIRED action to complete that step

## EXECUTION

### Step 1: Get Source Document

- Ask user for the source document path if not provided already
- Verify file exists and is accessible
- Verify file is markdown format (.md extension)
- If file not found or not markdown: HALT with error message

### Step 2: Get Destination Folder

- Determine default destination: same location as source file, folder named after source file without .md extension
  - Example: `/path/to/architecture.md` --> `/path/to/architecture/`
- Ask user for the destination folder path (`[y]` to confirm use of default: `[suggested-path]`, else enter a new path)
- If user accepts default: use the suggested destination path
- If user provides custom path: use the custom destination path
- Verify destination folder exists or can be created
- Check write permissions for destination
- If permission denied: HALT with error message

### Step 3: Execute Sharding

- Inform user that sharding is beginning
- Execute command: `npx @kayvan/markdown-tree-parser explode [source-document] [destination-folder]`
- Capture command output and any errors
- If command fails: HALT and display error to user

### Step 4: Verify Output

- Check that destination folder contains sharded files
- Verify index.md was created in destination folder
- Count the number of files created
- If no files created: HALT with error message

### Step 5: Report Completion

- Display completion report to user including:
  - Source document path and name
  - Destination folder path
  - Number of section files created
  - Confirmation that index.md was created
  - Any tool output or warnings
- Inform user that sharding completed successfully

### Step 6: Handle Original Document

> **Critical:** Keeping both the original and sharded versions defeats the purpose of sharding and can cause confusion.

Present user with options for the original document:

> What would you like to do with the original document `[source-document-name]`?
>
> Options:
> - `[d]` Delete - Remove the original (recommended - shards can always be recombined)
> - `[m]` Move to archive - Move original to a backup/archive location
> - `[k]` Keep - Leave original in place (NOT recommended - defeats sharding purpose)
>
> Your choice (d/m/k):

#### If user selects `d` (delete)

- Delete the original source document file
- Confirm deletion to user: "Original document deleted: [source-document-path]"
- Note: The document can be reconstructed from shards by concatenating all section files in order

#### If user selects `m` (move)

- Determine default archive location: same directory as source, in an `archive` subfolder
  - Example: `/path/to/architecture.md` --> `/path/to/archive/architecture.md`
- Ask: Archive location (`[y]` to use default: `[default-archive-path]`, or provide custom path)
- If user accepts default: use default archive path
- If user provides custom path: use custom archive path
- Create archive directory if it does not exist
- Move original document to archive location
- Confirm move to user: "Original document moved to: [archive-path]"

#### If user selects `k` (keep)

- Display warning to user:
  - Keeping both original and sharded versions is NOT recommended
  - The discover_inputs protocol may load the wrong version
  - Updates to one will not reflect in the other
  - Duplicate content taking up space
  - Consider deleting or archiving the original document
- Confirm user choice: "Original document kept at: [source-document-path]"

## HALT CONDITIONS

- HALT if npx command fails or produces no output files
