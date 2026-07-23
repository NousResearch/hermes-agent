---
name: onedrive-shared-links
description: Inspect and work with publicly shared OneDrive folders/files, extract useful metadata, and avoid dead ends with large folders or auth redirects.
version: 1.0.0
author: Hermes Agent
license: MIT
metadata:
  hermes:
    tags: [onedrive, shared-links, browser, file-access, troubleshooting]
---

# OneDrive Shared Links

Use this skill when the user provides a public OneDrive/1drv.ms shared link and wants you to inspect files or retrieve specific documents.

## What this skill is for

This is best for:
- Confirming whether a shared OneDrive link is accessible
- Listing visible files/folders in a shared folder
- Finding specific filenames in the shared page
- Extracting useful internal metadata (item ids, api root hints) from the page
- Determining the fastest next step when the folder is very large

This is **not** a guarantee that every file can be downloaded directly through browser automation. OneDrive often allows listing/preview while making deep navigation or large-file retrieval awkward.

## Core findings from experience

1. `browser_navigate()` usually opens public OneDrive shared links successfully.
2. `browser_snapshot(full=true)` can expose visible filenames, sizes, and folder structure in the page.
3. Clicking deeper into large folders can be slow or time out.
4. Direct navigation to inferred OneDrive child-folder URLs can redirect to sign-in even when the parent share is public.
5. Useful file metadata may be embedded in HTML attributes such as `data-drop-target-key` or `data-drag-source-key`.
6. Direct API fetching from `my.microsoftpersonalcontent.com/_api/v2.0/...` may fail from browser context due to CORS/session constraints.
7. For large exports, it is often faster to ask the user for a narrower subfolder link or direct file links instead of crawling the whole share.

## Recommended workflow

### Step 1: Open the shared link
Use browser navigation first.

```json
browser_navigate({"url":"<shared onedrive url>"})
```

### Step 2: Inspect visible items
Take a full snapshot.

```json
browser_snapshot({"full":true})
```

Look for:
- folder names
- filenames
- sizes
- whether the user’s target file is already visible

### Step 3: If the target file is visible, inspect the page state
Use `browser_console()` to inspect DOM text and nearby metadata.

Useful expressions:

```js
document.title
```

```js
document.body.innerText.includes('target-file-name')
```

```js
(() => {
  const html = document.documentElement.outerHTML;
  const idx = html.indexOf('target-file-name');
  return idx === -1 ? null : html.slice(Math.max(0, idx - 1500), idx + 5000);
})()
```

This can reveal internal ids like:
- drive id
- item id
- API root hints

## Practical heuristics

### If the shared folder is large
Do **not** spend many retries clicking through a huge export tree.
Instead:
- confirm access works
- identify the visible relevant files/folders
- ask the user for a direct link to the specific subfolder or file
- prioritize the exact course/project folder over full history exports

### If the user wants specific study/work documents
Prefer this order:
1. direct file link
2. direct subfolder link
3. top-level share crawl

### If a file is huge
For very large `.md`, `.zip`, or export files:
- do not promise full download immediately
- first locate the exact relevant course folder or smaller files
- work with the minimum needed subset

## Reliable outputs you can provide

Even when full download is awkward, you can still provide:
- whether the share is accessible
- the names/sizes of visible files
- whether a target file is present
- which next link/request from the user would unblock efficient analysis

## Good response pattern

Use a grounded answer like:
- “Ja, de OneDrive-link werkt voor mij.”
- “Ik zie map X en bestand Y.”
- “De snelste route is nu een directe link naar subfolder Z of naar bestand A.”

Avoid claiming that you have fully downloaded or parsed a file unless you actually have.

## Anti-patterns

Do not:
- assume public parent-share access means all child URLs can be navigated directly
- promise recursive crawling of huge OneDrive exports without checking page behavior
- claim a direct API route works just because item ids were visible in HTML
- waste many turns brute-forcing OneDrive navigation when a narrower link would solve it faster

## Escalation rule

If OneDrive browsing becomes slow, redirects to sign-in, or blocks deeper retrieval:
- stop brute force
- report exactly what you can already see
- request a direct subfolder/file link
- continue from there

## Best-fit use case discovered

For academic or project support, shared OneDrive works best when the user gives:
- a direct subfolder like `05_opleiding`
- or direct links to the exact documents for the current task

That is faster and more reliable than mining a full chat/export archive first.
