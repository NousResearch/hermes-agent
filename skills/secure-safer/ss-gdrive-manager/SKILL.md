---
name: ss-gdrive-manager
description: "Google Drive management for Secure Safer Content — creates folder structure, saves articles as Google Docs, uploads images for banners and social graphics. Uses Composio MCP for Google Drive/Docs operations."
version: 1.0.0
author: Hermes Agent via Rafiul
license: MIT
metadata:
  hermes:
    tags: [secure-safer, google-drive, google-docs, storage, images]
    related_skills: [ss-content-generation, social, image_gen]
---

# Secure Safer Google Drive Manager

## Overview

Manages the Secure Safer Content folder structure in Google Drive.
After you create content, this skill handles saving everything in an organized way.

## Folder Structure

```
Secure Safer Content/                    ← Root folder (create once)
├── Articles/                            ← All articles
│   ├── 2026-06-14 - NY Auto Limits/    ← Date-prefixed topic folder
│   │   ├── ny-auto-insurance-limits    ← Google Doc (the article)
│   │   └── images/                     ← Banners + social graphics
│   │       ├── article-banner.png      ← 1200x630
│   │       ├── linkedin-post.png       ← 1200x627
│   │       ├── facebook-post.png       ← 1200x630
│   │       └── x-post.png              ← 1200x675
│   └── ...
└── Templates/                          ← Content templates
    └── compact-research-brief-template
```

## Requirements

Composio MCP server must be connected and authenticated:
```yaml
# Already configured in secure-safer profile
mcp_servers:
  composio:
    url: https://connect.composio.dev/mcp
```

You need to authenticate Google Drive at: https://dashboard.composio.dev

## Workflow

### Step 1: Create Folder Structure
```
composio tool: GOOGLEDRIVE_CREATE_FOLDER
  name: "Secure Safer Content"
  parent: "root"

composio tool: GOOGLEDRIVE_CREATE_FOLDER
  name: "Articles"
  parent: [Secure Safer Content folder ID]

composio tool: GOOGLEDRIVE_CREATE_FOLDER
  name: "2026-06-14 - [Topic]"
  parent: [Articles folder ID]

composio tool: GOOGLEDRIVE_CREATE_FOLDER
  name: "images"
  parent: [Topic folder ID]
```

### Step 2: Save Article as Google Doc
```
composio tool: GOOGLEDOCS_CREATE
  title: "[Topic] — Secure Safer Insurance"
  content: [Full article content with formatting]
  folder: [Topic folder ID]
```

### Step 3: Generate and Upload Images
```
1. Generate banner image via image_gen tool
2. Save to local path
3. Upload via Composio:
   composio tool: GOOGLEDRIVE_UPLOAD_FILE
     file: [local image path]
     parent: [images folder ID]
     name: "article-banner.png"
```

### Step 4: Generate Social Graphics
Repeat Step 3 for:
- linkedin-post.png (1200x627)
- facebook-post.png (1200x630)
- x-post.png (1200x675)

### Step 5: Log URLs
After saving, log the Google Doc URL and image URLs to:
```
/Users/rafiul/Documents/Social Media/Social Media App/_run/Session Log.md
```

## Image Specifications

| Asset | Size | Description |
|-------|------|-------------|
| article-banner.png | 1200x630 | Blog/article header image |
| linkedin-post.png | 1200x627 | LinkedIn feed image |
| facebook-post.png | 1200x630 | Facebook feed image |
| x-post.png | 1200x675 | X/Twitter card image |

## Image Style Guide
- **Colors:** Vibrant Orange (#fc820c) primary on clean white/light gray (#f9f9ff) surfaces. Dark gray (#141b2b) for text. No deep blues.
- **Font:** Plus Jakarta Sans (headlines), Manrope (body)
- **Style:** Minimal, modern, "energetic precision" — clean typography, abstract geometric elements, no heavy shadows
- **Mood:** Approachable authority — professional but warm
- **Insurance context:** Avoid generic stock photos. Use abstract shapes, clean data visualization, typography-focused compositions

## Error Handling

| Problem | What to Do |
|---------|-----------|
| Composio 401 | User needs to authenticate at dashboard.composio.dev |
| Folder already exists | Search for existing folder by name, reuse ID |
| Upload fails | Save files locally to `~/Desktop/secure-safer-images/` temporarily |
| No image_gen tool | Skip image generation, note in session log |

## References
- `references/composio-mcp-setup.md` — full OAuth setup guide for Composio MCP integration
- `_architecture/Brand Design System.md` in vault — brand colors, fonts, image style
- `ss-content-generation` skill — content pipeline that feeds into Drive
