---
name: facebook-access
description: "Access Facebook via Meta Graph API — pages, groups, posts, comments, and messaging."
version: 1.0.0
author: "Ali Ahsan (Aliahasan399)"
license: MIT
platforms: [linux, macos, windows]
prerequisites:
  commands: [python3]
metadata:
  hermes:
    tags: [facebook, graph-api, pages, groups, messaging, social-media, meta]
    homepage: https://github.com/Aliahasan399/facebook-access
    related_skills: [xurl]
---

# Facebook Access Skill

Manage your Facebook Pages and Groups through **Hermes Agent** using the **Meta Graph API**.

> **Created by:** [Ali Ahsan](https://github.com/Aliahasan399) ·
> [LinkedIn](https://www.linkedin.com/in/ali-ahasan-md-moshiur-rahaman-a272343a7) ·
> [Project Repo](https://github.com/Aliahasan399/facebook-access)

## What You Can Do

| Feature | Description |
|---------|-------------|
| **📝 Page Posts** | Create, edit, delete posts on your Pages |
| **📊 Engagement** | View likes, comments, shares, reach analytics |
| **💬 Comments** | Reply to and moderate Page comments |
| **🖼️ Media** | Upload photos and videos to Pages |
| **📋 Page List** | List all Pages you manage |
| **✉️ Inbox** | Read and reply to Page messages |
| **⚙️ Settings** | Update Page metadata and settings |
| **👥 Groups** | Post to groups you belong to |
| **🔑 Token Verify** | Check access token validity and scopes |

## Setup

### 1. Create a Facebook App

1. Go to [Facebook Developers](https://developers.facebook.com/apps/creation/)
2. Click **Create App** → Choose **Business** or **Consumer**
3. Add the **Pages API** product
4. Note your **App ID** and **App Secret**

### 2. Get Access Tokens

Use the [Graph API Explorer](https://developers.facebook.com/tools/explorer/):

1. Select your app → **User Token**
2. Required permissions:
   - `pages_show_list`, `pages_read_engagement`, `pages_manage_posts`
   - `pages_manage_comments`, `pages_manage_photos`, `pages_messaging`
   - `pages_read_user_content`, `publish_to_groups`
3. Click **Generate Access Token**
4. Click **Add to Token** → select your Page → get **Page Access Token**

### 3. Switch to Live Mode

In your App Dashboard: **Settings → Basic → App Mode → Live**

### 4. Set Environment Variable

```bash
export FACEBOOK_ACCESS_TOKEN="your_page_access_token"
```

Or add to your `~/.hermes/.env`:

```
FACEBOOK_ACCESS_TOKEN=your_page_access_token
```

## Usage

This skill includes a Python script (`facebook_graph.py`) that wraps the Meta Graph API.

### Basic Commands

```bash
# List your Pages
python3 facebook_graph.py list-pages

# Create a Page post
python3 facebook_graph.py create-post PAGE_ID "Your message here"

# Get engagement stats
python3 facebook_graph.py get-engagement PAGE_ID

# Reply to a comment
python3 facebook_graph.py reply-comment COMMENT_ID "Your reply"

# Upload a photo
python3 facebook_graph.py upload-photo PAGE_ID /path/to/photo.jpg "Caption"

# List Page inbox conversations
python3 facebook_graph.py list-conversations PAGE_ID

# Verify your token
python3 facebook_graph.py verify-token
```

### All Endpoints

| Command | Description |
|---------|-------------|
| `list-pages` | Show all Pages you manage |
| `create-post PAGE_ID "text"` | Create a text post |
| `create-link-post PAGE_ID URL "msg"` | Post a link with message |
| `get-engagement PAGE_ID` | Get post likes, comments, shares |
| `list-posts PAGE_ID [limit]` | List recent posts (default 10) |
| `delete-post POST_ID` | Delete a post |
| `reply-comment COMMENT_ID "text"` | Reply to a comment |
| `upload-photo PAGE_ID path "caption"` | Upload photo to Page |
| `list-conversations PAGE_ID` | List inbox conversations |
| `send-message CONV_ID "text"` | Send a message in a conversation |
| `post-to-group GROUP_ID "text"` | Post to a group |
| `verify-token` | Check token validity and scopes |

## Files

- `facebook_graph.py` — Main Python script (all API operations)
- `scripts/setup.py` — One-click setup for Hermes Agent

## How It Works

The script uses the **Meta Graph API v19.0+** with simple REST calls:

```
GET  /me/accounts              → List Pages
POST /{page-id}/feed           → Create post
GET  /{page-id}/posts          → List posts
GET  /{post-id}/insights       → Get engagement
POST /{comment-id}/replies     → Reply to comment
POST /{page-id}/photos         → Upload photo
GET  /{page-id}/conversations  → List inbox
POST /{conversation-id}/messages → Send message
```

All requests use `FACEBOOK_ACCESS_TOKEN` for authentication.

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `(#10) Application does not have permission` | Switch app to **Live** mode |
| `(#200) Access token doesn't have permission` | Add missing permissions in Graph API Explorer |
| `Error validating access token` | Token expired — generate a new one |
| `(#100) param message is required` | Post needs non-empty message text |

Your token lasts ~60 days for a User Token. Page Tokens (generated from a User Token with `pages_manage_posts`) can be long-lived by extending them:

```bash
# Exchange short-lived token for long-lived
curl -s "https://graph.facebook.com/v19.0/oauth/access_token?grant_type=fb_exchange_token&client_id=YOUR_APP_ID&client_secret=YOUR_APP_SECRET&fb_exchange_token=SHORT_LIVED_TOKEN"
```
