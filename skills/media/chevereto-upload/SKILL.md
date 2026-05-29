---
name: chevereto-upload
description: Use when asked to upload images to a self-hosted Chevereto image hosting service via its REST API. Handles batch upload, album association, duplicate detection, and POST-redirect issues.
version: 1.0.0
author: Hermes Agent
license: MIT
metadata:
  hermes:
    tags: [chevereto, image-hosting, upload, batch, api, media]
    related_skills: [gif-search, youtube-content]
---

# Chevereto Upload

## Overview

Upload images to a self-hosted [Chevereto](https://chevereto.com/) image hosting service using its REST API v1. Supports single and batch uploads, album association via numeric album ID, duplicate detection, and handles the POST-redirect behavior common in Chevereto deployments behind reverse proxies.

## When to Use

- User asks to upload images to a Chevereto instance
- User provides a Chevereto URL + API key + local image directory
- User wants images organized into a specific Chevereto album
- Don't use for: Imgur, Cloudinary, or other non-Chevereto services (use their respective skills or tools instead)

## Prerequisites

### 1. Chevereto API Key

- Log into Chevereto as admin
- Go to **Dashboard → Settings → API**
- Copy the API v1 key (e.g. `50a99cff82fd48821b70eb7c815b39de`)

### 2. Album Numeric ID (for album association)

- Go to the album page (e.g. `/album/DCc`)
- Open browser DevTools → Console, run:
  ```javascript
  JSON.parse(document.body.innerHTML.match(/\"id\":(\d+)/)[1])
  ```
- Or check the album JSON from the page source's `window.CHEVERETO` data
- The numeric ID (e.g. `1`) is different from the URL slug (e.g. `DCc`)

### 3. Images must exist locally

Verify the image directory is accessible from the agent's environment.

## Usage

### Step 1: Identify your Chevereto instance and API key

You need three things:
- **Base URL** — e.g. `https://home.shangyiwuyanxinhome.com.cn:10803`
- **API Key** — from Dashboard → Settings → API
- **Album Numeric ID** — from album page inspection (see above)

### Step 2: Identify image directory

Get the local path to images, e.g. `/home/shangyi/disk0/aicg/workspace/team43/images`

### Step 3: Run upload script

```bash
API_KEY="your-api-key-here"
API_URL="https://your-chevereto.com/api/1/upload/"
ALBUM_ID="1"  # numeric album ID, not the URL slug
IMG_DIR="/path/to/images"

for img in "$IMG_DIR"/*; do
  fname=$(basename "$img")
  echo -n "Uploading $fname ... "
  result=$(curl -s -L --post301 -X POST "$API_URL" \
    -F "key=$API_KEY" \
    -F "source=@$img" \
    -F "album=$ALBUM_ID" \
    -F "format=json")

  code=$(echo "$result" | python3 -c "import sys,json; print(json.load(sys.stdin).get('status_code',''))" 2>/dev/null)
  if [ "$code" = "200" ]; then
    url=$(echo "$result" | python3 -c "import sys,json; print(json.load(sys.stdin).get('image',{}).get('url_viewer',''))" 2>/dev/null)
    echo "✓ $url"
  else
    msg=$(echo "$result" | python3 -c "import sys,json; print(json.load(sys.stdin).get('error',{}).get('message',''))" 2>/dev/null)
    echo "✗ $code - $msg"
  fi
done
```

### Step 4: Verify results

- `200` → image uploaded successfully (check `url_viewer` in response)
- `400 Duplicated upload` → image already exists, Chevereto blocks duplicates by MD5
- `400 Invalid API v1 key` → wrong API key
- `400 Invalid API action` → wrong API URL or action
- Empty response → likely a POST-redirect issue, ensure `-L --post301` flags are used

## Key Chevereto API v1 Notes

| Aspect | Detail |
|--------|--------|
| Endpoint | `https://your-site.com/api/1/upload/` |
| Auth | `key=<API_KEY>` form field |
| Album param | `album=<numeric_id>` (not URL slug like `DCc`) |
| HTTP method | POST multipart (`-F source=@file`) |
| Response format | `format=json` |
| Critical flag | `-L --post301` — Chevereto often returns 301 redirect; without this flag, curl drops POST data on redirect |
| Duplicate handling | Chevereto uses MD5 to detect duplicates; set `Allow duplicate uploads` in Dashboard → Settings → Upload to override |
| Album ID vs slug | Album ID `1` ≠ slug `DCc`. API requires numeric ID. |

## Common Pitfalls

1. **Using slug instead of numeric album ID** — API rejects slugs like `DCc`, needs the numeric `1`. Always convert via browser console or page source.

2. **Missing `-L --post301` flags** — Chevereto returns 301 after POST, and curl drops the POST body on redirect by default. Without these flags the API returns `Invalid API v1 key` even with a correct key.

3. **Duplicate upload blocked** — Chevereto blocks re-upload of identical files by MD5. Workarounds: rename the file, enable "Allow duplicate uploads" in settings, or delete the existing image first.

4. **Wrong API key format** — API v1 key must be passed as a form field (`-F key=...`), not as a URL parameter for POST uploads (works for GET though).

5. **Album ID not found in UI** — The album URL slug (`DCc`) is visible everywhere, but the numeric ID is hidden. Use browser DevTools to extract it from page data.

## Finding Album Numeric ID

Open the album page in browser, then in DevTools console:

```javascript
// Option 1: Look for window.CHEVERETO
JSON.stringify(window.CHEVERETO?.album?.id)

// Option 2: Search page HTML for id_encoded vs id
document.body.innerHTML.match(/"id":(\d+)/g)

// Option 3: Check network responses for the album API call
```

The numeric ID is what the API needs, not the `id_encoded` (slug) shown in URLs.

## One-Shot Recipe

```
User says: "上传 /home/user/images 到相册 https://example.com/album/abc"
Steps:
1. Extract base URL: https://example.com
2. Extract slug: abc
3. Find numeric ID: browser inspection (see above)
4. Run upload script with API key from user + numeric album ID
5. Report results
```
