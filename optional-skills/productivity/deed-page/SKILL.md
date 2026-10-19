---
name: deed-page
description: Publish files to a permanent deed.page URL. No account.
version: 1.0.0
author: Avatar 8 LLC (@f-o-x11), Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    category: productivity
    tags: [deed.page, publish, deploy, hosting, static-site, web, share, URL]
    homepage: https://deed.page
    requires_toolsets: [terminal]
---

# deed.page Skill

Publishes a file or folder Hermes generated (a report, dashboard, slide deck, or page) to a permanent `https://<slug>.deed.page/` URL in one request, with no account, API key, email code, or claim link. It serves static files only. It does not run server code or store private data, and free sites are public.

## When to Use

- The user asks to publish, host, deploy, share online, or "give me a link" for an HTML file or a static folder.
- The link must keep working after today, and nobody can sign up or click a claim link.

Do not use it for secrets, personal data, or anything the user wants private.

## Prerequisites

- Python 3.9 or later (stdlib only). Network access to `https://deed.page`.
- Optional: `DEEDPAGE_TOKEN`. It is returned by the first publish and needed to update or delete.

## How to Run

Run the helper with the `terminal` tool. The path is relative to this skill:

```bash
python scripts/deed_publish.py ./report.html --slug q3-report-7f3a2c
python scripts/deed_publish.py ./site-folder --slug q3-report-7f3a2c
python scripts/deed_publish.py --delete q3-report-7f3a2c
```

It prints one JSON object. Answer the user with its `url`.

## Quick Reference

| Need | Command or field |
| --- | --- |
| Live URL | `url` in the output |
| Update the same slug | Re-run with `DEEDPAGE_TOKEN` set |
| Expiring page | `--ttl 86400` |
| Single-page app | `--spa` |
| Delete | `--delete <slug>` with `DEEDPAGE_TOKEN` set |
| Docs | https://deed.page/docs |

## Procedure

1. Pick a slug: lowercase letters, digits, and hyphens, 1 to 63 characters, with a random suffix. `hello`, `my-page`, `my-app`, and `my-site` are reserved.
2. Run the helper on the file or folder. A folder is sent as a tarball with its contents at the site root, so it should contain `index.html`.
3. If the output has `token`, it is shown only once. Tell the user it is the key to update or delete the site, and keep it as `DEEDPAGE_TOKEN` for this session. Never put it in a published page, a commit, or a log.
4. Reply with `url`.

## Pitfalls

- `409 slug_taken`: use the `suggestion` field or a new suffix. Slugs are never rewritten. A bad slug returns `400 slug_invalid`.
- `422 content_rejected`: the upload contained an executable or phishing markers. Remove the file and do not retry.
- A single request is capped at 4 MB. Larger sites need the manifest upload described in the docs.
- `429`: wait `retry_after` seconds. The Free plan allows 60 publishes per hour and 25 sites.
- A lost token cannot be recovered unless the owner attached an optional recovery email. The site stays online.

## Verification

Fetch `url` with `web_extract` or `curl -sI` from `terminal` and confirm a 200. After `--delete`, the same URL returns 404.
