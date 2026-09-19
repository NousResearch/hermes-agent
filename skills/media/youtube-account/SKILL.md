---
name: youtube-account
description: "Read a user's private YouTube channel, subscriptions, and playlists through a profile-scoped OAuth grant. Use for account-specific YouTube requests, not public transcripts or YouTube Music/TV control."
metadata:
  hermes:
    tags: [YouTube, OAuth, Media]
---

# YouTube Account

Use this skill for read-only access to the YouTube account authorized in the
current Hermes profile. The grant is deliberately separate from Google
Workspace and uses only `https://www.googleapis.com/auth/youtube.readonly`.

For public video transcripts and summaries, use `youtube-content` instead.
YouTube Music and YouTube TV do not have account-control routes in this skill.

## Setup

`SKILL_DIR` is the directory containing this file. Run scripts with the same
Hermes-managed Python environment used by the active profile.

1. Check the current profile:

   ```bash
   python SKILL_DIR/scripts/setup.py --check
   ```

2. If needed, store a Google Desktop OAuth client. Reusing the same client
   application as Google Workspace is allowed; the YouTube token remains
   separate.

   ```bash
   python SKILL_DIR/scripts/setup.py --client-secret /path/to/client.json
   ```

3. Generate an authorization URL and give it to the user. Never automate or
   collect their Google password, MFA, or security-key response.

   ```bash
   python SKILL_DIR/scripts/setup.py --auth-url
   ```

4. The redirect to `http://localhost:1` is expected to fail. Ask the user for
   the full redirected URL, then exchange it (or its raw `code` value):

   ```bash
   python SKILL_DIR/scripts/setup.py --auth-code 'FULL_REDIRECT_URL_OR_CODE'
   python SKILL_DIR/scripts/setup.py --check
   ```

Google Cloud must have YouTube Data API v3 enabled for the OAuth project. Treat
administrator or Google policy denials as authoritative.

## Read-only commands

```bash
python SKILL_DIR/scripts/youtube_api.py channel
python SKILL_DIR/scripts/youtube_api.py subscriptions --max 25
python SKILL_DIR/scripts/youtube_api.py playlists --max 25
python SKILL_DIR/scripts/youtube_api.py playlist-items PLAYLIST_ID --max 25
```

Return only the private account data the user requested. Never print or expose
the OAuth client, access token, refresh token, or pending PKCE state. This skill
has no write operations.
