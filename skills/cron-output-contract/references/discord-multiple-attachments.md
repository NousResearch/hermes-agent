# Discord Multiple Attachments in Cron Deliveries

## The Problem

When a cron produces multiple MEDIA files (e.g. HTML report + audio summary), the scheduler sends them as **separate messages**:

1. Message 1: text content (no attachment)
2. Message 2: first MEDIA file (HTML — separate from text)
3. Message 3: second MEDIA file (audio — separate)

Users see 3 separate posts instead of 1 coherent lesson.

## Root Cause

In `cron/scheduler.py`, line 713:

```python
if platform_name.lower() == "discord" and text_to_send and len(media_files) == 1:
```

This condition only combines text + attachment for exactly 1 file. With 2+ files, text goes as a separate message and each file gets its own follow-up.

## The Fix

Patch `cron/scheduler.py` to iterate over media_files and find the first non-audio file to attach the text as caption. Audio files still send separately (Discord doesn't support captions reliably on audio).

**Applied 2026-05-22.** The fix:

```python
combine_discord_attachment = False
if platform_name.lower() == "discord" and text_to_send and media_files:
    from gateway.platforms.base import should_send_media_as_audio
    first_non_audio_idx = None
    for idx, (m_path, m_voice) in enumerate(media_files):
        m_ext = Path(m_path).suffix.lower()
        if not should_send_media_as_audio(platform, m_ext, is_voice=m_voice):
            first_non_audio_idx = idx
            break
    if first_non_audio_idx is not None:
        if first_non_audio_idx > 0:
            media_files.insert(0, media_files.pop(first_non_audio_idx))
        combine_discord_attachment = True
    elif len(media_files) == 1:
        combine_discord_attachment = True
```

## Behaviour After Fix

| Scenario | Result |
|---|---|
| 1 HTML file (no audio) | Text + HTML in 1 message (caption on doc) |
| 1 HTML + 1 audio | Message 1: text + HTML (caption). Message 2: audio only |
| 2+ non-audio files | Text + first non-audio in 1 message. Rest as separate |
| Audio only | Text + audio in 1 message (caption may not render) |

## Verification

After patching, restart the gateway and manually trigger the cron:

```bash
sudo systemctl restart hermes-gateway
hermes cron run {job_id}
```

Check the Discord channel. The main lesson should be a single message with text + HTML attachment. Audio (if present) should be a separate follow-up message.
