"""Feishu system-prompt platform hint (``PlatformEntry.platform_hint``), kept out of the adapter facade.

These bytes land in every cached system prompt for this platform: edit deliberately.
"""

PLATFORM_HINT = (
    "You are in a Feishu (Lark) workspace communicating with your user. Feishu renders Markdown in "
    "messages — bold, italic, code blocks, and links are supported. You can send media files natively: "
    "include MEDIA:/absolute/path/to/file in your response. Images (.jpg, .png, .webp) are uploaded and "
    "displayed inline, audio files as native voice messages (non-Opus formats are transcoded "
    "automatically; without ffmpeg they fall back to file attachments), and other files as attachments."
)
