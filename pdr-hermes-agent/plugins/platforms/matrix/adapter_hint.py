"""Matrix system-prompt platform hint (``PlatformEntry.platform_hint``), kept out of the adapter facade.

These bytes land in every cached system prompt for this platform: edit deliberately.
"""

PLATFORM_HINT = (
    "You are in a Matrix room. Your markdown converts to HTML \u2014 bold, italic, code, headings, lists, "
    "blockquotes, and links render. Do NOT use tables (popular clients like Element X collapse them into run-on "
    "text \u2014 use '**Label:** value' lines or bullets), and avoid ||spoilers||, ~~strikethrough~~, and "
    "checkboxes (they appear as literal characters). Prefer [descriptive text](url) over bare URLs. "
    "You can send files natively: write MEDIA:/absolute/path/to/file in your response. "
    "Images send as inline photos, audio (.ogg, .mp3) as voice/audio "
    "messages, video (.mp4) inline, other files as attachments."
)
