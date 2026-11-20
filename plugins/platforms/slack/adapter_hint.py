"""Slack system-prompt platform hint (``PlatformEntry.platform_hint``), kept out of the adapter facade.

These bytes land in every cached system prompt for this platform: edit deliberately.
"""

PLATFORM_HINT = (
    "You are in a Slack workspace communicating with your user. Standard markdown is auto-converted to Slack "
    "formatting (bold, headers, links, code); tables are NOT supported — use bullet lists or labeled lines. You "
    "can send media files natively: include MEDIA:/absolute/path/to/file in your response. Images (.png, .jpg, "
    ".webp) are uploaded as photo attachments, audio as file attachments. You can also include image URLs in "
    "markdown format ![alt](url) and they will be uploaded as attachments."
)

# Appended (``PlatformEntry.platform_hint_extra_notes``) when ``markdown_blocks`` or ``rich_blocks`` is on: both
# render GFM pipe tables as real Slack tables, so the base hint's no-tables advice would make the model hand-build
# fixed-width tables inside ``` fences instead, which Slack shows as raw monospace text.
BLOCK_TABLES_NOTE = (
    "Block Kit rendering is on for this workspace, so the no-tables advice above does not apply: for tabular data "
    "write a GFM pipe table (a header row, then a |---| separator row) and Slack renders it as a real table. Never "
    "put a table inside a ``` code block or align columns with spaces; that shows up as raw monospace text."
)
