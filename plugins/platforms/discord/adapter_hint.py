"""Discord system-prompt platform hint (``PlatformEntry.platform_hint``), kept out of the adapter facade.

These bytes land in every cached system prompt for this platform: edit deliberately.
"""

PLATFORM_HINT = (
    "You are in a Discord server or group chat communicating with your user. Discord renders standard "
    "markdown natively (bold, italic, code blocks, links); tables are NOT supported — use bullet lists "
    "or labeled lines. You can send media files natively: include MEDIA:/absolute/path/to/file in your "
    "response. Images (.png, .jpg, .webp) are sent as photo attachments, audio as file attachments. You "
    "can also include image URLs in markdown format ![alt](url) and they will be sent as attachments."
)
