"""PTB filter for rich-only messages, without stealing ordinary attachment handlers."""

from telegram.ext import filters

from plugins.platforms.telegram.telegram_rich_inbound import has_rich_message


class RichMessageFilter(filters.MessageFilter):
    def filter(self, message):
        return has_rich_message(message)


def rich_text_filter():
    # No payload walk here: sender/chat/topic gates still run before projection.
    attachments = (filters.ATTACHMENT | filters.LOCATION |
                   getattr(filters, "VENUE", filters.LOCATION))
    return RichMessageFilter() & ~attachments
