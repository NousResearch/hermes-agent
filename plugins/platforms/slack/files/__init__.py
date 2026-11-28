"""What an inbound Slack file puts in the message text, used by ``slack/adapter.py``.

``document``: the ``[Content of <name>]`` text of a small text-like document, and the size ceilings
on inlined file text and on cached files.
``email``: a mail sent to a channel's address, read from its file object (sender, recipients,
subject, plain-text body) instead of its HTML download and cached as a text document; and its
thread-context marker.
"""

from .document import DOCUMENT_MAX, document_injection
from .email import cache_email_file, email_marker, is_email_file

__all__ = ["DOCUMENT_MAX", "cache_email_file", "document_injection", "email_marker", "is_email_file"]
