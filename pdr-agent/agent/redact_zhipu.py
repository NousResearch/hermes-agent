"""Zhipu API key pattern for agent.redact."""

import re

# Zhipu API keys use an unprefixed ``id.secret`` form. Keep this deliberately
# provider-shaped instead of applying a generic high-entropy dotted-token rule:
# the ID is exactly 32 lowercase hex chars and the credential suffix is a run of
# at least 16 alphanumerics, so content-hash filenames (``<sha>.bundle``,
# ``<md5>.sqlite3``) never match.
_ZHIPU_API_KEY_RE = re.compile(
    r"(?<![A-Za-z0-9_.-])([0-9a-f]{32}\.[A-Za-z0-9]{16,})(?![A-Za-z0-9_.-])"
)
