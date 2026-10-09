"""Launchd definition comparison helpers."""

from gateway.service_definition import normalize_service_definition

def _normalize_launchd_plist_for_comparison(text: str) -> str:
    """Normalize plist text for staleness checks, ignoring the PATH payload: the generated PATH is
    captured from the invoking shell and varies across shells."""
    import re
    return re.sub(
        r"(<key>PATH</key>\s*<string>)(.*?)(</string>)", r"\1__HERMES_PATH__\3",
        normalize_service_definition(text), flags=re.S,
    )
