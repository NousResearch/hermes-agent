"""Bang-command text parsing for the Slack adapter.

Turns a typed ``!cmd`` into the gateway's ``/command`` text (consulting the command
registry), honouring the optional namespace prefix shared with native slashes, which
adapter_slash_payload decodes.
"""

from __future__ import annotations


def _rewrite_known_bang_command(text: str, prefix: str = "") -> str:
    """Rewrite a known leading ``!cmd`` to the gateway ``/cmd`` form.

    Mirrors the native-slash surface: with a namespace prefix configured
    (e.g. ``myorg-``), bang commands carry the same prefix (``!myorg-stop``)
    and it is stripped before dispatch. Unprefixed bangs stay plain text so
    two namespaced apps sharing a channel do not both execute ``!stop``.
    """
    if not text.startswith("!"):
        return text
    tokens = text[1:].split(maxsplit=1)
    if not tokens:
        return text
    cmd_name = tokens[0].split("@", 1)[0].lower()
    rest = text[1:]
    if prefix:
        if not cmd_name.startswith(prefix):
            return text
        cmd_name = cmd_name[len(prefix) :]
        rest = rest[len(prefix) :]
    if not cmd_name or "/" in cmd_name:
        return text
    from hermes_cli.commands import is_gateway_known_command
    return "/" + rest if is_gateway_known_command(cmd_name) else text
