"""Filename-only policy for keeping likely credential attachments out of prompts."""
import os
import re

_DISCORD_PATH_ONLY_TEXT_BASENAMES = frozenset({
    '.env', '.envrc', '.git-credentials', '.netrc', '.npmrc', '.pypirc',
    'id_dsa', 'id_ecdsa', 'id_ed25519', 'id_rsa',
})
_DISCORD_PATH_ONLY_TEXT_EXTENSIONS = frozenset({
    '.jks', '.kdbx', '.key', '.keystore', '.p12', '.pem', '.pfx', '.ppk',
})
_DISCORD_PATH_ONLY_TEXT_TOKENS = frozenset({
    'apikey', 'auth', 'credential', 'credentials', 'key', 'oauth', 'passwd',
    'password', 'passwords', 'privatekey', 'secret', 'secrets', 'token', 'tokens',
})


def _discord_attachment_name_requires_path_only(filename: str | None) -> bool:
    """Whether filename metadata strongly indicates credentials; contents are not inspected."""
    name = str(filename or '').replace('\\', '/').rsplit('/', 1)[-1].strip().lower()
    if not name:
        return False
    if (
        name in _DISCORD_PATH_ONLY_TEXT_BASENAMES
        or name.startswith('.env.')
        or os.path.splitext(name)[1] in _DISCORD_PATH_ONLY_TEXT_EXTENSIONS
    ):
        return True
    tokens = {part for part in re.split(r'[^a-z0-9]+', name) if part}
    return bool(tokens & _DISCORD_PATH_ONLY_TEXT_TOKENS)
