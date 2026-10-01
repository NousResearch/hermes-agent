"""Secret references for file-authored webhook verification."""
import re
from agent.secret_scope import get_secret


def parse_secret_reference(value: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"env:[A-Za-z_][A-Za-z0-9_]*", value):
        raise ValueError("Use env:SECRET_NAME, never a literal secret value.")
    return value[4:]


def resolve_secret_reference(value: str) -> str:
    name = parse_secret_reference(value)
    secret = get_secret(name)
    if not secret:
        raise ValueError(f"Secret {name} is not configured for this profile.")
    return secret
