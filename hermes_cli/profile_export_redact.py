"""Secret redaction for staged profile-export files that must stay parseable.

The export scrub runs ``redact_sensitive_text`` over whole files. On JSON and YAML that text pass
can produce a file that no longer parses: an unquoted ``KEY_TOKEN=value`` at the end of a string
swallows the closing quote and comma, and a masked plain YAML scalar (``api_key: ***``) reads as
an alias. The importing profile then runs on default settings, and its cron store refuses to
load. A file the text pass breaks is redacted value by value instead, from its original parse.
"""

import json
from io import StringIO
from pathlib import Path
from typing import Any, Callable, Optional

_JSON_SUFFIXES = frozenset({".json"})
_JSONL_SUFFIXES = frozenset({".jsonl"})
_YAML_SUFFIXES = frozenset({".yaml", ".yml"})


def _redact(text: str) -> str:
    from agent.redact import redact_sensitive_text
    return redact_sensitive_text(text, force=True)


def _redact_leaf(key: Any, value: str) -> str:
    """Redact *value*, then run the pass again with it rendered under *key*.

    A value alone loses the name the text pass keys on (``"api_key": "…"``, ``password: …``,
    ``Authorization: …``), and without it an opaque secret would pass unmasked. A JSON pair the
    pass breaks again (the same overrun) is skipped: the value was already redacted on its own."""
    redacted = _redact(value)
    if not isinstance(key, str):
        return redacted
    json_pair = json.dumps({key: redacted}, ensure_ascii=False)
    masked = _redact(json_pair)
    if masked != json_pair:
        try:
            parsed = json.loads(masked)
        except ValueError:
            parsed = None
        if isinstance(parsed, dict) and len(parsed) == 1 and isinstance(next(iter(parsed.values())), str):
            redacted = next(iter(parsed.values()))
    yaml_pair = f"{key}: {redacted}"
    masked = _redact(yaml_pair)
    if masked != yaml_pair:
        prefix = f"{key}: "
        redacted = masked[len(prefix):] if masked.startswith(prefix) else "***"
    return redacted


def _redact_tree(node: Any) -> Any:
    """Redact every string in a parsed document in place (mapping keys and values, list items)."""
    if isinstance(node, dict):
        for k in list(node.keys()):
            child = node[k]
            new_child = _redact_leaf(k, child) if isinstance(child, str) else _redact_tree(child)
            new_key = _redact(k) if isinstance(k, str) else k
            if new_key != k:
                del node[k]
            node[new_key] = _same_scalar_style(child, new_child)
        return node
    if isinstance(node, list):
        for i, item in enumerate(node):
            node[i] = _same_scalar_style(item, _redact_tree(item))
        return node
    if isinstance(node, str):
        return _redact(node)
    return node


def _same_scalar_style(old: Any, new: Any) -> Any:
    """Keep a ruamel quoted/literal scalar's style when its text changes."""
    if isinstance(old, str) and isinstance(new, str) and new != old and type(old) is not str:
        try:
            return type(old)(new)
        except TypeError:
            return new
    return new


def _json_redacted(text: str) -> str:
    data = json.loads(text)
    indent = 2 if "\n" in text.strip() else None
    out = json.dumps(_redact_tree(data), ensure_ascii=False, indent=indent)
    return out + "\n" if text.endswith("\n") else out


def _jsonl_redacted(text: str) -> str:
    lines = []
    for line in text.splitlines(keepends=True):
        body = line.rstrip("\r\n")
        masked = _redact(body)
        if masked != body and _parses(json.loads, body) and not _parses(json.loads, masked):
            masked = json.dumps(_redact_tree(json.loads(body)), ensure_ascii=False)
        lines.append(masked + line[len(body):])
    return "".join(lines)


def _yaml_load(text: str) -> Any:
    import hermes_yaml
    return hermes_yaml.roundtrip_yaml().load(text)


def _yaml_redacted(text: str) -> str:
    import hermes_yaml
    yaml_rt = hermes_yaml.roundtrip_yaml()
    data = yaml_rt.load(text)
    out = StringIO()
    yaml_rt.dump(_redact_tree(data), out)
    return out.getvalue()


def _parses(loader: Callable[[str], Any], text: str) -> bool:
    try:
        loader(text)
    except Exception:  # json.JSONDecodeError / ruamel YAMLError and its many subclasses
        return False
    return True


def redact_export_text(path: Path, text: str) -> str:
    """Force-redacted *text* of export file *path*; JSON, JSONL and YAML files keep parsing."""
    redacted = _redact(text)
    if redacted == text:
        return text
    suffix = path.suffix.lower()
    if suffix in _JSONL_SUFFIXES:
        return _jsonl_redacted(text)
    rebuild: Optional[Callable[[str], str]] = None
    loader: Optional[Callable[[str], Any]] = None
    if suffix in _JSON_SUFFIXES:
        loader, rebuild = json.loads, _json_redacted
    elif suffix in _YAML_SUFFIXES:
        loader, rebuild = _yaml_load, _yaml_redacted
    if loader is None or _parses(loader, redacted) or not _parses(loader, text):
        return redacted
    return rebuild(text)
