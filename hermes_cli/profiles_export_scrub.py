"""What a profile export scrubs from its staged copy, and the saved-gateways stores profiles never carry.

Split out of :mod:`hermes_cli.profiles` (its facade over the clone/export paths).
"""
from __future__ import annotations

import json
import re
from io import StringIO
from pathlib import Path
from typing import Any, Callable, Optional

# The machine's saved-gateways list and its OAuth sign-ins (``hermes_cli.gateway_connections``), with
# their ``.lock`` dir and ``.corrupt-<ms>`` sidecars: machine-wide root state every profile reads at the
# root, holding plaintext session tokens. A copy in a clone is dead weight, and an export must not ship it.
GATEWAY_CONNECTION_STORES = ("gateway-connections.json", "gateway-connection-tokens.json")
GATEWAY_CONNECTION_STORE_RE = re.compile(r"gateway-connections?(?:-tokens)?\.json(?:\..+)?")


# Text/config suffixes secret-scrubbed on export; binary DBs, images etc. are left alone.
_EXPORT_REDACT_SUFFIXES = frozenset({
    ".md", ".txt", ".yaml", ".yml", ".json", ".jsonl", ".toml", ".ini", ".cfg", ".conf", ".py", ".sh",
    ".bash", ".zsh", ".js", ".ts", ".tsx", ".jsx", ".css", ".html", ".xml", ".csv",
})
# ``Path(".cursorrules").suffix`` is "" — name-match; ``*.env.example`` uses endswith; a hand-named
# config copy (``config.yaml.bak-my-note``) holds config.yaml's secrets under a suffix of its own.
_EXPORT_REDACT_NAMES = frozenset({".cursorrules"})


def _should_redact_export_file(path: Path) -> bool:
    name = path.name
    return (
        name in _EXPORT_REDACT_NAMES
        or name.lower().endswith(".env.example")
        or name.lower().startswith("config.yaml.")
        or path.suffix.lower() in _EXPORT_REDACT_SUFFIXES
    )


def _masked_envelopes(node):
    """*node* with the value of every ``{"encoding", "value"}`` secret envelope masked."""
    if isinstance(node, list):
        return [_masked_envelopes(item) for item in node]
    if not isinstance(node, dict):
        return node
    if set(node) == {"encoding", "value"} and isinstance(node["value"], str) and node["value"]:
        return {"encoding": node["encoding"], "value": "***"}
    return {key: _masked_envelopes(value) for key, value in node.items()}


def _mask_secret_envelopes(text: str) -> str:
    """Hermes stores secrets as ``{"encoding": "plain", "value": ...}`` envelopes, whose key names
    no secret: the text redactor cannot see them, so JSON documents are masked structurally."""
    try:
        doc = json.loads(text)
    except ValueError:
        return text
    masked = _masked_envelopes(doc)
    return text if masked == doc else json.dumps(masked, indent=2)


# The text pass alone can leave JSON/YAML that no longer parses: an unquoted ``KEY_TOKEN=value`` at the
# end of a string swallows the closing quote and comma, and a masked plain YAML scalar (``api_key: ***``)
# reads as an alias, so the imported profile runs on defaults and its cron store refuses to load. A file
# the text pass breaks is redacted value by value instead, from its original parse.
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
    _redact_comments(data)
    out = StringIO()
    yaml_rt.dump(_redact_tree(data), out)
    return out.getvalue()


def _redact_comments(node: Any) -> None:
    """Redact the comments a round-trip dump writes back: the value pass never sees them, and a
    commented-out ``# api_key: sk-…`` would otherwise ship verbatim."""
    from ruamel.yaml.tokens import CommentToken

    def tokens(slot: Any):
        if isinstance(slot, CommentToken):
            yield slot
        elif isinstance(slot, (list, tuple)):
            for item in slot:
                yield from tokens(item)

    ca = getattr(node, "ca", None)
    if ca is not None:
        for token in tokens([ca.comment, list(ca.items.values()), getattr(ca, "end", None)]):
            token.value = _redact(token.value)
    children = node.values() if isinstance(node, dict) else node if isinstance(node, list) else ()
    for child in children:
        _redact_comments(child)


def _parses(loader: Callable[[str], Any], text: str) -> bool:
    from hermes_yaml import YAMLError
    try:
        loader(text)
    except (ValueError, YAMLError):  # json.JSONDecodeError is a ValueError
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


def scrub_export_secrets(staged: Path) -> None:
    """Force-redact secret-shaped strings in a staged export tree (same pass as ``hermes
    sessions export --redact``). Runs on the staged copy only; symlinks to text files are
    materialized when content changes so redaction never follows a link back into the source."""
    for path in staged.rglob("*"):
        try:
            is_link = path.is_symlink()
            if not path.is_file():  # broken links, symlinked dirs, non-files
                continue
        except OSError:
            continue
        if not _should_redact_export_file(path):
            continue
        try:
            text = path.read_text(encoding="utf-8-sig")
        except (UnicodeDecodeError, OSError):
            continue
        redacted = redact_export_text(path, _mask_secret_envelopes(text))
        if redacted == text:
            continue
        if is_link:
            path.unlink()
        path.write_text(redacted, encoding="utf-8")
