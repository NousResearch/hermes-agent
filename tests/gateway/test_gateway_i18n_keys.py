"""Locale-key coverage for gateway strings routed through ``t()`` (#43355).

Guards the contract between code and catalogs:

1. every message key referenced from ``gateway/run.py`` exists in ``en.yaml``
   (the fallback catalog) and ``zh.yaml`` (the locale the issue is about);
2. ``str.format`` placeholders match between en and zh, so a translation can
   never break formatting at runtime;
3. every ``gateway.commands.desc`` entry corresponds to a real registry
   command (no orphans surviving a command rename/removal), and every
   registry command has a zh description;
4. ``gateway_help_lines()`` renders localized descriptions under zh and falls
   back to the registry's English description for commands without an entry.
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml

from hermes_cli.commands import COMMAND_REGISTRY, CommandDef, gateway_help_lines

REPO_ROOT = Path(__file__).resolve().parents[2]
LOCALES = REPO_ROOT / "locales"

# Keys introduced by the gateway i18n migration. If you remove one of these
# from the catalogs, remove its callsite in gateway/run.py first.
MESSAGE_KEYS = [
    "gateway.drain.queued_next_turn",
    "gateway.drain.busy_another_turn",
    "gateway.drain.busy_new_work",
    "gateway.status_action.restarting",
    "gateway.status_action.shutting_down",
    "gateway.unknown_command",
    "gateway.first_message_note",
    "gateway.reasoning.header",
    "gateway.reasoning.more_lines",
]

_PLACEHOLDER = re.compile(r"{(\w+)}")


def _flatten(node: object, prefix: str = "") -> dict[str, str]:
    out: dict[str, str] = {}
    if isinstance(node, dict):
        for key, value in node.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            out.update(_flatten(value, child))
    elif isinstance(node, str):
        out[prefix] = node
    return out


def _catalog(lang: str) -> dict[str, str]:
    with open(LOCALES / f"{lang}.yaml", encoding="utf-8") as fh:
        return _flatten(yaml.safe_load(fh))


def test_message_keys_exist_in_en_and_zh() -> None:
    catalogs = {"en": _catalog("en"), "zh": _catalog("zh")}
    missing = [(lang, key) for lang, cat in catalogs.items() for key in MESSAGE_KEYS if key not in cat]
    assert not missing, f"locale keys missing from catalogs: {missing}"


def test_message_key_placeholders_match_between_en_and_zh() -> None:
    en, zh = _catalog("en"), _catalog("zh")
    mismatched = [
        key for key in MESSAGE_KEYS if set(_PLACEHOLDER.findall(en[key])) != set(_PLACEHOLDER.findall(zh[key]))
    ]
    assert not mismatched, f"format placeholders differ between en and zh: {mismatched}"


def test_zh_command_descriptions_match_registry() -> None:
    with open(LOCALES / "zh.yaml", encoding="utf-8") as fh:
        desc = yaml.safe_load(fh)["gateway"]["commands"]["desc"]
    registry_names = {cmd.name for cmd in COMMAND_REGISTRY}

    orphans = sorted(set(desc) - registry_names)
    assert not orphans, f"gateway.commands.desc keys without a registry command: {orphans}"

    untranslated = sorted(registry_names - set(desc))
    assert not untranslated, (
        "registry commands missing a zh description -- add gateway.commands.desc "
        f"entries to locales/zh.yaml (or drop this check if intentional): {untranslated}"
    )


def test_gateway_help_lines_localizes_and_falls_back(monkeypatch) -> None:
    monkeypatch.setenv("HERMES_LANGUAGE", "zh")
    sentinel = CommandDef("zz-i18n-probe", "Probe-only description", "Session")
    COMMAND_REGISTRY.append(sentinel)
    try:
        rendered = "\n".join(gateway_help_lines())
    finally:
        COMMAND_REGISTRY.remove(sentinel)

    # `/new` has a zh entry -> localized description.
    assert "开始新会话" in rendered
    # The probe command has no zh entry -> registry English fallback.
    assert "Probe-only description" in rendered


def test_gateway_help_lines_english_output_unchanged(monkeypatch) -> None:
    monkeypatch.setenv("HERMES_LANGUAGE", "en")
    rendered = "\n".join(gateway_help_lines())
    assert "Start a new session (fresh session ID + history)" in rendered
