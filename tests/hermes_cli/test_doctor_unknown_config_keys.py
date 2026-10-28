"""`hermes doctor` warns on config.yaml keys the runtime never reads (#91876, salvage of #94860).

The walk asks ``hermes config set``'s validator about every on-disk path, so doctor and the
write path share one schema. Two invariants: a typo is reported once with a did-you-mean and an
unknown section is not descended into; every key in the documented configuration reference
validates (zero false positives on the documented surface — a new documented key must be seeded
in DEFAULT_CONFIG or registered in ``hermes_cli.config._RUNTIME_READ_CONFIG_KEYS``).
"""

import re
from pathlib import Path

import hermes_yaml as yaml

from hermes_cli.doctor_config import collect_unknown_config_keys

REPO = Path(__file__).resolve().parents[2]
DOC_CORPUS = (REPO / "website/docs/user-guide/configuration.md", REPO / "cli-config.yaml.example")


def _documented_config() -> dict:
    """Deep-merge of every YAML block in the configuration docs plus the shipped example file."""
    merged: dict = {}

    def deep(dst: dict, src: dict) -> None:
        for k, v in src.items():
            if isinstance(v, dict) and isinstance(dst.get(k), dict):
                deep(dst[k], v)
            else:
                dst[k] = v

    for path in DOC_CORPUS:
        text = path.read_text(encoding="utf-8-sig")
        blocks = re.findall(r"```ya?ml\n(.*?)```", text, re.S) if path.suffix == ".md" else [text]
        for block in blocks:
            try:
                data = yaml.safe_load(block)
            except Exception:  # a prose block that is not a whole YAML document
                continue
            if isinstance(data, dict):
                deep(merged, data)
    return merged


def test_typos_reported_once_with_suggestion_and_unknown_sections_not_descended():
    raw = {
        "modle": {"default": "x", "provider": "y"},   # typo'd section: one finding, no children
        "display": {"compact": True, "tool_progress_comand": False},  # typo'd leaf in a known section
        "compression": {"model_thresholds": {"gpt-9": 1}},  # open mapping: user-chosen keys accepted
        "_internal": {"anything": 1},                  # intentionally non-schema
    }
    findings = collect_unknown_config_keys(raw)
    assert dict(findings) == {"modle": "model", "display.tool_progress_comand": "display.tool_progress_command"}
    assert collect_unknown_config_keys(None) == [] and collect_unknown_config_keys({}) == []


def test_every_documented_config_key_validates():
    documented = _documented_config()
    assert len(documented) > 30, "docs corpus did not parse — the ratchet would be vacuous"
    assert collect_unknown_config_keys(documented) == []
