"""Unknown keys in the top-level ``model:`` block are warned about, known ones are not (#92082)."""

import logging
import re
from pathlib import Path

import pytest

from hermes_cli import config as config_mod
from hermes_cli.config import _KNOWN_MODEL_KEYS, _warn_unknown_model_keys

_REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def _fresh_dedup(monkeypatch):
    monkeypatch.setattr(config_mod, "_MODEL_NORMALIZE_WARNED", set())


def _warnings(caplog):
    return [r.getMessage() for r in caplog.records if "model: unknown config keys" in r.getMessage()]


def test_bogus_key_warns_with_source(caplog):
    with caplog.at_level(logging.WARNING, logger=config_mod.logger.name):
        _warn_unknown_model_keys({"model": {"default": "m", "seed": 7}}, "/tmp/config.yaml")
    assert _warnings(caplog) == ["/tmp/config.yaml: model: unknown config keys ignored: seed"]


def test_every_known_key_is_quiet(caplog):
    block = {key: "x" for key in _KNOWN_MODEL_KEYS}
    with caplog.at_level(logging.WARNING, logger=config_mod.logger.name):
        _warn_unknown_model_keys({"model": block})
    assert _warnings(caplog) == []


@pytest.mark.parametrize("model_value", ["some/model-id", {"default": {"model": "m", "provider": "p"}}, None])
def test_shorthand_and_nested_default_are_quiet(caplog, model_value):
    with caplog.at_level(logging.WARNING, logger=config_mod.logger.name):
        _warn_unknown_model_keys({"model": model_value})
    assert _warnings(caplog) == []


def test_camel_case_key_suggests_snake_case(caplog):
    with caplog.at_level(logging.WARNING, logger=config_mod.logger.name):
        _warn_unknown_model_keys({"model": {"baseUrl": "http://x"}})
    assert _warnings(caplog) == ["config: model: unknown config keys ignored: baseUrl (did you mean 'base_url'?)"]


def test_warns_once_per_source_and_key_set(caplog):
    with caplog.at_level(logging.WARNING, logger=config_mod.logger.name):
        for _ in range(3):
            _warn_unknown_model_keys({"model": {"seed": 1}}, "a.yaml")
        _warn_unknown_model_keys({"model": {"seed": 1}}, "b.yaml")
    assert len(_warnings(caplog)) == 2


def test_load_config_warns_for_user_model_block(caplog):
    path = config_mod.get_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("model:\n  default: some/model\n  api_key: sk-test\n  seed: 7\n", encoding="utf-8")
    with caplog.at_level(logging.WARNING, logger=config_mod.logger.name):
        config_mod.load_config()
    assert _warnings(caplog) == [f"{path}: model: unknown config keys ignored: seed"]


# Unambiguous reads of the top-level block: ``<cfg>.get("model") or {}).get("k")`` and
# ``cfg_get(<cfg>, "model", "k")``. A key read this way but absent from the set would make a
# working config warn that the key is ignored.
_DIRECT_READ = re.compile(
    r'\.get\(\s*"model"\s*(?:,\s*\{\}\s*)?\)\s*(?:or\s*\{\}\s*)?\)\s*\.get\(\s*"(\w+)"'
    r'|cfg_get\([^,()]+,\s*"model",\s*"(\w+)"'
)


def test_direct_model_block_reads_are_known():
    read = set()
    for top in ("agent", "hermes_cli", "gateway", "tools"):
        for path in (_REPO_ROOT / top).rglob("*.py"):
            for match in _DIRECT_READ.finditer(path.read_text(encoding="utf-8", errors="ignore")):
                read.add(match.group(1) or match.group(2))
    assert read, "pattern matched nothing; the scan is broken"
    assert read <= _KNOWN_MODEL_KEYS, sorted(read - _KNOWN_MODEL_KEYS)
