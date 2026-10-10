"""Auxiliary defaults remain part of the canonical, copyable configuration tree."""

from copy import deepcopy

from hermes_cli.config_defaults import DEFAULT_CONFIG
from hermes_cli.config_defaults_auxiliary import AUXILIARY_DEFAULTS


def test_auxiliary_defaults_share_the_canonical_mapping():
    assert DEFAULT_CONFIG["auxiliary"] is AUXILIARY_DEFAULTS
    assert list(AUXILIARY_DEFAULTS)[-2:] == ["moa_reference", "moa_aggregator"]


def test_auxiliary_task_defaults_are_independently_copyable():
    task_blocks = [block for block in AUXILIARY_DEFAULTS.values()
                   if isinstance(block, dict) and "extra_body" in block]
    assert len({id(block["extra_body"]) for block in task_blocks}) == len(task_blocks)

    copied = deepcopy(DEFAULT_CONFIG)
    copied["auxiliary"]["vision"]["extra_body"]["test"] = True
    assert copied["auxiliary"] is not AUXILIARY_DEFAULTS
    assert "test" not in AUXILIARY_DEFAULTS["vision"]["extra_body"]
