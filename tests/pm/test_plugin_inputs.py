"""The worker receives the same plugin input the caller built."""
from __future__ import annotations

import json

import pytest

from pm.plugin_inputs import Candidates, Members, Selection, StagedUpdate, decode, encode


@pytest.mark.parametrize("build", [
    lambda root: None,
    lambda root: Members([]),
    lambda root: Members([root / "plugins/a", root / "plugins/b"]),
    lambda root: Members({root / "home/plugins/a": root / "staged/a"}),
    lambda root: Candidates([root / "plugins/candidate"]),
    lambda root: Selection({"home": str(root / "home"), "enabled": ["a"], "disabled": [],
                            "expected_config": "missing"}),
    lambda root: StagedUpdate({"target": str(root / "home/plugins/a"), "staged": str(root / "staged/a"),
                               "target_digest": None, "old_metadata": {},
                               "new_metadata": {"a": {"revision": "new"}}}),
])
def test_plugin_input_survives_the_worker_wire(build, tmp_path):
    plugins = build(tmp_path)
    assert decode(json.loads(json.dumps(encode(plugins)))) == plugins
