"""Tests for the #64934 data-layer diagnostic script.

The script lives in ``scripts/`` (not a package), so we load it via importlib
and exercise its pure detection functions — the same logic the CLI runs
against ``state.db.gateway_routing`` in production.
"""

import importlib.util
from pathlib import Path


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "check_cross_adapter_session_sew",
        Path(__file__).resolve().parents[2] / "scripts" / "check_cross_adapter_session_sew.py",
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestAdapterIdFromKey:
    def test_extracts_adapter_segment(self):
        mod = _load_script()
        assert mod.adapter_id_from_key(
            "agent:main:feishu:adapter=feishu%3AappA:group:oc_x:omt_y"
        ) == "feishu:appA"

    def test_namespace_agnostic(self):
        # A multi-profile key must still resolve — run.py's _parse_session_key
        # misses these (it hard-codes parts[1]=='main'), this script must not.
        mod = _load_script()
        assert mod.adapter_id_from_key(
            "agent:coder:feishu:adapter=feishu%3AappB:dm:oc_z"
        ) == "feishu:appB"

    def test_no_adapter_segment(self):
        mod = _load_script()
        assert mod.adapter_id_from_key("agent:main:feishu:group:oc_x") is None
        assert mod.adapter_id_from_key("") is None


class TestDetectSews:
    def test_flags_cross_adapter_sew(self):
        mod = _load_script()
        # The #64934 shape: two distinct adapters onto one session_id.
        sewn = [
            ("agent:main:feishu:adapter=feishu%3AappA:group:oc_c:omt_t", "sess_X", 1.0),
            ("agent:main:feishu:adapter=feishu%3AappB:group:oc_c:omt_t", "sess_X", 2.0),
        ]
        found = mod.detect_sews(sewn)
        assert len(found) == 1
        assert found[0]["session_id"] == "sess_X"
        assert sorted(found[0]["adapters"]) == ["feishu:appA", "feishu:appB"]
        assert found[0]["updated_at"] == 2.0

    def test_ignores_same_adapter_alias(self):
        mod = _load_script()
        # Same adapter, two keys on one session is a legitimate alias, not a sew.
        alias = [
            ("agent:main:feishu:adapter=feishu%3AappA:dm:oc_c", "sess_Y", 1.0),
            ("agent:main:feishu:adapter=feishu%3AappA:dm:oc_d", "sess_Y", 2.0),
        ]
        assert mod.detect_sews(alias) == []

    def test_clean_distinct_sessions(self):
        mod = _load_script()
        clean = [
            ("agent:main:feishu:adapter=feishu%3AappA:group:oc_c:omt_t", "sess_A", 1.0),
            ("agent:main:feishu:adapter=feishu%3AappB:group:oc_c:omt_t", "sess_B", 2.0),
        ]
        assert mod.detect_sews(clean) == []

    def test_skips_rows_without_adapter_or_sid(self):
        mod = _load_script()
        rows = [
            ("agent:main:feishu:group:oc_c", "sess_Z", 1.0),  # no adapter
            ("agent:main:feishu:adapter=feishu%3AappA:group:oc_c", None, 2.0),  # no sid
        ]
        assert mod.detect_sews(rows) == []

    def test_report_pass_and_fail(self):
        mod = _load_script()
        assert "PASS" in mod.report([], 5)
        fail_report = mod.report(
            [{"session_id": "s1", "adapters": ["feishu:appA", "feishu:appB"],
              "keys": ["k1", "k2"], "updated_at": None}],
            5,
        )
        assert "FAIL" in fail_report and "s1" in fail_report
