from pathlib import Path

from agent.system_prompt import _load_yesterday_context_pack
from hermes_constants import reset_hermes_home_override, set_hermes_home_override


def test_yesterday_context_pack_absent_returns_empty(tmp_path):
    token = set_hermes_home_override(tmp_path)
    try:
        assert _load_yesterday_context_pack() == ""
    finally:
        reset_hermes_home_override(token)


def test_yesterday_context_pack_loads_profile_local_file(tmp_path):
    pack = tmp_path / "company_context" / "_shared" / "yesterday_context_pack.md"
    pack.parent.mkdir(parents=True)
    pack.write_text("# Yesterday\n\n- Durable note\n", encoding="utf-8")
    token = set_hermes_home_override(tmp_path)
    try:
        out = _load_yesterday_context_pack()
    finally:
        reset_hermes_home_override(token)
    assert "YESTERDAY CONTEXT PACK" in out
    assert "Durable note" in out


def test_yesterday_context_pack_truncates_with_footer(tmp_path):
    pack = tmp_path / "company_context" / "_shared" / "yesterday_context_pack.md"
    pack.parent.mkdir(parents=True)
    pack.write_text("x" * 200, encoding="utf-8")
    token = set_hermes_home_override(tmp_path)
    try:
        out = _load_yesterday_context_pack(max_chars=80)
    finally:
        reset_hermes_home_override(token)
    assert len(out) > 80  # includes fixed heading plus capped body/footer
    assert "TRUNCATED" in out
    assert str(pack) in out
