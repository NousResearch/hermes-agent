"""``hermes doctor`` names the user-written regular expressions the RE2 engine will refuse or read
differently, with what will happen to each setting; a refused one fails the run like the other upcoming-change notices."""

import json
from types import SimpleNamespace

import pytest

from hermes_constants import get_hermes_home

_CONFIG = """\
hooks:
  pre_tool_call:
    - matcher: "^(?!mcp_).*"
      command: /opt/guard.sh
      fail_closed: true
    - matcher: "terminal|patch"
      command: /opt/audit.sh
"""


def _doctor(monkeypatch, capsys):
    from hermes_cli import doctor
    from hermes_cli.doctor_regex import _check_user_regex_settings
    from hermes_cli.main import cmd_doctor

    monkeypatch.setattr(doctor, "DOCTOR_CHECKS", (("Regex Settings", _check_user_regex_settings),))
    status = cmd_doctor(SimpleNamespace(fix=False, ack=None, live=False))
    return status, capsys.readouterr().out


@pytest.fixture(autouse=True)
def _no_pattern_env(monkeypatch):
    for name in ("MATRIX_IGNORE_USER_PATTERNS", "TELEGRAM_MENTION_PATTERNS", "SLACK_MENTION_PATTERNS",
                 "WHATSAPP_MENTION_PATTERNS", "DINGTALK_MENTION_PATTERNS", "PHOTON_MENTION_PATTERNS",
                 "BLUEBUBBLES_MENTION_PATTERNS"):
        monkeypatch.delenv(name, raising=False)


def test_refused_and_inexact_patterns_are_named_with_their_consequence_and_refused_ones_fail_doctor(monkeypatch, capsys):
    home = get_hermes_home()
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(_CONFIG, encoding="utf-8")
    (home / "webhook_subscriptions.json").write_text(json.dumps({"deploys": {
        "secret": "s", "filters": [{"not": {"field": "payload.ref", "regex": r"(a)\1"}}]}}), encoding="utf-8")
    monkeypatch.setenv("MATRIX_IGNORE_USER_PATTERNS", r"^@(?!admin)[^:]+:example\.org$,^@whatsapp_")
    monkeypatch.setenv("TELEGRAM_MENTION_PATTERNS", json.dumps([r"caf\b.é", "hermes"]))

    status, out = _doctor(monkeypatch, capsys)

    assert status == 1  # a refused pattern is a manual issue, like the other upcoming-change notices
    assert "regex setting(s) the RE2 engine will refuse" in out
    assert "hooks.pre_tool_call matcher of /opt/guard.sh: '^(?!mcp_).*' is not a valid RE2 pattern" in out
    assert "every tool call blocked" in out
    assert "MATRIX_IGNORE_USER_PATTERNS: " in out and "the Matrix adapter will not start" in out
    assert "webhook subscription 'deploys' filter regex" in out and "requests get 404" in out
    assert "TELEGRAM_MENTION_PATTERNS: " in out and "may match differently" in out
    # patterns RE2 accepts unchanged are not listed
    assert "terminal|patch" not in out and "^@whatsapp_" not in out and "'hermes'" not in out


def test_a_profile_without_refused_patterns_gets_one_ok_line(monkeypatch, capsys):
    home = get_hermes_home()
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text("hooks:\n  pre_tool_call:\n    - matcher: terminal\n      command: /opt/audit.sh\n",
                                      encoding="utf-8")

    status, out = _doctor(monkeypatch, capsys)

    assert status == 0
    assert "user regex setting(s) work with the RE2 engine" in out
    assert "not a valid RE2 pattern" not in out


def test_an_env_pattern_is_split_like_its_adapter_splits_it(monkeypatch, capsys):
    # Telegram reads one pattern per line: the comma belongs to the pattern, which RE2 accepts.
    monkeypatch.setenv("TELEGRAM_MENTION_PATTERNS", "^(hey|hi),? hermes")

    status, out = _doctor(monkeypatch, capsys)

    assert status == 0
    assert "not a valid RE2 pattern" not in out


def test_an_outbound_webhook_is_named_by_its_index_never_by_any_part_of_its_url(monkeypatch, capsys):
    """A webhook URL carries its token (some services in the host name), and doctor output gets pasted into issues."""
    home = get_hermes_home()
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(
        "hooks:\n  outbound:\n    - url: https://hooks.example.com/services/T0/B0/tok3n\n"
        "      events: [post_tool_call]\n      matcher: \"x(?=y)z\"\n", encoding="utf-8")

    status, out = _doctor(monkeypatch, capsys)

    assert status == 1
    assert "hooks.outbound[0] matcher of unnamed target: 'x(?=y)z' is not a valid RE2 pattern" in out
    assert "tok3n" not in out and "hooks.example.com" not in out


@pytest.mark.parametrize("label, key, env, adapter_reads", [
    pytest.param("telegram mention patterns", "mention_patterns", "TELEGRAM_MENTION_PATTERNS", "env", id="telegram"),
    pytest.param("Matrix ignore_user_patterns", "ignore_user_patterns", "MATRIX_IGNORE_USER_PATTERNS", "env",
                 id="matrix"),
    pytest.param("slack mention patterns", "mention_patterns", "SLACK_MENTION_PATTERNS", "yaml", id="slack"),
    pytest.param("dingtalk mention patterns", "mention_patterns", "DINGTALK_MENTION_PATTERNS", "yaml", id="dingtalk"),
    pytest.param("bluebubbles mention patterns", "mention_patterns", "BLUEBUBBLES_MENTION_PATTERNS", "yaml",
                 id="bluebubbles"),
])
def test_a_setting_set_in_both_places_is_checked_where_its_adapter_reads_it(monkeypatch, label, key, env,
                                                                             adapter_reads):
    """The shadowed value never runs, so a refused pattern there must not fail doctor."""
    from hermes_cli.doctor_regex import SETTINGS, Inputs

    monkeypatch.setenv(env, "from-env(")
    platform = label.split()[0].lower()
    setting = next(s for s in SETTINGS if s.label == label)
    found = [pattern for _where, pattern in setting.read(Inputs(extras={platform: {key: ["from-yaml("]}}))]
    assert found == ["from-env(" if adapter_reads == "env" else "from-yaml("]


def test_check_is_skipped_without_the_engine(monkeypatch, capsys):
    import hermes_re

    def _no_engine(*_args, **_kwargs):
        raise ImportError("No module named 're2'")

    monkeypatch.setattr(hermes_re, "build", _no_engine)

    status, out = _doctor(monkeypatch, capsys)

    assert status == 0
    assert "skipped: google-re2 not installed" in out
