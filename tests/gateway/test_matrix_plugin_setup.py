"""Tests for the Matrix plugin's interactive_setup wizard home-channel flow.

The interactive_setup wizard lazy-imports its CLI helpers from
``hermes_cli.config`` (get_env_value / save_env_value / remove_env_value),
``hermes_cli.cli_output`` (prompt / prompt_yes_no / print_*), and
``tools.lazy_deps`` (mautrix ensure). We patch each at its source module so
the wizard runs without touching pip or the network. Covers the home-channel
clear-on-blank behavior added in the follow-up to PR #58421.
"""
import hermes_cli.config as config_mod
import hermes_cli.cli_output as cli_output_mod
import tools.lazy_deps as lazy_deps_mod
from plugins.platforms.matrix.adapter import interactive_setup


def _patch_setup_io(monkeypatch, prompts, yes_no_responses, saved, removed, existing):
    prompt_iter = iter(prompts)
    yes_no_iter = iter(yes_no_responses)
    monkeypatch.setattr(config_mod, "get_env_value", lambda key: existing.get(key, ""))
    monkeypatch.setattr(config_mod, "save_env_value", lambda k, v: saved.update({k: v}))

    def _remove(key):
        removed.append(key)
        return existing.pop(key, None) is not None

    monkeypatch.setattr(config_mod, "remove_env_value", _remove)
    monkeypatch.setattr(cli_output_mod, "prompt", lambda *_a, **_kw: next(prompt_iter))
    monkeypatch.setattr(
        cli_output_mod, "prompt_yes_no", lambda *_a, **_kw: next(yes_no_iter)
    )
    for name in ("print_header", "print_info", "print_success", "print_warning"):
        monkeypatch.setattr(cli_output_mod, name, lambda *_a, **_kw: None)
    # Block the auto-install path so the test never invokes pip.
    monkeypatch.setattr(lazy_deps_mod, "feature_missing", lambda feature: ())
    monkeypatch.setattr(lazy_deps_mod, "ensure", lambda *a, **kw: None)


# Matrix prompts (after the E2EE yes_no): allowed_users, home_channel.
# Homeserver, token are still text prompts before E2EE.
_PROMPTS_NONEMPTY = [
    "https://matrix.example.org",  # homeserver
    "syt_test_token_value",        # access token (password)
    "@bot:matrix.example.org",     # user_id (optional)
    "",                            # allowed_users
    "!AbCdEfGhIjKlMn:matrix.example.org",  # home room
]
_PROMPTS_BLANK = [
    "https://matrix.example.org",
    "syt_test_token_value",
    "@bot:matrix.example.org",
    "",
    "",
]
_PROMPTS_WHITESPACE = [
    "https://matrix.example.org",
    "syt_test_token_value",
    "@bot:matrix.example.org",
    "",
    "   ",
]
# E2EE? = False so we don't pull the [encryption] extras.
_YES_NO = [False]


class TestMatrixHomeChannelClear:
    """Blank home-room answer must clear MATRIX_HOME_ROOM (#12423)."""

    def test_blank_removes_existing_home_room(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        saved, removed = {}, []
        _patch_setup_io(
            monkeypatch,
            _PROMPTS_BLANK,
            _YES_NO,
            saved,
            removed,
            existing={"MATRIX_HOME_ROOM": "!oldRoomId:matrix.example.org"},
        )
        interactive_setup()
        assert "MATRIX_HOME_ROOM" in removed
        assert "MATRIX_HOME_ROOM" not in saved



class TestMatrixE2EEInstallRouting:
    """The E2EE answer must decide what gets installed (#62401).

    ``matrix_pkg = "mautrix[encryption]" if want_e2ee else "mautrix"`` used to
    be cosmetic — it fed print strings while the install call hardcoded
    ``platform.matrix``, whose spec carried the ``[encryption]`` extra. So
    answering "no" still triggered a python-olm build, which is fatal on macOS.
    """

    def _run(self, monkeypatch, *, want_e2ee, blocked_reason=""):
        import plugins.platforms.matrix.adapter as matrix_mod

        ensured, saved, removed = [], {}, []
        _patch_setup_io(
            monkeypatch, _PROMPTS_BLANK, [want_e2ee], saved, removed, existing={}
        )
        # Every feature reports work to do, so a skipped install can only mean
        # the wizard never asked for that feature.
        monkeypatch.setattr(lazy_deps_mod, "feature_missing", lambda f: ("pkg==1",))
        monkeypatch.setattr(
            lazy_deps_mod, "ensure",
            lambda feature, **kw: ensured.append(feature),
        )
        monkeypatch.setattr(
            matrix_mod, "_e2ee_unsupported_reason", lambda: blocked_reason
        )
        interactive_setup()
        return ensured, saved

    def test_declining_e2ee_never_installs_the_crypto_group(self, monkeypatch):
        ensured, saved = self._run(monkeypatch, want_e2ee=False)
        assert ensured == ["platform.matrix"], (
            "answering 'no' to E2EE must install the plaintext group only — "
            f"got {ensured}"
        )
        assert "MATRIX_ENCRYPTION" not in saved

    def test_accepting_e2ee_installs_both_groups(self, monkeypatch):
        ensured, saved = self._run(monkeypatch, want_e2ee=True)
        assert ensured == ["platform.matrix", "platform.matrix.e2ee"]
        assert saved.get("MATRIX_ENCRYPTION") == "true"

    def test_e2ee_not_offered_where_olm_cannot_build(self, monkeypatch):
        """On macOS/Windows the wizard must not walk the user into a doomed
        build. The queued answer is "yes", so if the E2EE question were still
        asked the wizard would take it and both assertions below would fail."""
        ensured, saved = self._run(
            monkeypatch, want_e2ee=True, blocked_reason="unsupported on macOS: ..."
        )
        assert ensured == ["platform.matrix"]
        assert "MATRIX_ENCRYPTION" not in saved
