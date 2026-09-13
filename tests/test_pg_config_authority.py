"""Backend selection must fail closed on a structurally unusable config.

Review finding at head `cac143fa40`:

    Normalizing a well-formed non-mapping YAML root to `{}` destroys the
    distinction between an empty configuration and an existing but
    structurally unusable selection source. A scalar or list top level cannot
    safely authorize a fallback to SQLite; operator intent is unknown in
    exactly the same way as an unparseable file.

The rule these tests pin:

    absent / empty / whitespace / comments-only  -> genuinely no selection
    list root / scalar root / unparseable        -> intent UNKNOWN, fail closed

The consequence of getting it wrong is not a bad error message: it silently
routes the process to SQLite while the operator may have selected PostgreSQL,
splitting session history across two physical stores.
"""

from __future__ import annotations

import pytest

# Bodies that legitimately express "the operator configured nothing".
NO_SELECTION = [
    ("empty file", ""),
    ("whitespace only", "   \n\n  \n"),
    ("comments only", "# nothing here\n# really\n"),
]

# Bodies where the operator wrote something whose intent cannot be determined.
UNUSABLE = [
    ("list root", "- a\n- b\n"),
    ("scalar root", "just-a-string\n"),
    ("int root", "42\n"),
    ("unparseable", "sessions:\n  state_backend: [unclosed\n"),
]


class TestStrictAuthorityReader:
    """The config layer owns the strict reader; these are its semantics."""

    def _write(self, tmp_path, body, name="config.yaml"):
        p = tmp_path / name
        p.write_text(body, encoding="utf-8")
        return p

    def test_absent_file_returns_none(self, tmp_path):
        from hermes_cli.config_readers import read_user_config_for_authority

        assert read_user_config_for_authority(tmp_path / "nope.yaml") is None

    @pytest.mark.parametrize("label,body", NO_SELECTION)
    def test_empty_documents_return_none(self, tmp_path, label, body):
        from hermes_cli.config_readers import read_user_config_for_authority

        assert read_user_config_for_authority(self._write(tmp_path, body)) is None

    @pytest.mark.parametrize("label,body", UNUSABLE)
    def test_unusable_documents_raise(self, tmp_path, label, body):
        from hermes_cli.config_readers import read_user_config_for_authority

        with pytest.raises(Exception):
            read_user_config_for_authority(self._write(tmp_path, body))

    def test_mapping_root_is_returned_verbatim(self, tmp_path):
        from hermes_cli.config_readers import read_user_config_for_authority

        got = read_user_config_for_authority(
            self._write(tmp_path, "sessions:\n  state_backend: postgres\n")
        )
        assert got == {"sessions": {"state_backend": "postgres"}}, (
            "the strict reader must not merge defaults or expand env vars"
        )

    def test_none_is_distinguishable_from_empty_mapping(self, tmp_path):
        """The whole point: `None` and `{}` must not be interchangeable.

        A reader that returns `{}` for an empty file cannot tell it apart from
        one that returned `{}` for a list root — which is the bug.
        """
        from hermes_cli.config_readers import read_user_config_for_authority

        assert read_user_config_for_authority(self._write(tmp_path, "")) is None
        got = read_user_config_for_authority(self._write(tmp_path, "a: 1\n"))
        assert got == {"a": 1}

    def test_existing_raw_reader_contract_is_unchanged(self, tmp_path):
        """The additive design must not disturb `read_user_config_raw`.

        Its existing callers (write-back round-trips, diagnostics, env
        bridges) depend on the normalize-to-`{}` behaviour.
        """
        from hermes_cli.config import read_user_config_raw

        assert read_user_config_raw(self._write(tmp_path, "")) == {}
        assert read_user_config_raw(self._write(tmp_path, "- a\n- b\n")) == {}
        assert read_user_config_raw(tmp_path / "absent.yaml") == {}


class TestActiveProfileSelectionMatrix:
    """`resolve_postgres_dsn()` — the active process's own selection source."""

    def _home(self, tmp_path, body, tag):
        home = tmp_path / f"home_{tag}"
        home.mkdir()
        if body is not None:
            (home / "config.yaml").write_text(body, encoding="utf-8")
        return home

    def _clear(self, monkeypatch):
        for k in ("HERMES_STATE_BACKEND", "HERMES_STATE_DATABASE_URL",
                  "HERMES_STATE_POSTGRES_DSN"):
            monkeypatch.delenv(k, raising=False)

    @pytest.mark.parametrize("label,body", NO_SELECTION)
    def test_no_selection_bodies_resolve_quietly(
        self, tmp_path, monkeypatch, label, body
    ):
        import hermes_state_postgres as hsp

        home = self._home(tmp_path, body, label.replace(" ", "_"))
        monkeypatch.setenv("HERMES_HOME", str(home))
        self._clear(monkeypatch)

        assert hsp.resolve_postgres_dsn() is None

    def test_absent_config_resolves_quietly(self, tmp_path, monkeypatch):
        """The default install may have no config.yaml at all."""
        import hermes_state_postgres as hsp

        home = self._home(tmp_path, None, "absent")
        monkeypatch.setenv("HERMES_HOME", str(home))
        self._clear(monkeypatch)

        assert hsp.resolve_postgres_dsn() is None

    def test_mapping_without_sessions_resolves_quietly(self, tmp_path, monkeypatch):
        import hermes_state_postgres as hsp

        home = self._home(tmp_path, "model:\n  default: x\n", "nosessions")
        monkeypatch.setenv("HERMES_HOME", str(home))
        self._clear(monkeypatch)

        assert hsp.resolve_postgres_dsn() is None

    @pytest.mark.parametrize("label,body", UNUSABLE)
    def test_unusable_bodies_fail_closed(
        self, tmp_path, monkeypatch, label, body
    ):
        import hermes_state_postgres as hsp

        home = self._home(tmp_path, body, label.replace(" ", "_"))
        monkeypatch.setattr(hsp, "get_hermes_home", lambda h=home: h,
                            raising=False)
        monkeypatch.setenv("HERMES_HOME", str(home))
        self._clear(monkeypatch)

        with pytest.raises(RuntimeError):
            hsp.resolve_postgres_dsn()


class TestNamedProfileSelectionMatrix:
    """`profile_selects_postgres()` — a cross-profile reader's decision."""

    def _profile(self, tmp_path, monkeypatch, body, tag):
        from hermes_cli import profiles as profiles_mod

        pdir = tmp_path / "profiles" / f"p_{tag}"
        pdir.mkdir(parents=True)
        if body is not None:
            (pdir / "config.yaml").write_text(body, encoding="utf-8")
        monkeypatch.setattr(profiles_mod, "normalize_profile_name", lambda n: n)
        monkeypatch.setattr(profiles_mod, "profile_exists", lambda n: True)
        monkeypatch.setattr(profiles_mod, "get_profile_dir",
                            lambda n, d=str(pdir): d)
        return f"p_{tag}"

    @pytest.mark.parametrize("label,body", NO_SELECTION)
    def test_no_selection_bodies_return_false(
        self, tmp_path, monkeypatch, label, body
    ):
        from hermes_state_postgres import profile_selects_postgres

        name = self._profile(tmp_path, monkeypatch, body,
                             label.replace(" ", "_"))
        assert profile_selects_postgres(name) is False

    def test_absent_config_returns_false(self, tmp_path, monkeypatch):
        from hermes_state_postgres import profile_selects_postgres

        name = self._profile(tmp_path, monkeypatch, None, "absent")
        assert profile_selects_postgres(name) is False

    @pytest.mark.parametrize("label,body", UNUSABLE)
    def test_unusable_bodies_fail_closed(
        self, tmp_path, monkeypatch, label, body
    ):
        from hermes_state_postgres import profile_selects_postgres

        name = self._profile(tmp_path, monkeypatch, body,
                             label.replace(" ", "_"))
        with pytest.raises(RuntimeError):
            profile_selects_postgres(name)

    def test_valid_postgres_selection_is_honoured(self, tmp_path, monkeypatch):
        """Don't over-correct: a valid selection must still be read."""
        from hermes_state_postgres import profile_selects_postgres

        name = self._profile(
            tmp_path, monkeypatch,
            "sessions:\n  state_backend: postgres\n", "valid",
        )
        assert profile_selects_postgres(name) is True
