"""Setup-wizard E2EE install path (issue #31116).

Pre-fix: when the user enabled E2EE, the wizard ran
``pip install 'mautrix[encryption]'``. That extra only pulls
``python-olm`` / ``pycryptodome`` / ``unpaddedbase64`` — *not* the two
packages the gateway actually imports at runtime when an encrypted
client is created:

* ``asyncpg``  — ``mautrix.crypto.store.asyncpg.store`` does
  ``from asyncpg import UniqueViolationError`` at module import time,
  so even SQLite-backed crypto stores need it on the import path.
* ``aiosqlite`` — ``mautrix.util.async_db.Database.create("sqlite:///…")``
  refuses to start without it ("Unknown database scheme sqlite").

The result was the gateway crashing at first connect with
``No module named 'asyncpg'`` despite the wizard reporting
``mautrix[encryption] installed``.

Fix: route the install through ``tools.lazy_deps.ensure("platform.matrix")``,
which mirrors the ``[matrix]`` extra in ``pyproject.toml`` and pulls
the complete dep set.
"""

from __future__ import annotations

import sys
from unittest.mock import patch


# ---------------------------------------------------------------------------
# Lazy-deps allowlist must include the two packages that were missing.
# ---------------------------------------------------------------------------


class TestPlatformMatrixLazyDepsAreComplete:
    """Without this, the wizard fix is meaningless — ``ensure`` would
    install whatever is in the allowlist and we'd still be missing the
    two packages."""

    def test_asyncpg_in_platform_matrix(self):
        from tools.lazy_deps import LAZY_DEPS

        specs = LAZY_DEPS["platform.matrix"]
        names = [s.split("[")[0].split("=")[0].split(">")[0].split("<")[0]
                 for s in specs]
        assert "asyncpg" in names, (
            f"platform.matrix lazy-deps must include asyncpg "
            f"(mautrix.crypto.store.asyncpg imports it at module load). "
            f"Got: {specs}"
        )

    def test_aiosqlite_in_platform_matrix(self):
        from tools.lazy_deps import LAZY_DEPS

        specs = LAZY_DEPS["platform.matrix"]
        names = [s.split("[")[0].split("=")[0].split(">")[0].split("<")[0]
                 for s in specs]
        assert "aiosqlite" in names, (
            f"platform.matrix lazy-deps must include aiosqlite "
            f"(Database.create('sqlite://...') refuses to start without it). "
            f"Got: {specs}"
        )

    def test_mautrix_encryption_in_platform_matrix(self):
        """The original install kept mautrix[encryption] — sanity check."""
        from tools.lazy_deps import LAZY_DEPS

        specs = LAZY_DEPS["platform.matrix"]
        assert any("mautrix" in s and "encryption" in s for s in specs), (
            f"platform.matrix must still include mautrix[encryption]. "
            f"Got: {specs}"
        )


# ---------------------------------------------------------------------------
# Wizard install path: when E2EE is enabled, lazy_deps.ensure must be the
# install vector (not a one-off pip subprocess that misses asyncpg/aiosqlite).
# ---------------------------------------------------------------------------


class TestMatrixSetupWithE2eeUsesLazyEnsure:
    """The setup wizard's Matrix branch is the fix surface for the user."""

    def _make_prompt_stack(self, answers):
        """Returns a callable that pops answers in order."""
        queue = list(answers)

        def _prompt(label, *args, **kwargs):
            if not queue:
                return ""
            return queue.pop(0)

        return _prompt

    def test_e2ee_enabled_calls_lazy_ensure_for_platform_matrix(
        self, monkeypatch, tmp_path
    ):
        """The full path: user picks Matrix, enters a token, says yes to
        E2EE → wizard must call ``tools.lazy_deps.ensure('platform.matrix')``
        rather than its old custom subprocess that only installed
        ``mautrix[encryption]``."""

        # Hermes home isolation so save_env_value writes don't pollute.
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))

        # Force a non-existing existing-token path so the wizard runs to the
        # E2EE branch.
        monkeypatch.setattr(
            "hermes_cli.setup.get_env_value",
            lambda key, default=None: None,
        )

        # Capture writes without touching the real .env.
        saved = {}
        monkeypatch.setattr(
            "hermes_cli.setup.save_env_value",
            lambda key, value: saved.setdefault(key, value) or saved.update({key: value}),
        )

        # Stub the prompts: homeserver, access-token, user-id (auto-detect),
        # then E2EE yes, allowlist empty, home-room empty.
        prompts = self._make_prompt_stack([
            "https://matrix.example.org",
            "fake-access-token",
            "@bot:example.org",
            "",  # allowlist
            "",  # home room
        ])
        monkeypatch.setattr("hermes_cli.setup.prompt", prompts)

        # E2EE = yes; reconfigure-Matrix never asked because no existing
        # config (get_env_value returns None above).
        yn_queue = [True]

        def _yn(*args, **kwargs):
            return yn_queue.pop(0) if yn_queue else False

        monkeypatch.setattr("hermes_cli.setup.prompt_yes_no", _yn)

        # No-op all the cosmetic CLI helpers so the test stays quiet.
        for fn in (
            "print_header",
            "print_info",
            "print_success",
            "print_warning",
        ):
            monkeypatch.setattr(f"hermes_cli.setup.{fn}", lambda *a, **k: None)

        # Stub the lazy-deps ensure call so the test doesn't actually run pip.
        # Capture its invocations.
        ensure_calls = []

        def _fake_ensure(feature, *, prompt=True):
            ensure_calls.append((feature, prompt))

        # The wizard imports lazy_deps lazily inside the matrix branch.
        # Inject our stub via a real module entry so the local import
        # (`from tools.lazy_deps import ensure as ...`) picks it up.
        import tools.lazy_deps as _ld

        monkeypatch.setattr(_ld, "ensure", _fake_ensure)

        from hermes_cli.setup import _setup_matrix

        _setup_matrix()

        assert ensure_calls, (
            "Wizard did not call tools.lazy_deps.ensure(...) when E2EE was "
            "enabled. With this regression the install only pulls "
            "mautrix[encryption] and the runtime still crashes with "
            "'No module named asyncpg'. Issue #31116."
        )
        feature, prompt_flag = ensure_calls[0]
        assert feature == "platform.matrix", (
            f"Wizard called lazy_deps.ensure({feature!r}), expected "
            f"'platform.matrix' so asyncpg + aiosqlite come along."
        )
        assert prompt_flag is False, (
            "Wizard ran lazy install with prompt=True; this would block on "
            "an extra confirmation in the middle of the setup flow."
        )

        # MATRIX_ENCRYPTION must be persisted regardless of install outcome.
        assert saved.get("MATRIX_ENCRYPTION") == "true"

    def test_e2ee_install_failure_is_surfaced_not_silent(
        self, monkeypatch, tmp_path
    ):
        """A FeatureUnavailable from lazy_deps must reach the user with a
        manual-install hint — not be silently swallowed (which is what
        led to the original ticket: install reported success but Matrix
        was still broken)."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setattr(
            "hermes_cli.setup.get_env_value",
            lambda key, default=None: None,
        )
        saved = {}
        monkeypatch.setattr(
            "hermes_cli.setup.save_env_value",
            lambda key, value: saved.update({key: value}),
        )
        prompts = self._make_prompt_stack([
            "https://matrix.example.org",
            "fake-access-token",
            "@bot:example.org",
            "",
            "",
        ])
        monkeypatch.setattr("hermes_cli.setup.prompt", prompts)

        yn_queue = [True]
        monkeypatch.setattr(
            "hermes_cli.setup.prompt_yes_no",
            lambda *a, **k: yn_queue.pop(0) if yn_queue else False,
        )

        # Capture warnings emitted by the wizard.
        warnings = []
        monkeypatch.setattr(
            "hermes_cli.setup.print_warning",
            lambda msg: warnings.append(msg),
        )
        for fn in ("print_header", "print_info", "print_success"):
            monkeypatch.setattr(f"hermes_cli.setup.{fn}", lambda *a, **k: None)

        import tools.lazy_deps as _ld

        def _fake_ensure_fail(feature, *, prompt=True):
            raise _ld.FeatureUnavailable(
                feature,
                ("asyncpg==0.31.0",),
                "pip install failed: simulated PyPI 503",
            )

        monkeypatch.setattr(_ld, "ensure", _fake_ensure_fail)

        from hermes_cli.setup import _setup_matrix

        _setup_matrix()

        assert any("hermes-agent[matrix]" in w for w in warnings), (
            f"Install-failure path must print a manual-install hint. "
            f"Got warnings: {warnings}"
        )
