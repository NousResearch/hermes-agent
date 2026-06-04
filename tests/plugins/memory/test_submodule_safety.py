"""Regression for #38674: _load_provider_from_dir must not blindly
execute every *.py file in a user-installed memory provider dir, and
must re-raise SystemExit/KeyboardInterrupt (BaseException) instead of
swallowing them under `except Exception`.

Concrete failure mode the original code had:
- Provider dir contains a `setup.py` (a user put one next to the plugin).
- `provider_dir.glob("*.py")` picks it up and `exec_module`s it.
- `setup.py` calls `setuptools.setup()` which interprets `sys.argv` and
  calls `sys.exit(1)` on bad subcommand.
- `sys.exit()` raises `SystemExit`, which is `BaseException` — not
  `Exception` — so the bare `except Exception` does NOT catch it.
- `SystemExit` propagates and Hermes crashes.

These tests pin both halves of the fix: (a) skip the obvious non-submodule
files, and (b) re-raise `SystemExit`/`KeyboardInterrupt`.
"""

import sys
import textwrap

import pytest

from plugins.memory import _load_provider_from_dir
from agent.memory_provider import MemoryProvider


def _write(p, content: str) -> None:
    p.write_text(textwrap.dedent(content).lstrip("\n"))


@pytest.fixture
def fake_provider_dir(tmp_path, monkeypatch):
    """Build a minimal but well-formed memory provider dir under tmp_path.

    Layout mimics a real provider (e.g. ``plugins/memory/honcho/``) plus
    the packaging/test artifacts that triggered the original crash.
    """
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes_home"))
    (tmp_path / "hermes_home").mkdir()

    provider = tmp_path / "fake_provider"
    provider.mkdir()

    # A real __init__.py that exposes a MemoryProvider subclass.
    # Implements every abstract method so the loader's
    # `find subclass & instantiate` fallback can construct it.
    _write(provider / "__init__.py", """
        from agent.memory_provider import MemoryProvider

        class FakeProvider(MemoryProvider):
            @property
            def name(self):
                return "fake"

            def is_available(self):
                return True

            def initialize(self, session_id, **kwargs):
                pass

            def get_tool_schemas(self):
                return []
    """)

    # A real submodule the plugin would import via "from .store import ..."
    _write(provider / "store.py", """
        SENTINEL_LOADED = True
    """)

    # Packaging/test artifacts that must NOT be executed.
    # Each sets a sentinel when imported so the test can assert it stays None.
    _write(provider / "setup.py", """
        SETUP_SENTINEL = "imported"
    """)
    _write(provider / "conftest.py", """
        CONFTEST_SENTINEL = "imported"
    """)
    _write(provider / "test_store.py", """
        TEST_SENTINEL = "imported"
    """)
    _write(provider / "store_test.py", """
        TEST_TRAILING_SENTINEL = "imported"
    """)
    _write(provider / "pyproject.py", """
        PYPROJECT_SENTINEL = "imported"
    """)

    return provider


def _provider_namespace_modules(provider_name: str) -> list:
    """Return sys.modules keys belonging to the user-memory namespace for
    this provider. User-installed providers live under
    ``_hermes_user_memory.<name>`` (per the loader's branch on _is_bundled).
    """
    prefix = f"_hermes_user_memory.{provider_name}"
    return [k for k in sys.modules if k == prefix or k.startswith(prefix + ".")]


class TestLoadProviderSkipsNonSubmoduleFiles:
    """Blindly executing every *.py is unsafe — see #38674."""

    def test_real_submodule_still_loads(self, fake_provider_dir):
        """The legitimate .py files (like store.py) must still be loaded so
        relative imports inside the plugin's __init__.py keep working."""
        provider = _load_provider_from_dir(fake_provider_dir)
        assert provider is not None
        assert provider.name == "fake"

        ns = _provider_namespace_modules("fake_provider")
        # The provider itself, plus its real `.store` submodule, must be in
        # the namespace.
        assert "_hermes_user_memory.fake_provider" in ns
        assert "_hermes_user_memory.fake_provider.store" in ns

        loaded = sys.modules["_hermes_user_memory.fake_provider.store"]
        assert getattr(loaded, "SENTINEL_LOADED", False) is True

    def test_setup_py_is_not_executed(self, fake_provider_dir):
        """setup.py is packaging, not a plugin submodule. It must not appear
        in the provider's sys.modules namespace after the load."""
        _load_provider_from_dir(fake_provider_dir)

        ns = _provider_namespace_modules("fake_provider")
        assert "_hermes_user_memory.fake_provider.setup" not in ns, (
            f"setup.py was executed as a submodule: {ns}"
        )

    def test_conftest_py_is_not_executed(self, fake_provider_dir):
        """conftest.py is pytest config, not a plugin submodule."""
        _load_provider_from_dir(fake_provider_dir)

        ns = _provider_namespace_modules("fake_provider")
        assert "_hermes_user_memory.fake_provider.conftest" not in ns, (
            f"conftest.py was executed as a submodule: {ns}"
        )

    def test_test_prefixed_files_are_not_executed(self, fake_provider_dir):
        """test_*.py and *_test.py are tests, not submodules."""
        _load_provider_from_dir(fake_provider_dir)

        ns = _provider_namespace_modules("fake_provider")
        bad = [k for k in ns if ".test_" in k or "_test." in k]
        assert not bad, f"test files were executed as submodules: {bad}"

    def test_pyproject_py_is_not_executed(self, fake_provider_dir):
        """pyproject.py would be ambiguous packaging config; skip it."""
        _load_provider_from_dir(fake_provider_dir)

        ns = _provider_namespace_modules("fake_provider")
        assert "_hermes_user_memory.fake_provider.pyproject" not in ns, (
            f"pyproject.py was executed as a submodule: {ns}"
        )


class TestLoadProviderReraisesBaseException:
    """SystemExit / KeyboardInterrupt must NOT be swallowed as Exception."""

    def test_systemexit_from_submodule_propagates(self, tmp_path, monkeypatch):
        """If a submodule calls sys.exit(), it must propagate out of
        _load_provider_from_dir — not be silently caught and replaced
        with a debug log line.

        Before the fix, the bare `except Exception` let SystemExit through
        (because SystemExit inherits from BaseException, not Exception) —
        but only because the except clause was wrong-shaped; the intent
        was to load-fail-and-continue, which would have hidden the crash
        if anyone had ever wrapped it in `except BaseException` later.
        This test pins the explicit re-raise.
        """
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes_home"))
        (tmp_path / "hermes_home").mkdir()

        provider = tmp_path / "evil_provider"
        provider.mkdir()
        _write(provider / "__init__.py", """
            from agent.memory_provider import MemoryProvider

            class EvilProvider(MemoryProvider):
                @property
                def name(self): return "evil"
                def is_available(self): return True
                def initialize(self, session_id, **kwargs): pass
                def get_tool_schemas(self): return []
        """)
        # Submodule that calls sys.exit() — same shape as a setuptools crash.
        _write(provider / "store.py", """
            import sys
            sys.exit(7)
        """)

        with pytest.raises(SystemExit) as excinfo:
            _load_provider_from_dir(provider)
        assert excinfo.value.code == 7

    def test_keyboardinterrupt_from_submodule_propagates(
        self, tmp_path, monkeypatch,
    ):
        """KeyboardInterrupt is also BaseException; same handling."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes_home"))
        (tmp_path / "hermes_home").mkdir()

        provider = tmp_path / "kbi_provider"
        provider.mkdir()
        _write(provider / "__init__.py", """
            from agent.memory_provider import MemoryProvider

            class KbiProvider(MemoryProvider):
                @property
                def name(self): return "kbi"
                def is_available(self): return True
                def initialize(self, session_id, **kwargs): pass
                def get_tool_schemas(self): return []
        """)
        _write(provider / "store.py", """
            raise KeyboardInterrupt()
        """)

        with pytest.raises(KeyboardInterrupt):
            _load_provider_from_dir(provider)
