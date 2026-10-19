"""Memory-provider load failures must be visible, not a silent "no provider".

The provider loader had the same two blind spots as the category-plugin loader: a
``register()`` that raised with nothing registered was logged at DEBUG, and
``_instantiate_subclass`` discarded a constructor exception with a bare ``pass``.
A provider whose database cannot be opened therefore looked exactly like one that
is not installed.
"""

from __future__ import annotations

import logging
import sys

import plugins.memory as memory_plugins


def _cleanup(tmp_path) -> None:
    prefix = memory_plugins._module_name(tmp_path, tmp_path.name)
    for name in tuple(sys.modules):
        if name == prefix or name.startswith(f"{prefix}."):
            sys.modules.pop(name, None)


def test_register_failure_names_the_provider_and_the_error(tmp_path, caplog):
    """A provider whose register() raises is reported at WARNING with its traceback."""
    provider = tmp_path / "brokendb"
    provider.mkdir()
    (provider / "__init__.py").write_text(
        "def register(ctx):\n"
        "    raise RuntimeError('provider database is locked')\n",
        encoding="utf-8",
    )
    caplog.set_level(logging.DEBUG, logger=memory_plugins.logger.name)

    try:
        assert memory_plugins._load_provider_from_dir(provider) is None
    finally:
        _cleanup(provider)

    warned = [record for record in caplog.records if record.levelno >= logging.WARNING]
    assert len(warned) == 1, [record.getMessage() for record in warned]
    message = warned[0].getMessage()
    assert "brokendb" in message
    assert "provider database is locked" in message
    assert warned[0].exc_info is not None


def test_constructor_failure_is_reported_instead_of_discarded(caplog):
    """A subclass whose constructor raises is a WARNING, not a swallowed ``pass``."""
    from agent.memory_provider import MemoryProvider

    class Exploding(MemoryProvider):
        def __init__(self) -> None:
            raise RuntimeError("memory store is malformed")

        @property
        def name(self) -> str:
            return "exploding"

        def is_available(self) -> bool:
            return True

        def initialize(self, session_id: str, **kwargs) -> None:
            raise AssertionError("must not be reached")

        def get_tool_schemas(self):
            return []

    caplog.set_level(logging.DEBUG, logger=memory_plugins.logger.name)

    assert memory_plugins._instantiate_subclass(type("ns", (), {"Exploding": Exploding})) is None

    warned = [record for record in caplog.records if record.levelno >= logging.WARNING]
    assert len(warned) == 1, [record.getMessage() for record in warned]
    message = warned[0].getMessage()
    assert "Exploding" in message
    assert "memory store is malformed" in message
    assert warned[0].exc_info is not None


def test_a_working_provider_stays_quiet(tmp_path, caplog):
    """The success path keeps returning the provider and logs no warning."""
    provider = tmp_path / "goodmem"
    provider.mkdir()
    (provider / "__init__.py").write_text(
        "from agent.memory_provider import MemoryProvider\n"
        "\n"
        "\n"
        "class Provider(MemoryProvider):\n"
        "    @property\n"
        "    def name(self):\n"
        "        return 'goodmem'\n"
        "\n"
        "    def is_available(self):\n"
        "        return True\n"
        "\n"
        "    def initialize(self, session_id, **kwargs):\n"
        "        pass\n"
        "\n"
        "    def get_tool_schemas(self):\n"
        "        return []\n"
        "\n"
        "\n"
        "def register(ctx):\n"
        "    ctx.register_memory_provider(Provider())\n",
        encoding="utf-8",
    )
    caplog.set_level(logging.DEBUG, logger=memory_plugins.logger.name)

    try:
        loaded = memory_plugins._load_provider_from_dir(provider)
    finally:
        _cleanup(provider)

    assert loaded is not None
    assert loaded.name == "goodmem"
    assert [record for record in caplog.records if record.levelno >= logging.WARNING] == []
