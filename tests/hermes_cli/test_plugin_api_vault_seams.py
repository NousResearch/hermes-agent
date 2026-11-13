"""The browser-vault plugin seams are on the published plugin API.

``hermes_agent.agent.vault_backends.base:LoginBackend``, ``hermes_agent.agent.vault_store:VaultItemMeta`` and
``hermes_agent.agent.browser_page_backend:BrowserPageBackend`` are what ``browser-vault-plugin.md`` tells a plugin to
subclass or return: ``ctx.register_login_backend`` / ``ctx.register_browser_page_backend`` accept only
instances of the base classes, and the core login-backend wrapper accepts only ``VaultItemMeta`` items.
A plugin built the way the guide shows passes ``hermes plugins validate`` without a private-import
warning, registers through the real ``PluginContext``, and is driven by the vault.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

import hermes_yaml as yaml
from hermes_cli.plugin_validate import validate_plugin_dir

_GUIDE = Path(__file__).resolve().parents[2] / "website/docs/developer-guide/browser-vault-plugin.md"


def _private_warnings(plugin: Path) -> list[str]:
    return [w for w in validate_plugin_dir(plugin).warnings if w.startswith("private core imports")]


def _plugin(root: Path, files: dict[str, str]) -> Path:
    root.mkdir(parents=True)
    (root / "plugin.yaml").write_text(yaml.safe_dump({"name": root.name, "version": "1.0.0", "description": "x"}),
                                      encoding="utf-8")
    for rel, text in {"__init__.py": "def register(ctx):\n    pass\n", **files}.items():
        (root / rel).write_text(text, encoding="utf-8")
    return root


@pytest.mark.parametrize("module, name", [
    ("agent.vault_backends.base", "LoginBackend"),
    ("agent.vault_store", "VaultItemMeta"),
    ("agent.browser_page_backend", "BrowserPageBackend"),
])
def test_plugin_importing_the_vault_seam_gets_no_private_import_warning(tmp_path, module, name):
    plugin = _plugin(tmp_path / "vaultplug", {"backend.py": f"from {module} import {name}\n"})

    assert _private_warnings(plugin) == []


def test_browser_vault_guide_examples_validate_clean(tmp_path):
    """A plugin that copies the guide's code blocks verbatim gets no private-import warning."""
    blocks = re.findall(r"```python\n(.*?)```", _GUIDE.read_text(encoding="utf-8"), re.DOTALL)
    assert any("LoginBackend" in b for b in blocks) and any("BrowserPageBackend" in b for b in blocks)
    plugin = _plugin(tmp_path / "guidecopy", {f"example_{i}.py": b for i, b in enumerate(blocks)})

    assert _private_warnings(plugin) == []


def _bundled_context(tmp_path, monkeypatch, name: str):
    """A bundled plugin's real ``PluginContext`` for the active profile (``HERMES_HOME``)."""
    from hermes_cli.plugins import PluginContext, PluginManager
    from hermes_cli.plugins_manifest import PluginManifest

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    return PluginContext(PluginManifest(name=name, source="bundled"), PluginManager(scope_key=str(home)))


def test_login_backend_subclassed_as_documented_serves_vault_item_meta_through_core(tmp_path, monkeypatch):
    """The guide's ``KeePassBackend`` shape (class attributes + the three abstract methods returning
    ``VaultItemMeta``) registers via ``ctx.register_login_backend`` and the vault reads it through core's
    wrapper; an item that is not a ``VaultItemMeta`` is refused, so the dataclass is the contract."""
    from hermes_agent.agent.vault_backends.base import LoginBackend, backend_for_handle
    from hermes_agent.agent.vault_backends.plugin import PluginBackendError
    from hermes_agent.agent.vault_store import VaultItemMeta

    item = VaultItemMeta(id="keepass:1", kind="login", label="Example", origin="https://example.com",
                         created_at="2026-01-01T00:00:00Z", identifier="me@example.com",
                         identifier_type="email", allowed_origins=("https://example.com",))

    class KeePassBackend(LoginBackend):
        name = "keepass"
        prefix = "keepass:"
        display_name = "KeePass"
        items: list = [item]

        def list_items(self) -> list[VaultItemMeta]:
            return list(self.items)

        def get_meta(self, handle: str) -> VaultItemMeta | None:
            return next((i for i in self.items if getattr(i, "id", None) == handle), None)

        def resolve_password(self, handle: str) -> str:
            return "pw-for-" + handle

    plugin_backend = KeePassBackend()
    lease = _bundled_context(tmp_path, monkeypatch, "keepass-vault").register_login_backend(plugin_backend)
    assert lease is not None
    try:
        assert backend_for_handle("keepass:1") is None  # off until the profile opts in
        (tmp_path / "home" / "config.yaml").write_text("vault:\n  keepass:\n    enabled: true\n", encoding="utf-8")
        backend = backend_for_handle("keepass:1")
        assert backend is not None and backend.name == "keepass"
        [listed] = backend.list_items()
        assert isinstance(listed, VaultItemMeta) and listed == item and listed is not item
        assert backend.get_meta("keepass:1") == item
        assert backend.resolve_password("keepass:1") == "pw-for-keepass:1"

        plugin_backend.items = [item.to_dict()]
        with pytest.raises(PluginBackendError):
            backend.list_items()
    finally:
        lease.dispose()


def test_browser_page_backend_subclassed_as_documented_serves_its_task(tmp_path, monkeypatch):
    """The guide's ``MyBrowserPages`` shape registers via ``ctx.register_browser_page_backend`` (behind
    the guide's ``hasattr`` guard) and is the backend the vault picks for the task it holds a page for."""
    from hermes_agent.agent import browser_page_registry
    from hermes_agent.agent.browser_page_backend import BrowserPageBackend

    calls = []

    class MyBrowserPages(BrowserPageBackend):
        name = "mybrowser"

        def has_page(self, task_id: str) -> bool:
            return task_id == "task-1"

        def evaluate(self, task_id: str, expression: str, *, secret: bool):
            calls.append((task_id, expression, secret))
            return "https://example.com"

    ctx = _bundled_context(tmp_path, monkeypatch, "mybrowser-pages")
    assert hasattr(ctx, "register_browser_page_backend")
    lease = ctx.register_browser_page_backend(MyBrowserPages())
    assert lease is not None
    try:
        backend = browser_page_registry.backend_for_task("task-1")
        assert isinstance(backend, MyBrowserPages)
        assert browser_page_registry.backend_for_task("other-task") is None
        assert backend.evaluate("task-1", "location.origin", secret=False) == "https://example.com"
        assert calls == [("task-1", "location.origin", False)]
    finally:
        lease.dispose()
