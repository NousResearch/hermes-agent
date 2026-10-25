"""Scoped Pyright settings survive trusted multi-root attachment in either order."""

import asyncio
import sys

import pytest

from agent.lsp import manager, servers
from agent.lsp.client import LSPClient, file_uri
from agent.lsp.workspace import clear_cache


@pytest.mark.parametrize("override", [False, True])
@pytest.mark.parametrize("scratch_first", [False, True])
def test_scratch_import_configuration_is_scoped_in_both_opening_orders(tmp_path, monkeypatch, scratch_first, override):
    runtime = tmp_path / "runtime"
    (runtime / ".git").mkdir(parents=True)
    task = runtime / "task"
    task.mkdir()
    external = tmp_path / "external"
    (external / ".git").mkdir(parents=True)
    scratch_file = task / "script.py"
    external_file = external / "first.py"
    scratch_file.write_text("", encoding="utf-8")
    external_file.write_text("", encoding="utf-8")
    monkeypatch.setenv("HERMES_KANBAN_TASK", "task")
    monkeypatch.setenv("HERMES_KANBAN_WORKSPACE", str(task))
    monkeypatch.chdir(task)
    monkeypatch.setattr(servers, "_detect_python", lambda root: None)
    clear_cache()
    notifications = []
    delivered = {}

    class RecordingClient(LSPClient):
        @property
        def is_running(self):
            return True

        async def start(self):
            pass

        async def shutdown(self):
            pass

        async def _send_notification(self, method, params):
            notifications.append((method, params))
            if method == "workspace/didChangeWorkspaceFolders":
                for folder in params["event"]["added"]:
                    delivered[folder["uri"]] = await self._handle_workspace_configuration(
                        {"items": [{"scopeUri": folder["uri"], "section": "python.analysis"}]}
                    )

    explicit = {"python": {"analysis": {"extraPaths": [str(tmp_path / "operator-imports")]}}}
    monkeypatch.setattr(manager, "LSPClient", RecordingClient)
    service = manager.LSPService(
        enabled=True, wait_mode="document", wait_timeout=5, install_strategy="off",
        trusted_workspaces=[str(runtime), str(external)],
        binary_overrides={"pyright": [sys.executable]},
        init_overrides={"pyright": explicit} if override else {},
    )
    try:
        order = [scratch_file, external_file] if scratch_first else [external_file, scratch_file]
        first = service._loop.run(service._get_or_spawn(str(order[0])), timeout=5)
        second = service._loop.run(service._get_or_spawn(str(order[1])), timeout=5)
        assert first is second
        assert set(first.workspace_folders) == {str(task), str(external)}
        assert len(service._clients) == 1
        expected = {"extraPaths": [str(runtime)], "autoSearchPaths": True}
        if override:
            expected = explicit["python"]["analysis"]
        external_expected = expected if override else None
        items = [
            {"scopeUri": file_uri(task), "section": "python.analysis"},
            {"scopeUri": file_uri(scratch_file), "section": "python.analysis.extraPaths"},
            {"scopeUri": file_uri(external), "section": "python.analysis"},
            {"scopeUri": file_uri(external_file), "section": "python.analysis"},
        ]
        assert service._loop.run(first._handle_workspace_configuration({"items": items}), timeout=5) == [
            expected, expected["extraPaths"], external_expected, external_expected,
        ]
        attached = order[1].parent
        assert delivered[file_uri(attached)] == ([external_expected] if scratch_first else [expected])
        # Repeated edits neither rebuild settings nor announce another workspace.
        service._loop.run(service._get_or_spawn(str(order[1])), timeout=5)
        assert len(notifications) == 1
    finally:
        service.shutdown()
        clear_cache()


def test_scoped_configuration_uses_nearest_attached_root_and_releases_options(tmp_path, monkeypatch):
    outer = str(tmp_path / "outer")
    inner = str(tmp_path / "outer" / "inner")
    sibling = str(tmp_path / "outer-other")
    client = LSPClient(server_id="pyright", workspace_root=outer, command=[],
                       initialization_options={"python": {"analysis": {"mode": "outer"}}})
    notifications = []

    async def notify(method, params):
        notifications.append((method, params))

    monkeypatch.setattr(client, "_send_notification", notify)

    async def check():
        inner_options = {"python": {"analysis": {"mode": "inner"}}}
        await client.add_workspace_folder(inner, initialization_options=inner_options)
        await client.add_workspace_folder(sibling, initialization_options={})
        items = [
            {"scopeUri": file_uri(inner + "/script.py"), "section": "python.analysis.mode"},
            {"scopeUri": file_uri(sibling + "/script.py"), "section": "python.analysis"},
            {"scopeUri": file_uri(inner)},
            {"section": "python.analysis.mode"},
            {"scopeUri": "untitled:probe", "section": "python.analysis.mode"},
            {"section": "missing"},
            None,
        ]
        assert await client._handle_workspace_configuration({"items": items}) == [
            "inner", None, inner_options, "outer", "outer", None, None,
        ]
        await client.remove_workspace_folder(inner)
        assert await client._handle_workspace_configuration({"items": items[:1]}) == ["outer"]
        await client.add_workspace_folder(inner, initialization_options={})
        assert await client._handle_workspace_configuration({"items": items[:1]}) == [None]
        assert await client._handle_workspace_configuration(None) == [None]
        assert await client._handle_workspace_configuration({}) == []
        assert len(notifications) == 4

    asyncio.run(check())


@pytest.mark.platforms("windows")
def test_scoped_configuration_accepts_pyright_encoded_lowercase_drive_uri(tmp_path, monkeypatch):
    root = str(tmp_path / "workspace with spaces")
    client = LSPClient(server_id="pyright", workspace_root=str(tmp_path / "external"), command=[])

    async def notify(method, params):
        pass

    monkeypatch.setattr(client, "_send_notification", notify)

    async def check():
        options = {"python": {"analysis": {"extraPaths": [str(tmp_path)]}}}
        await client.add_workspace_folder(root, initialization_options=options)
        drive = tmp_path.drive
        # Pyright percent-encodes the colon and folds the drive letter in scopeUri.
        uri = file_uri(root).replace(drive, drive.lower().replace(":", "%3A"))
        result = await client._handle_workspace_configuration(
            {"items": [{"scopeUri": uri, "section": "python.analysis"}]}
        )
        assert result == [options["python"]["analysis"]]

    asyncio.run(check())
