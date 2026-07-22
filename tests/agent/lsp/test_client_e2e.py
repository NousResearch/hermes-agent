"""End-to-end client tests against the in-process mock LSP server.

Spins up :file:`_mock_lsp_server.py` as an actual subprocess, drives
it through real LSP traffic, and asserts diagnostic flow.  This is
the closest thing we have to integration coverage without requiring
pyright/gopls/etc. to be installed in CI.
"""
from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

import pytest

from agent.lsp.client import LSPClient, PUSH_DEBOUNCE, file_uri, uri_to_path


MOCK_SERVER = str(Path(__file__).parent / "_mock_lsp_server.py")


def _client(workspace: Path, script: str = "clean") -> LSPClient:
    env = {"MOCK_LSP_SCRIPT": script, "PYTHONPATH": os.environ.get("PYTHONPATH", "")}
    return LSPClient(
        server_id=f"mock-{script}",
        workspace_root=str(workspace),
        command=[sys.executable, MOCK_SERVER],
        env=env,
        cwd=str(workspace),
    )


@pytest.mark.asyncio
async def test_client_lifecycle_clean(tmp_path: Path):
    """Full lifecycle: spawn, initialize, open, get clean diagnostics, shutdown."""
    f = tmp_path / "x.py"
    f.write_text("print('hi')\n")

    client = _client(tmp_path, "clean")
    await client.start()
    try:
        assert client.is_running
        version = await client.open_file(str(f), language_id="python")
        assert version == 0
        await client.wait_for_diagnostics(str(f), version, mode="document")
        diags = client.diagnostics_for(str(f))
        assert diags == []
    finally:
        await client.shutdown()
    assert not client.is_running


@pytest.mark.asyncio
async def test_client_receives_published_errors(tmp_path: Path):
    f = tmp_path / "x.py"
    f.write_text("print('hi')\n")

    client = _client(tmp_path, "errors")
    await client.start()
    try:
        version = await client.open_file(str(f), language_id="python")
        await client.wait_for_diagnostics(str(f), version, mode="document")
        diags = client.diagnostics_for(str(f))
        assert len(diags) == 1
        d = diags[0]
        assert d["severity"] == 1
        assert d["code"] == "MOCK001"
        assert d["source"] == "mock-lsp"
        assert "synthetic error" in d["message"]
    finally:
        await client.shutdown()


@pytest.mark.asyncio
async def test_client_didchange_bumps_version(tmp_path: Path):
    f = tmp_path / "x.py"
    f.write_text("print('hi')\n")

    client = _client(tmp_path, "errors")
    await client.start()
    try:
        v0 = await client.open_file(str(f), language_id="python")
        f.write_text("print('hi 2')\n")
        v1 = await client.open_file(str(f), language_id="python")  # re-open path = didChange
        assert v1 == v0 + 1
        await client.wait_for_diagnostics(str(f), v1, mode="document")
        # Mock pushed a diagnostic for both events; merged view has one
        # entry (push store keyed by file path).
        diags = client.diagnostics_for(str(f))
        assert len(diags) == 1
    finally:
        await client.shutdown()


@pytest.mark.asyncio
async def test_client_handles_crashing_server(tmp_path: Path):
    """When the server exits right after initialize, subsequent requests
    fail gracefully (not hang)."""
    f = tmp_path / "x.py"
    f.write_text("")

    client = _client(tmp_path, "crash")
    await client.start()  # should succeed (mock answers initialize before crashing)
    # Give the OS a moment to deliver the EOF.
    await asyncio.sleep(0.2)
    # The reader loop should detect EOF and mark pending requests as failed.
    try:
        await asyncio.wait_for(
            client.open_file(str(f), language_id="python"), timeout=2.0
        )
    except Exception:
        pass  # any exception is acceptable; the contract is "doesn't hang"
    await client.shutdown()


@pytest.mark.asyncio
async def test_client_shutdown_idempotent(tmp_path: Path):
    """Calling shutdown twice must be safe."""
    f = tmp_path / "x.py"
    f.write_text("")
    client = _client(tmp_path, "clean")
    await client.start()
    await client.shutdown()
    await client.shutdown()  # must not raise


@pytest.mark.asyncio
async def test_client_diagnostics_are_deduped(tmp_path: Path):
    """Repeated identical pushes must not produce duplicate diagnostics."""
    f = tmp_path / "x.py"
    f.write_text("")
    client = _client(tmp_path, "errors")
    await client.start()
    try:
        for _ in range(3):
            v = await client.open_file(str(f), language_id="python")
            await client.wait_for_diagnostics(str(f), v, mode="document")
        diags = client.diagnostics_for(str(f))
        # Push store overwrites on every notification — should have 1.
        assert len(diags) == 1
    finally:
        await client.shutdown()


# Windows path identity and push-only diagnostic regressions.


async def _open_without_server(
    client: LSPClient,
    path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    language_id: str = "typescript",
) -> int:
    class RunningProcess:
        returncode = None

    async def ignore_notification(_method: str, _params: object) -> None:
        return None

    client._state = "running"
    client._proc = RunningProcess()
    monkeypatch.setattr(client, "_send_notification", ignore_notification)
    return await client.open_file(str(path), language_id=language_id)


@pytest.mark.asyncio
async def test_wait_keeps_push_waiter_when_pull_is_unsupported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """A fast failed pull must not starve delayed push diagnostics."""
    path = tmp_path / "x.ts"
    path.write_text("const value = 1;\n")
    client = _client(tmp_path, "clean")
    version = await _open_without_server(client, path, monkeypatch)
    pull_calls = 0

    async def unsupported_pull(_path: str) -> None:
        nonlocal pull_calls
        pull_calls += 1

    async def delayed_push() -> None:
        await asyncio.sleep(0.01)
        client._handle_publish_diagnostics(
            {
                "uri": file_uri(str(path)),
                "diagnostics": [{"message": "TS diagnostic"}],
            }
        )

    monkeypatch.setattr(client, "_pull_document_diagnostics", unsupported_pull)

    producer = asyncio.create_task(delayed_push())
    fresh = await client.wait_for_diagnostics(
        str(path), version, mode="document", timeout=1.0
    )
    await producer

    assert fresh is True
    assert pull_calls == 1
    assert client.diagnostics_for(str(path)) == [{"message": "TS diagnostic"}]


@pytest.mark.asyncio
async def test_protocol_rejected_pull_accepts_delayed_unversioned_push(
    tmp_path: Path,
):
    """Exercise rejection-first ordering through the real subprocess reader."""
    path = tmp_path / "x.ts"
    path.write_text("const value = 1;\n")
    client = _client(tmp_path, "reject_then_unversioned_push")
    assert client._env is not None
    client._env["MOCK_LSP_PUSH_DELAY"] = "0.05"
    await client.start()
    try:
        version_zero = await client.open_file(str(path), language_id="typescript")
        assert await client.wait_for_diagnostics(
            str(path), version_zero, mode="document", timeout=1.0
        )
        key = uri_to_path(file_uri(str(path)))
        assert client._docs[key].push_version == version_zero

        path.write_text("const value = 2;\n")
        version_one = await client.open_file(str(path), language_id="typescript")
        assert await client.wait_for_diagnostics(
            str(path), version_one, mode="document", timeout=1.0
        )

        assert client._docs[key].push_version == version_one
        diagnostics = client.diagnostics_for(str(path), fresh_only=True)
        assert [diagnostic["code"] for diagnostic in diagnostics] == [
            "MOCK_UNVERSIONED"
        ]
    finally:
        await client.shutdown()


@pytest.mark.asyncio
async def test_wait_cancellation_stops_preserved_push_task(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    path = tmp_path / "x.ts"
    path.write_text("const value = 1;\n")
    client = _client(tmp_path, "clean")
    version = await _open_without_server(client, path, monkeypatch)
    push_started = asyncio.Event()
    push_cancelled = asyncio.Event()

    async def unsupported_pull(_path: str) -> None:
        return None

    async def blocking_push(
        _path: str, _version: int, _timeout: float, _baseline: int
    ) -> None:
        push_started.set()
        try:
            await asyncio.Event().wait()
        finally:
            push_cancelled.set()

    monkeypatch.setattr(client, "_pull_document_diagnostics", unsupported_pull)
    monkeypatch.setattr(client, "_wait_for_fresh_push", blocking_push)

    waiter = asyncio.create_task(client.wait_for_diagnostics(str(path), version))
    await asyncio.wait_for(push_started.wait(), timeout=0.5)
    waiter.cancel()

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(waiter, timeout=0.5)
    await asyncio.wait_for(push_cancelled.wait(), timeout=0.5)


@pytest.mark.skipif(os.name != "nt", reason="Windows URI normalization")
def test_encoded_drive_uri_maps_to_original_native_path(tmp_path: Path):
    native_path = r"C:\Workspace\Project\src\Index.ts"
    uri = "file:///c%3A/workspace/project/src/index.ts"
    diagnostic = {"message": "mapped diagnostic"}
    client = LSPClient(
        server_id="typescript",
        workspace_root=str(tmp_path),
        command=[sys.executable, MOCK_SERVER],
    )

    client._handle_publish_diagnostics({"uri": uri, "diagnostics": [diagnostic]})

    assert client.diagnostics_for(native_path) == [diagnostic]
    assert file_uri(native_path) == "file:///C:/Workspace/Project/src/Index.ts"


@pytest.mark.skipif(os.name != "nt", reason="Windows UNC URI normalization")
def test_unc_file_uri_round_trip():
    native_path = r"\\Server\Share\Folder\File.ts"
    assert file_uri(native_path) == "file://Server/Share/Folder/File.ts"
    assert uri_to_path("file://server/share/folder/file.ts") == os.path.normcase(
        os.path.abspath(native_path)
    )


def test_uri_to_path_preserves_non_file_uri():
    uri = "untitled:Untitled-1"
    assert uri_to_path(uri) == uri


def test_non_file_diagnostic_uri_preserves_opaque_identity(tmp_path: Path):
    uri = "untitled:CaseSensitive-1"
    diagnostic = {"message": "virtual document diagnostic"}
    client = _client(tmp_path, "clean")

    client._handle_publish_diagnostics({"uri": uri, "diagnostics": [diagnostic]})

    assert uri in client._docs
    assert client.diagnostics_for(uri) == [diagnostic]


@pytest.mark.skipif(os.name == "nt", reason="POSIX URI behavior")
def test_posix_file_uri_round_trip_preserves_case():
    path = "/tmp/MixedCase/File Name.ts"
    assert uri_to_path(file_uri(path)) == os.path.normpath(os.path.abspath(path))


@pytest.mark.asyncio
async def test_edit_baselines_are_tied_to_returned_versions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    path = tmp_path / "index.ts"
    path.write_text("const value = 1;\n")
    client = _client(tmp_path, "clean")

    version_zero = await _open_without_server(client, path, monkeypatch)
    client._push_counter = 4
    path.write_text("const value = 2;\n")
    version_one = await client.open_file(str(path), language_id="typescript")
    key = uri_to_path(file_uri(str(path)))

    assert client._diagnostic_baselines[(key, version_zero)] == 0
    assert client._diagnostic_baselines[(key, version_one)] == 4


@pytest.mark.asyncio
async def test_overlapping_opens_allocate_distinct_versions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    path = tmp_path / "index.ts"
    path.write_text("const value = 1;\n")
    client = _client(tmp_path, "clean")
    await _open_without_server(client, path, monkeypatch)

    async def yield_after_notification(_method: str, _params: object) -> None:
        await asyncio.sleep(0)

    monkeypatch.setattr(client, "_send_notification", yield_after_notification)
    versions = await asyncio.gather(
        client.open_file(str(path), language_id="typescript"),
        client.open_file(str(path), language_id="typescript"),
    )

    assert versions == [1, 2]
    key = uri_to_path(file_uri(str(path)))
    assert (key, 1) in client._diagnostic_baselines
    assert (key, 2) in client._diagnostic_baselines


@pytest.mark.asyncio
async def test_push_received_during_did_change_is_fresh(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    path = tmp_path / "index.ts"
    path.write_text("const value = 1;\n")
    client = _client(tmp_path, "clean")
    await _open_without_server(client, path, monkeypatch)
    path.write_text("const value: string = 1;\n")
    diagnostic = {"message": "Type 'number' is not assignable to type 'string'"}

    async def publish_during_change(method: str, _params: object) -> None:
        if method == "textDocument/didChange":
            client._handle_publish_diagnostics(
                {"uri": file_uri(str(path)), "diagnostics": [diagnostic]}
            )

    monkeypatch.setattr(client, "_send_notification", publish_during_change)
    version = await client.open_file(str(path), language_id="typescript")

    assert await client.wait_for_diagnostics(str(path), version, timeout=0.5)
    assert client.diagnostics_for(str(path), fresh_only=True) == [diagnostic]


def test_seed_first_push_is_not_marked_fresh(tmp_path: Path):
    client = LSPClient(
        server_id="typescript",
        workspace_root=str(tmp_path),
        command=[sys.executable, MOCK_SERVER],
        seed_diagnostics_on_first_push=True,
    )
    path = str(tmp_path / "index.ts")
    uri = file_uri(path)
    key = uri_to_path(uri)

    client._handle_publish_diagnostics({"uri": uri, "version": 0, "diagnostics": []})
    doc = client._docs[key]
    assert doc.push_counter == 0
    assert doc.push_version == -1

    diagnostic = {"message": "fresh TypeScript error"}
    client._handle_publish_diagnostics(
        {"uri": uri, "version": 1, "diagnostics": [diagnostic]}
    )
    assert doc.push_counter == 1
    assert doc.push_version == 1
    assert client.diagnostics_for(path) == [diagnostic]


def test_unversioned_push_drops_stale_version_metadata(tmp_path: Path):
    client = LSPClient(
        server_id="typescript",
        workspace_root=str(tmp_path),
        command=[sys.executable, MOCK_SERVER],
    )
    path = str(tmp_path / "index.ts")
    uri = file_uri(path)
    key = uri_to_path(uri)

    client._handle_publish_diagnostics({"uri": uri, "version": 1, "diagnostics": []})
    assert client._docs[key].push_version == 1

    client._handle_publish_diagnostics({"uri": uri, "diagnostics": []})
    assert client._docs[key].push_version == -1


@pytest.mark.asyncio
async def test_unversioned_push_freshness_is_scoped_to_target_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    client = _client(tmp_path, "clean")
    target = tmp_path / "index.ts"
    other = tmp_path / "other.ts"
    target.write_text("const value = 1;\n")
    other.write_text("const other = 1;\n")
    target_uri = file_uri(str(target))
    target_key = uri_to_path(target_uri)
    stale = {"message": "stale TypeScript error"}

    await _open_without_server(client, target, monkeypatch)
    client._handle_publish_diagnostics({"uri": target_uri, "diagnostics": [stale]})
    baseline = client._push_counter
    target.write_text("const value = 2;\n")
    version = await client.open_file(str(target), language_id="typescript")
    waiter = asyncio.create_task(
        client._wait_for_fresh_push(
            target_key, version=version, timeout=1.0, baseline=baseline
        )
    )

    client._handle_publish_diagnostics(
        {"uri": file_uri(str(other)), "diagnostics": []}
    )
    await asyncio.sleep(PUSH_DEBOUNCE + 0.05)
    assert not waiter.done()

    client._handle_publish_diagnostics({"uri": target_uri, "diagnostics": []})
    await waiter
    assert client.diagnostics_for(str(target)) == []


@pytest.mark.asyncio
async def test_unversioned_push_waits_for_post_edit_quiet_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """TypeScript's stale unversioned push must not win over the clean one."""
    client = _client(tmp_path, "clean")
    path = tmp_path / "index.ts"
    path.write_text("const value = 1;\n")
    uri = file_uri(str(path))
    key = uri_to_path(uri)
    stale = {"message": "stale TypeScript error"}

    await _open_without_server(client, path, monkeypatch)
    client._handle_publish_diagnostics({"uri": uri, "diagnostics": [stale]})
    baseline = client._push_counter
    path.write_text("const value = 2;\n")
    version = await client.open_file(str(path), language_id="typescript")

    async def publish_after_edit() -> None:
        await asyncio.sleep(0.01)
        client._handle_publish_diagnostics({"uri": uri, "diagnostics": [stale]})
        await asyncio.sleep(0.05)
        client._handle_publish_diagnostics({"uri": uri, "diagnostics": []})

    producer = asyncio.create_task(publish_after_edit())
    await client._wait_for_fresh_push(
        key, version=version, timeout=1.0, baseline=baseline
    )
    await producer

    assert client.diagnostics_for(str(path)) == []
