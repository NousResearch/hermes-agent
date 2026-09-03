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

from agent.lsp.client import (
    LSPClient,
    PUSH_DEBOUNCE,
    UNVERSIONED_PUSH_DEBOUNCE,
    UNVERSIONED_PUSH_STABILIZATION,
    file_uri,
    uri_to_path,
)
from agent.lsp.protocol import LSPProtocolError


MOCK_SERVER = str(Path(__file__).parent / "_mock_lsp_server.py")
EARLY_TIMING_SLACK = 0.10
NO_FULL_WAIT_CEILING = 3.0


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
async def test_reader_exit_at_end_of_initialization_retires_client(tmp_path: Path):
    client = _client(tmp_path, "crash")

    try:
        await client.start()
    except LSPProtocolError:
        pass
    else:
        reader_task = client._reader_task
        if reader_task is not None:
            await asyncio.wait_for(asyncio.shield(reader_task), timeout=3.0)

    assert client.state == "error"
    assert not client.is_running
    assert client._proc is None
    await client.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize("script", ["clean_eof", "malformed_frame"])
async def test_reader_failure_retires_client_and_rejects_later_work(
    tmp_path: Path, script: str
):
    f = tmp_path / "x.py"
    f.write_text("print('hi')\n")

    client = _client(tmp_path, script)
    await client.start()
    proc = client._proc
    reader_task = client._reader_task
    assert proc is not None
    assert reader_task is not None
    try:
        version = await client.open_file(str(f), language_id="python")
        await asyncio.wait_for(asyncio.shield(reader_task), timeout=3.0)

        assert not client.is_running
        await asyncio.wait_for(proc.wait(), timeout=3.0)
        with pytest.raises(LSPProtocolError):
            await asyncio.wait_for(
                client.wait_for_diagnostics(str(f), version, timeout=3.0),
                timeout=0.5,
            )
        with pytest.raises(LSPProtocolError):
            await asyncio.wait_for(
                client.open_file(str(f), language_id="python"),
                timeout=0.5,
            )
    finally:
        await client.shutdown()


@pytest.mark.asyncio
async def test_transport_death_interrupts_preserved_push_wait(tmp_path: Path):
    path = tmp_path / "x.py"
    path.write_text("print('hi')\n")
    client = _client(tmp_path, "diagnostic_eof")
    await client.start()
    try:
        version = await client.open_file(str(path), language_id="python")
        with pytest.raises(LSPProtocolError):
            await asyncio.wait_for(
                client.wait_for_diagnostics(str(path), version, timeout=3.0),
                timeout=0.5,
            )
    finally:
        await client.shutdown()


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
    monkeypatch.setattr(client, "_proc", RunningProcess())
    monkeypatch.setattr(client, "_connection_is_open", lambda: True)
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

    async def unsupported_pull(_path: str, _version: int) -> None:
        nonlocal pull_calls
        pull_calls += 1

    async def delayed_push() -> None:
        await asyncio.sleep(0.01)
        client._handle_publish_diagnostics({
            "uri": file_uri(str(path)),
            "diagnostics": [{"message": "TS diagnostic"}],
        })

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

    async def unsupported_pull(_path: str, _version: int) -> None:
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
    await asyncio.wait_for(push_started.wait(), timeout=1.5)
    waiter.cancel()

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(waiter, timeout=1.5)
    await asyncio.wait_for(push_cancelled.wait(), timeout=1.5)


@pytest.mark.windows_only
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
    client._handle_publish_diagnostics({
        "uri": "FILE:///c%3A/workspace/project/src/index.ts",
        "diagnostics": [diagnostic],
    })

    assert client.diagnostics_for(native_path) == [diagnostic]
    assert file_uri(native_path) == "file:///C:/Workspace/Project/src/Index.ts"


@pytest.mark.windows_only
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
    upper_uri = "x:CaseSensitive-1"
    lower_uri = "x:casesensitive-1"
    upper_diagnostic = {"message": "upper virtual document diagnostic"}
    lower_diagnostic = {"message": "lower virtual document diagnostic"}
    client = _client(tmp_path, "clean")

    client._handle_publish_diagnostics({
        "uri": upper_uri,
        "diagnostics": [upper_diagnostic],
    })
    client._handle_publish_diagnostics({
        "uri": lower_uri,
        "diagnostics": [lower_diagnostic],
    })

    assert upper_uri in client._docs
    assert lower_uri in client._docs
    assert client.diagnostics_for(upper_uri) == [upper_diagnostic]
    assert client.diagnostics_for(lower_uri) == [lower_diagnostic]


def test_relative_path_with_colon_uses_open_document_key(tmp_path: Path):
    relative_path = "notes:2026.ts"
    diagnostic = {"message": "relative POSIX path diagnostic"}
    client = _client(tmp_path, "clean")
    client._handle_publish_diagnostics({
        "uri": file_uri(os.path.abspath(relative_path)),
        "diagnostics": [diagnostic],
    })

    assert client.diagnostics_for(relative_path) == [diagnostic]


def _assert_posix_file_uri_round_trip_preserves_case() -> None:
    path = "/tmp/MixedCase/File Name.ts"
    assert uri_to_path(file_uri(path)) == os.path.normpath(os.path.abspath(path))


@pytest.mark.linux_only
def test_posix_file_uri_round_trip_preserves_case_on_linux():
    _assert_posix_file_uri_round_trip_preserves_case()


@pytest.mark.macos_only
def test_posix_file_uri_round_trip_preserves_case_on_macos():
    _assert_posix_file_uri_round_trip_preserves_case()


@pytest.mark.asyncio
async def test_edit_baseline_tracks_current_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    path = tmp_path / "index.ts"
    path.write_text("const value = 1;\n")
    client = _client(tmp_path, "clean")

    await _open_without_server(client, path, monkeypatch)
    key = uri_to_path(file_uri(str(path)))
    assert client._docs[key].diagnostic_baseline == 0

    client._push_counter = 4
    path.write_text("const value = 2;\n")
    version_one = await client.open_file(str(path), language_id="typescript")

    assert client._docs[key].version == version_one
    assert client._docs[key].diagnostic_baseline == 4


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
    assert client._docs[key].version == 2


@pytest.mark.asyncio
async def test_superseded_wait_returns_false_without_a_clean_verdict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    path = tmp_path / "index.ts"
    path.write_text("const value = 0;\n")
    client = _client(tmp_path, "clean")
    version_zero = await _open_without_server(client, path, monkeypatch)

    async def unsupported_pull(*_args: object) -> None:
        return None

    monkeypatch.setattr(client, "_pull_document_diagnostics", unsupported_pull)
    old_wait = asyncio.create_task(
        client.wait_for_diagnostics(str(path), version_zero, timeout=1.0)
    )
    await asyncio.sleep(0)

    path.write_text("const value = 1;\n")
    version_one = await client.open_file(str(path), language_id="typescript")

    assert version_one == version_zero + 1
    assert await asyncio.wait_for(old_wait, timeout=1.5) is False
    assert client.diagnostics_for(str(path), fresh_only=True) == []


@pytest.mark.asyncio
async def test_superseded_pull_cannot_overwrite_latest_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    path = tmp_path / "index.ts"
    path.write_text("const value = 0;\n")
    client = _client(tmp_path, "clean")
    version_zero = await _open_without_server(client, path, monkeypatch)
    old_pull_started = asyncio.Event()
    release_old_pull = asyncio.Event()
    old_diagnostic = {"message": "old pull result"}
    latest_diagnostic = {"message": "latest pull result"}
    pull_calls = 0

    async def staged_request(
        _method: str, _params: object, *, timeout: float
    ) -> object:
        nonlocal pull_calls
        pull_calls += 1
        if pull_calls == 1:
            old_pull_started.set()
            try:
                await release_old_pull.wait()
            except asyncio.CancelledError:
                # Model a response already committed by the server/transport.
                await release_old_pull.wait()
            return {"items": [old_diagnostic]}
        return {"items": [latest_diagnostic]}

    monkeypatch.setattr(client, "_send_request_with_retry", staged_request)
    old_wait = asyncio.create_task(
        client.wait_for_diagnostics(str(path), version_zero, timeout=2.0)
    )
    await asyncio.wait_for(old_pull_started.wait(), timeout=1.5)

    path.write_text("const value = 1;\n")
    version_one = await client.open_file(str(path), language_id="typescript")
    latest_wait = asyncio.create_task(
        client.wait_for_diagnostics(str(path), version_one, timeout=1.5)
    )
    assert await latest_wait is True
    assert client.diagnostics_for(str(path), fresh_only=True) == [latest_diagnostic]

    release_old_pull.set()
    assert await asyncio.wait_for(old_wait, timeout=1.5) is False
    assert client.diagnostics_for(str(path), fresh_only=True) == [latest_diagnostic]


@pytest.mark.asyncio
async def test_related_pull_result_is_discarded_when_related_generation_advances(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    primary = tmp_path / "primary.ts"
    related = tmp_path / "related.ts"
    primary.write_text("export const primary = 0;\n")
    related.write_text("export const related = 0;\n")
    client = _client(tmp_path, "clean")
    primary_version = await _open_without_server(client, primary, monkeypatch)
    await client.open_file(str(related), language_id="typescript")
    pull_started = asyncio.Event()
    release_pull = asyncio.Event()
    stale_related = {"message": "stale related pull result"}
    current_related = {"message": "current related push result"}

    async def delayed_request(
        _method: str, _params: object, *, timeout: float
    ) -> object:
        pull_started.set()
        await release_pull.wait()
        return {
            "kind": "full",
            "items": [],
            "relatedDocuments": {
                file_uri(str(related)): {"kind": "full", "items": [stale_related]}
            },
        }

    monkeypatch.setattr(client, "_send_request_with_retry", delayed_request)
    pull = asyncio.create_task(
        client._pull_document_diagnostics(str(primary), primary_version)
    )
    await asyncio.wait_for(pull_started.wait(), timeout=1.5)

    related.write_text("export const related = 1;\n")
    related_version = await client.open_file(str(related), language_id="typescript")
    client._handle_publish_diagnostics({
        "uri": file_uri(str(related)),
        "version": related_version,
        "diagnostics": [current_related],
    })
    release_pull.set()
    await pull

    related_doc = client._docs[uri_to_path(file_uri(str(related)))]
    assert related_doc.pull == []
    assert related_doc.pull_version == -1
    assert client.diagnostics_for(str(related)) == [current_related]
    assert client.diagnostics_for(str(related), fresh_only=True) == [current_related]


@pytest.mark.asyncio
async def test_never_opened_related_pull_is_stored_but_not_fresh(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    primary = tmp_path / "primary.ts"
    related = tmp_path / "related.ts"
    primary.write_text("export const primary = 0;\n")
    client = _client(tmp_path, "clean")
    primary_version = await _open_without_server(client, primary, monkeypatch)
    related_diagnostic = {"message": "never-opened related pull result"}

    async def related_result(
        _method: str, _params: object, *, timeout: float
    ) -> object:
        return {
            "kind": "full",
            "items": [],
            "relatedDocuments": {
                file_uri(str(related)): {
                    "kind": "full",
                    "items": [related_diagnostic],
                }
            },
        }

    monkeypatch.setattr(client, "_send_request_with_retry", related_result)
    await client._pull_document_diagnostics(str(primary), primary_version)

    assert client.diagnostics_for(str(related)) == [related_diagnostic]
    assert client.diagnostics_for(str(related), fresh_only=True) == []


@pytest.mark.asyncio
async def test_overlapping_waits_settle_only_the_latest_edit_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    path = tmp_path / "index.ts"
    path.write_text("const value = 0;\n")
    client = _client(tmp_path, "clean")
    version_zero = await _open_without_server(client, path, monkeypatch)
    uri = file_uri(str(path))
    stale = {"message": "stale TypeScript error"}
    quiet_period = 0.60

    async def unsupported_pull(*_args: object) -> None:
        return None

    monkeypatch.setattr(client, "_pull_document_diagnostics", unsupported_pull)
    monkeypatch.setattr("agent.lsp.client.UNVERSIONED_PUSH_DEBOUNCE", quiet_period)
    monkeypatch.setattr("agent.lsp.client.UNVERSIONED_PUSH_STABILIZATION", 2.0)

    old_wait = asyncio.create_task(
        client.wait_for_diagnostics(str(path), version_zero, timeout=2.0)
    )
    await asyncio.sleep(0)
    path.write_text("const value = 1;\n")
    version_one = await client.open_file(str(path), language_id="typescript")
    latest_wait = asyncio.create_task(
        client.wait_for_diagnostics(str(path), version_one, timeout=2.0)
    )
    await asyncio.sleep(0)

    client._handle_publish_diagnostics({"uri": uri, "diagnostics": [stale]})
    await asyncio.sleep(0.40)
    client._handle_publish_diagnostics({"uri": uri, "diagnostics": []})
    await asyncio.sleep(0.35)

    assert await old_wait is False
    assert not latest_wait.done(), "the latest target push must reset quiet settling"
    assert await asyncio.wait_for(latest_wait, timeout=1.5) is True
    assert client.diagnostics_for(str(path), fresh_only=True) == []


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
            client._handle_publish_diagnostics({
                "uri": file_uri(str(path)),
                "diagnostics": [diagnostic],
            })

    monkeypatch.setattr(client, "_send_notification", publish_during_change)
    version = await client.open_file(str(path), language_id="typescript")

    assert await client.wait_for_diagnostics(str(path), version, timeout=1.5)
    assert client.diagnostics_for(str(path), fresh_only=True) == [diagnostic]


@pytest.mark.parametrize("version_offset", [-1, 1])
@pytest.mark.asyncio
async def test_mismatched_versioned_push_does_not_replace_open_document_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    version_offset: int,
):
    path = tmp_path / "index.ts"
    path.write_text("const value = 0;\n")
    client = _client(tmp_path, "clean")
    await _open_without_server(client, path, monkeypatch)
    path.write_text("const value = 1;\n")
    version = await client.open_file(str(path), language_id="typescript")
    current = {"message": "current generation diagnostic"}
    mismatched = {"message": "mismatched generation diagnostic"}
    uri = file_uri(str(path))

    client._handle_publish_diagnostics({
        "uri": uri,
        "version": version,
        "diagnostics": [current],
    })
    counter = client._push_counter
    client._handle_publish_diagnostics({
        "uri": uri,
        "version": version + version_offset,
        "diagnostics": [mismatched],
    })

    doc = client._docs[uri_to_path(uri)]
    assert doc.push == [current]
    assert doc.push_version == version
    assert doc.push_counter == counter
    assert client._push_counter == counter
    assert client.diagnostics_for(str(path)) == [current]
    assert client.diagnostics_for(str(path), fresh_only=True) == [current]


@pytest.mark.parametrize("version_offset", [-1, 1])
@pytest.mark.asyncio
async def test_mismatched_versioned_push_cannot_satisfy_current_wait(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    version_offset: int,
):
    path = tmp_path / "index.ts"
    path.write_text("const value = 0;\n")
    client = _client(tmp_path, "clean")
    await _open_without_server(client, path, monkeypatch)
    previous = {"message": "previous generation diagnostic"}
    client._handle_publish_diagnostics({
        "uri": file_uri(str(path)),
        "version": 0,
        "diagnostics": [previous],
    })
    path.write_text("const value = 1;\n")
    version = await client.open_file(str(path), language_id="typescript")

    async def unsupported_pull(_path: str, _version: int) -> None:
        return None

    monkeypatch.setattr(client, "_pull_document_diagnostics", unsupported_pull)
    client._handle_publish_diagnostics({
        "uri": file_uri(str(path)),
        "version": version + version_offset,
        "diagnostics": [{"message": "mismatched generation diagnostic"}],
    })

    assert not await client.wait_for_diagnostics(str(path), version, timeout=0.05)
    assert client.diagnostics_for(str(path)) == [previous]
    assert client.diagnostics_for(str(path), fresh_only=True) == []


def test_versioned_push_for_never_opened_path_is_stored_but_not_fresh(
    tmp_path: Path,
):
    client = _client(tmp_path, "clean")
    path = tmp_path / "related.ts"
    diagnostic = {"message": "never-opened versioned push"}

    client._handle_publish_diagnostics({
        "uri": file_uri(str(path)),
        "version": 7,
        "diagnostics": [diagnostic],
    })

    assert client.diagnostics_for(str(path)) == [diagnostic]
    assert client.diagnostics_for(str(path), fresh_only=True) == []


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

    client._handle_publish_diagnostics({"uri": uri, "diagnostics": "invalid"})
    assert key not in client._docs

    client._handle_publish_diagnostics({"uri": uri, "version": 0, "diagnostics": []})
    doc = client._docs[key]
    assert doc.push_counter == 0
    assert doc.push_version == -1

    diagnostic = {"message": "fresh TypeScript error"}
    client._handle_publish_diagnostics({
        "uri": uri,
        "version": 1,
        "diagnostics": [diagnostic],
    })
    assert doc.push_counter == 1
    assert doc.push_version == 1
    assert client.diagnostics_for(path) == [diagnostic]


@pytest.mark.asyncio
async def test_stale_versioned_seed_does_not_swallow_current_generation_push(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    path = tmp_path / "index.ts"
    path.write_text("const value = 0;\n")
    client = LSPClient(
        server_id="typescript",
        workspace_root=str(tmp_path),
        command=[sys.executable, MOCK_SERVER],
        seed_diagnostics_on_first_push=True,
    )
    await _open_without_server(client, path, monkeypatch)
    path.write_text("const value = 1;\n")
    version = await client.open_file(str(path), language_id="typescript")
    uri = file_uri(str(path))
    key = uri_to_path(uri)
    diagnostic = {"message": "current generation diagnostic"}

    async def unsupported_pull(*_args: object) -> None:
        return None

    monkeypatch.setattr(client, "_pull_document_diagnostics", unsupported_pull)
    waiter = asyncio.create_task(
        client.wait_for_diagnostics(str(path), version, timeout=1.0)
    )
    await asyncio.sleep(0)
    client._handle_publish_diagnostics({
        "uri": uri,
        "version": version - 1,
        "diagnostics": [{"message": "stale generation diagnostic"}],
    })
    client._handle_publish_diagnostics({
        "uri": uri,
        "version": version,
        "diagnostics": [diagnostic],
    })

    assert await waiter is True
    doc = client._docs[key]
    assert doc.seed_seen is True
    assert doc.push_counter == 1
    assert client.diagnostics_for(str(path), fresh_only=True) == [diagnostic]


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
async def test_unversioned_push_uses_rolling_target_path_quiet_period(
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
    quiet_period = 0.60

    await _open_without_server(client, target, monkeypatch)
    client._handle_publish_diagnostics({"uri": target_uri, "diagnostics": [stale]})
    baseline = client._push_counter
    target.write_text("const value = 2;\n")
    version = await client.open_file(str(target), language_id="typescript")
    monkeypatch.setattr("agent.lsp.client.UNVERSIONED_PUSH_DEBOUNCE", quiet_period)
    monkeypatch.setattr("agent.lsp.client.UNVERSIONED_PUSH_STABILIZATION", 2.0)
    waiter = asyncio.create_task(
        client._wait_for_fresh_push(
            target_key, version=version, timeout=2.0, baseline=baseline
        )
    )

    client._handle_publish_diagnostics({"uri": file_uri(str(other)), "diagnostics": []})
    await asyncio.sleep(0.05)
    assert not waiter.done()

    client._handle_publish_diagnostics({"uri": target_uri, "diagnostics": [stale]})
    await asyncio.sleep(0.40)
    client._handle_publish_diagnostics({"uri": target_uri, "diagnostics": []})
    await asyncio.sleep(0.35)
    assert not waiter.done(), "the latest target push must reset quiet settling"

    client._handle_publish_diagnostics({"uri": file_uri(str(other)), "diagnostics": []})
    await asyncio.wait_for(waiter, timeout=1.5)
    assert client.diagnostics_for(str(target)) == []


@pytest.mark.asyncio
async def test_unversioned_push_waits_for_slower_corrected_snapshot(
    tmp_path: Path,
):
    """A stale unversioned push must not win over a correction 250ms later."""
    path = tmp_path / "index.ts"
    path.write_text("const value = 1;\n")
    client = _client(tmp_path, "stale_then_corrected_unversioned_push")
    await client.start()
    try:
        version_zero = await client.open_file(str(path), language_id="typescript")
        assert await client.wait_for_diagnostics(str(path), version_zero, timeout=1.5)
        assert client.diagnostics_for(str(path), fresh_only=True)

        path.write_text("const value = 2;\n")
        version_one = await client.open_file(str(path), language_id="typescript")
        wait_started = asyncio.get_running_loop().time()
        assert await client.wait_for_diagnostics(str(path), version_one, timeout=5.0)
        elapsed = asyncio.get_running_loop().time() - wait_started
        assert elapsed < NO_FULL_WAIT_CEILING
        assert client.diagnostics_for(str(path), fresh_only=True) == []
    finally:
        await client.shutdown()


@pytest.mark.asyncio
async def test_versioned_push_reclassifies_when_unversioned_correction_follows(
    tmp_path: Path,
):
    """The latest target push must select the active settling policy."""
    path = tmp_path / "index.ts"
    path.write_text("const value = 1;\n")
    client = _client(tmp_path, "versioned_then_stale_corrected_unversioned_push")
    await client.start()
    try:
        version_zero = await client.open_file(str(path), language_id="typescript")
        assert await client.wait_for_diagnostics(str(path), version_zero, timeout=1.5)
        assert client.diagnostics_for(str(path), fresh_only=True)

        path.write_text("const value = 2;\n")
        version_one = await client.open_file(str(path), language_id="typescript")
        assert await client.wait_for_diagnostics(str(path), version_one, timeout=2.0)
        assert client.diagnostics_for(str(path), fresh_only=True) == []
    finally:
        await client.shutdown()


@pytest.mark.asyncio
async def test_stable_unversioned_push_does_not_spend_large_caller_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    path = tmp_path / "index.ts"
    path.write_text("const value = 1;\n")
    client = _client(tmp_path, "clean")
    version = await _open_without_server(client, path, monkeypatch)
    target_uri = file_uri(str(path))
    target_key = uri_to_path(target_uri)
    baseline = client._push_counter

    waiter = asyncio.create_task(
        client._wait_for_fresh_push(
            target_key, version=version, timeout=5.0, baseline=baseline
        )
    )
    await asyncio.sleep(0.01)
    client._handle_publish_diagnostics({"uri": target_uri, "diagnostics": []})
    push_time = asyncio.get_running_loop().time()
    await asyncio.wait_for(waiter, timeout=NO_FULL_WAIT_CEILING)

    elapsed_after_push = asyncio.get_running_loop().time() - push_time
    assert elapsed_after_push >= UNVERSIONED_PUSH_DEBOUNCE - EARLY_TIMING_SLACK


@pytest.mark.asyncio
async def test_versioned_push_keeps_shorter_debounce(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    path = tmp_path / "index.ts"
    path.write_text("const value = 1;\n")
    client = _client(tmp_path, "clean")
    version = await _open_without_server(client, path, monkeypatch)
    target_uri = file_uri(str(path))
    target_key = uri_to_path(target_uri)
    baseline = client._push_counter

    waiter = asyncio.create_task(
        client._wait_for_fresh_push(
            target_key, version=version, timeout=2.0, baseline=baseline
        )
    )
    await asyncio.sleep(0.01)
    client._handle_publish_diagnostics({
        "uri": target_uri,
        "version": version,
        "diagnostics": [],
    })
    push_time = asyncio.get_running_loop().time()
    await asyncio.wait_for(waiter, timeout=1.5)

    elapsed_after_push = asyncio.get_running_loop().time() - push_time
    assert elapsed_after_push >= PUSH_DEBOUNCE - EARLY_TIMING_SLACK


@pytest.mark.asyncio
async def test_unversioned_push_reclassifies_when_fresh_versioned_push_follows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    path = tmp_path / "index.ts"
    path.write_text("const value = 1;\n")
    client = _client(tmp_path, "clean")
    version = await _open_without_server(client, path, monkeypatch)
    target_uri = file_uri(str(path))
    target_key = uri_to_path(target_uri)
    baseline = client._push_counter

    waiter = asyncio.create_task(
        client._wait_for_fresh_push(
            target_key, version=version, timeout=2.0, baseline=baseline
        )
    )
    client._handle_publish_diagnostics({
        "uri": target_uri,
        "diagnostics": [{"message": "unversioned result"}],
    })
    await asyncio.sleep(0.05)
    assert not waiter.done()
    client._handle_publish_diagnostics({
        "uri": target_uri,
        "version": version,
        "diagnostics": [],
    })
    versioned_push_time = asyncio.get_running_loop().time()
    await asyncio.wait_for(waiter, timeout=1.5)

    elapsed_after_versioned_push = (
        asyncio.get_running_loop().time() - versioned_push_time
    )
    assert elapsed_after_versioned_push >= PUSH_DEBOUNCE - EARLY_TIMING_SLACK
    assert client.diagnostics_for(str(path), fresh_only=True) == []


@pytest.mark.asyncio
async def test_same_version_wait_can_be_repeated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    path = tmp_path / "index.ts"
    path.write_text("const value = 1;\n")
    client = _client(tmp_path, "clean")
    version = await _open_without_server(client, path, monkeypatch)
    diagnostic = {"message": "versioned result"}

    async def unsupported_pull(_path: str, _version: int) -> None:
        return None

    monkeypatch.setattr(client, "_pull_document_diagnostics", unsupported_pull)
    client._handle_publish_diagnostics({
        "uri": file_uri(str(path)),
        "version": version,
        "diagnostics": [diagnostic],
    })

    assert await client.wait_for_diagnostics(str(path), version, timeout=1.0)
    assert await client.wait_for_diagnostics(str(path), version, timeout=1.0)
    assert client.diagnostics_for(str(path), fresh_only=True) == [diagnostic]


@pytest.mark.asyncio
async def test_cancelled_wait_preserves_same_version_baseline_for_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    path = tmp_path / "index.ts"
    path.write_text("const value = 1;\n")
    client = _client(tmp_path, "clean")
    version = await _open_without_server(client, path, monkeypatch)
    pull_started = asyncio.Event()

    async def blocking_pull(_path: str, _version: int) -> None:
        pull_started.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(client, "_pull_document_diagnostics", blocking_pull)
    first_wait = asyncio.create_task(
        client.wait_for_diagnostics(str(path), version, timeout=1.0)
    )
    await asyncio.wait_for(pull_started.wait(), timeout=1.0)
    first_wait.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first_wait

    diagnostic = {"message": "result received after cancellation"}
    client._handle_publish_diagnostics({
        "uri": file_uri(str(path)),
        "version": version,
        "diagnostics": [diagnostic],
    })

    assert await client.wait_for_diagnostics(str(path), version, timeout=1.0)
    assert client.diagnostics_for(str(path), fresh_only=True) == [diagnostic]
