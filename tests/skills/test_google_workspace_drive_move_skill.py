"""Behavior tests for the bundled Google Drive move command."""

import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest


API_PATH = (
    Path(__file__).resolve().parents[2]
    / "skills/productivity/google-workspace/scripts/google_api.py"
)
FOLDER_MIME = "application/vnd.google-apps.folder"


@pytest.fixture
def api_module(monkeypatch, tmp_path):
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    spec = importlib.util.spec_from_file_location("local_google_api_test", API_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    module._gws_binary = lambda: "/usr/bin/gws"
    module._ensure_authenticated = lambda: None
    return module


def test_move_defaults_to_preview_without_updating(api_module, capsys):
    calls = []

    def fake_gws(parts, *, params=None, body=None):
        calls.append((parts, params, body))
        if parts[-1] == "get" and params["fileId"] == "file-1":
            return {
                "id": "file-1",
                "name": "Report.pdf",
                "mimeType": "application/pdf",
                "parents": ["old-folder"],
                "webViewLink": "https://drive.example/file-1",
            }
        if parts[-1] == "get" and params["fileId"] == "new-folder":
            return {
                "id": "new-folder",
                "name": "Archive",
                "mimeType": FOLDER_MIME,
                "parents": ["root"],
                "webViewLink": "https://drive.example/new-folder",
            }
        if parts[-1] == "list":
            return {"files": [{"id": "duplicate", "name": "Report.pdf"}]}
        raise AssertionError((parts, params, body))

    api_module._run_gws = fake_gws
    args = argparse.Namespace(
        file_id="file-1",
        destination_id="new-folder",
        execute=False,
        allow_cross_drive=False,
    )

    api_module.drive_move(args)

    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "preview"
    assert result["file"] == {"id": "file-1", "name": "Report.pdf"}
    assert result["from"] == {"id": "old-folder"}
    assert result["to"] == {"id": "new-folder", "name": "Archive"}
    assert result["duplicateNameWarning"][0]["id"] == "duplicate"
    assert result["requiresConfirmation"] is True
    assert not any(parts[-1] == "update" for parts, _, _ in calls)


def test_multiple_current_parents_are_rejected_before_lookup_or_write(api_module):
    calls = []

    def fake_gws(parts, *, params=None, body=None):
        calls.append((parts, params, body))
        if parts[-1] == "get" and params["fileId"] == "file-1":
            return {
                "id": "file-1",
                "name": "Report.pdf",
                "mimeType": "application/pdf",
                "parents": ["old-folder-1", "old-folder-2"],
            }
        if parts[-1] == "get" and params["fileId"] == "new-folder":
            return {
                "id": "new-folder",
                "name": "Archive",
                "mimeType": FOLDER_MIME,
                "parents": ["root"],
            }
        if parts[-1] == "list":
            return {"files": []}
        raise AssertionError((parts, params, body))

    api_module._run_gws = fake_gws
    args = argparse.Namespace(
        file_id="file-1",
        destination_id="new-folder",
        execute=False,
        allow_cross_drive=False,
    )

    with pytest.raises(
        SystemExit,
        match="Drive v3 supports a single parent; manual handling is required",
    ):
        api_module.drive_move(args)

    assert not any(parts[-1] in {"list", "update"} for parts, _, _ in calls)


def test_execute_moves_once_and_verifies_parent(api_module, capsys):
    calls = []
    source_reads = 0

    def fake_gws(parts, *, params=None, body=None):
        nonlocal source_reads
        calls.append((parts, params, body))
        if parts[-1] == "get" and params["fileId"] == "file-1":
            source_reads += 1
            return {
                "id": "file-1",
                "name": "Report.pdf",
                "mimeType": "application/pdf",
                "parents": ["old-folder"] if source_reads == 1 else ["new-folder"],
            }
        if parts[-1] == "get" and params["fileId"] == "new-folder":
            return {
                "id": "new-folder",
                "name": "Archive",
                "mimeType": FOLDER_MIME,
                "parents": ["root"],
            }
        if parts[-1] == "list":
            return {"files": [{"id": "duplicate", "name": "Report.pdf"}]}
        if parts[-1] == "update":
            return {"id": "file-1", "parents": ["new-folder"]}
        raise AssertionError((parts, params, body))

    api_module._run_gws = fake_gws
    args = argparse.Namespace(
        file_id="file-1",
        destination_id="new-folder",
        execute=True,
        allow_cross_drive=False,
    )

    api_module.drive_move(args)

    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "moved"
    assert result["verified"] is True
    assert result["duplicateNameWarning"] == [{"id": "duplicate", "name": "Report.pdf"}]
    assert result["from"] == {"id": "old-folder"}
    assert result["to"] == {"id": "new-folder", "name": "Archive"}
    assert result["rollback"] == {
        "fileId": "file-1",
        "to": "old-folder",
    }
    updates = [(params, body) for parts, params, body in calls if parts[-1] == "update"]
    assert updates == [({
        "fileId": "file-1",
        "addParents": "new-folder",
        "removeParents": "old-folder",
        "supportsAllDrives": True,
        "fields": api_module._DRIVE_MOVE_FIELDS,
    }, None)]
    assert source_reads == 2


def test_gws_parentless_source_omits_remove_parents(api_module, capsys):
    calls = []
    source_reads = 0

    def fake_gws(parts, *, params=None, body=None):
        nonlocal source_reads
        calls.append((parts, params, body))
        if parts[-1] == "get" and params["fileId"] == "file-1":
            source_reads += 1
            return {
                "id": "file-1",
                "name": "Report.pdf",
                "mimeType": "application/pdf",
                **({} if source_reads == 1 else {"parents": ["new-folder"]}),
            }
        if parts[-1] == "get" and params["fileId"] == "new-folder":
            return {
                "id": "new-folder",
                "name": "Archive",
                "mimeType": FOLDER_MIME,
                "parents": ["root"],
            }
        if parts[-1] == "list":
            return {"files": []}
        if parts[-1] == "update":
            return {"id": "file-1", "parents": ["new-folder"]}
        raise AssertionError((parts, params, body))

    api_module._run_gws = fake_gws
    args = argparse.Namespace(
        file_id="file-1",
        destination_id="new-folder",
        execute=True,
        allow_cross_drive=False,
    )

    api_module.drive_move(args)

    result = json.loads(capsys.readouterr().out)
    assert result["verified"] is True
    assert result["from"] is None
    assert result["rollback"] is None
    update = next(params for parts, params, _ in calls if parts[-1] == "update")
    assert "removeParents" not in update
    assert update["addParents"] == "new-folder"


def test_cross_drive_execute_requires_separate_override(api_module):
    calls = []

    def fake_gws(parts, *, params=None, body=None):
        calls.append((parts, params, body))
        if parts[-1] == "get" and params["fileId"] == "file-1":
            return {
                "id": "file-1",
                "name": "Report.pdf",
                "mimeType": "application/pdf",
                "parents": ["old-folder"],
            }
        if parts[-1] == "get" and params["fileId"] == "shared-folder":
            return {
                "id": "shared-folder",
                "name": "Shared Archive",
                "mimeType": FOLDER_MIME,
                "parents": ["shared-root"],
                "driveId": "shared-drive-1",
            }
        if parts[-1] == "list":
            return {"files": []}
        if parts[-1] == "update":
            return {"id": "file-1", "parents": ["shared-folder"]}
        raise AssertionError((parts, params, body))

    api_module._run_gws = fake_gws
    args = argparse.Namespace(
        file_id="file-1",
        destination_id="shared-folder",
        execute=True,
        allow_cross_drive=False,
    )

    with pytest.raises(SystemExit, match="--allow-cross-drive"):
        api_module.drive_move(args)

    assert not any(parts[-1] == "update" for parts, _, _ in calls)


def test_cross_drive_override_allows_verified_move(api_module, capsys):
    calls = []
    source_reads = 0

    def fake_gws(parts, *, params=None, body=None):
        nonlocal source_reads
        calls.append((parts, params, body))
        if parts[-1] == "get" and params["fileId"] == "file-1":
            source_reads += 1
            return {
                "id": "file-1",
                "name": "Report.pdf",
                "mimeType": "application/pdf",
                "parents": ["old-folder"] if source_reads == 1 else ["shared-folder"],
                **({} if source_reads == 1 else {"driveId": "shared-drive-1"}),
            }
        if parts[-1] == "get" and params["fileId"] == "shared-folder":
            return {
                "id": "shared-folder",
                "name": "Shared Archive",
                "mimeType": FOLDER_MIME,
                "parents": ["shared-root"],
                "driveId": "shared-drive-1",
            }
        if parts[-1] == "list":
            return {"files": []}
        if parts[-1] == "update":
            return {"id": "file-1", "parents": ["shared-folder"]}
        raise AssertionError((parts, params, body))

    api_module._run_gws = fake_gws
    args = argparse.Namespace(
        file_id="file-1",
        destination_id="shared-folder",
        execute=True,
        allow_cross_drive=True,
    )

    api_module.drive_move(args)

    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "moved"
    assert result["verified"] is True
    assert result["crossDrive"] is True
    assert sum(parts[-1] == "update" for parts, _, _ in calls) == 1


def test_cross_drive_override_allows_shared_drive_to_my_drive(api_module, capsys):
    calls = []
    source_reads = 0

    def fake_gws(parts, *, params=None, body=None):
        nonlocal source_reads
        assert params is not None
        calls.append((parts, params, body))
        if parts[-1] == "get" and params["fileId"] == "file-1":
            source_reads += 1
            return {
                "id": "file-1",
                "name": "Report.pdf",
                "mimeType": "application/pdf",
                "parents": ["shared-folder"] if source_reads == 1 else ["my-folder"],
                **({"driveId": "shared-drive-1"} if source_reads == 1 else {}),
            }
        if parts[-1] == "get" and params["fileId"] == "my-folder":
            return {
                "id": "my-folder",
                "name": "My Drive Archive",
                "mimeType": FOLDER_MIME,
                "parents": ["root"],
            }
        if parts[-1] == "list":
            return {"files": []}
        if parts[-1] == "update":
            return {"id": "file-1", "parents": ["my-folder"]}
        raise AssertionError((parts, params, body))

    api_module._run_gws = fake_gws
    args = argparse.Namespace(
        file_id="file-1",
        destination_id="my-folder",
        execute=True,
        allow_cross_drive=True,
    )

    api_module.drive_move(args)

    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "moved"
    assert result["verified"] is True
    assert result["crossDrive"] is True
    assert sum(parts[-1] == "update" for parts, _, _ in calls) == 1


def test_same_folder_is_verified_noop(api_module, capsys):
    calls = []

    def fake_gws(parts, *, params=None, body=None):
        calls.append((parts, params, body))
        if parts[-1] == "get" and params["fileId"] == "file-1":
            return {
                "id": "file-1",
                "name": "Report.pdf",
                "mimeType": "application/pdf",
                "parents": ["folder-1"],
            }
        if parts[-1] == "get" and params["fileId"] == "folder-1":
            return {
                "id": "folder-1",
                "name": "Current",
                "mimeType": FOLDER_MIME,
                "parents": ["root"],
            }
        if parts[-1] == "list":
            return {"files": []}
        if parts[-1] == "update":
            return {"id": "file-1", "parents": ["folder-1"]}
        raise AssertionError((parts, params, body))

    api_module._run_gws = fake_gws
    args = argparse.Namespace(
        file_id="file-1",
        destination_id="folder-1",
        execute=True,
        allow_cross_drive=False,
    )

    api_module.drive_move(args)

    result = json.loads(capsys.readouterr().out)
    assert result == {
        "status": "unchanged",
        "verified": True,
        "file": {"id": "file-1", "name": "Report.pdf"},
        "parent": {"id": "folder-1", "name": "Current"},
    }
    assert not any(parts[-1] in {"list", "update"} for parts, _, _ in calls)


def test_non_folder_destination_is_rejected_before_write(api_module):
    calls = []

    def fake_gws(parts, *, params=None, body=None):
        calls.append((parts, params, body))
        if parts[-1] == "get" and params["fileId"] == "file-1":
            return {
                "id": "file-1",
                "name": "Report.pdf",
                "mimeType": "application/pdf",
                "parents": ["folder-1"],
            }
        if parts[-1] == "get" and params["fileId"] == "not-a-folder":
            return {
                "id": "not-a-folder",
                "name": "Other.pdf",
                "mimeType": "application/pdf",
                "parents": ["root"],
            }
        raise AssertionError((parts, params, body))

    api_module._run_gws = fake_gws
    args = argparse.Namespace(
        file_id="file-1",
        destination_id="not-a-folder",
        execute=True,
        allow_cross_drive=False,
    )

    with pytest.raises(SystemExit, match="not a Google Drive folder"):
        api_module.drive_move(args)

    assert not any(parts[-1] in {"list", "update"} for parts, _, _ in calls)


def test_folder_cannot_be_moved_into_itself(api_module):
    api_module._run_gws = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("No Drive API call should happen for a self move")
    )
    args = argparse.Namespace(
        file_id="folder-1",
        destination_id="folder-1",
        execute=False,
        allow_cross_drive=False,
    )

    with pytest.raises(SystemExit, match="cannot be moved into itself"):
        api_module.drive_move(args)


def test_folder_cannot_be_moved_into_descendant_before_lookup_or_write(api_module):
    calls = []

    def fake_gws(parts, *, params=None, body=None):
        calls.append((parts, params, body))
        if parts[-1] == "get":
            items = {
                "folder-1": {
                    "id": "folder-1",
                    "name": "Parent",
                    "mimeType": FOLDER_MIME,
                    "parents": ["old-folder"],
                },
                "child-2": {
                    "id": "child-2",
                    "name": "Grandchild",
                    "mimeType": FOLDER_MIME,
                    "parents": ["child-1"],
                },
                "child-1": {
                    "id": "child-1",
                    "name": "Child",
                    "mimeType": FOLDER_MIME,
                    "parents": ["folder-1"],
                },
            }
            return items[params["fileId"]]
        if parts[-1] == "list":
            return {"files": []}
        raise AssertionError((parts, params, body))

    api_module._run_gws = fake_gws
    args = argparse.Namespace(
        file_id="folder-1",
        destination_id="child-2",
        execute=False,
        allow_cross_drive=False,
    )

    with pytest.raises(SystemExit, match="cannot be moved into its descendant"):
        api_module.drive_move(args)

    assert not any(parts[-1] in {"list", "update"} for parts, _, _ in calls)


def test_ordinary_folder_move_preview_walks_to_root(api_module, capsys):
    calls = []

    def fake_gws(parts, *, params=None, body=None):
        calls.append((parts, params, body))
        if parts[-1] == "get":
            items = {
                "folder-1": {
                    "id": "folder-1",
                    "name": "Project",
                    "mimeType": FOLDER_MIME,
                    "parents": ["old-folder"],
                },
                "new-folder": {
                    "id": "new-folder",
                    "name": "Archive",
                    "mimeType": FOLDER_MIME,
                    "parents": ["root"],
                },
                "root": {
                    "id": "root",
                    "name": "My Drive",
                    "mimeType": FOLDER_MIME,
                },
            }
            return items[params["fileId"]]
        if parts[-1] == "list":
            return {"files": []}
        raise AssertionError((parts, params, body))

    api_module._run_gws = fake_gws
    args = argparse.Namespace(
        file_id="folder-1",
        destination_id="new-folder",
        execute=False,
        allow_cross_drive=False,
    )

    api_module.drive_move(args)

    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "preview"
    assert result["file"] == {"id": "folder-1", "name": "Project"}
    assert result["to"] == {"id": "new-folder", "name": "Archive"}
    ancestry_reads = [
        params["fileId"]
        for parts, params, _ in calls
        if parts[-1] == "get"
    ]
    assert ancestry_reads == ["folder-1", "new-folder", "root"]
    assert not any(parts[-1] == "update" for parts, _, _ in calls)


def test_folder_ancestry_cycle_fails_loudly_before_lookup_or_write(api_module):
    calls = []

    def fake_gws(parts, *, params=None, body=None):
        calls.append((parts, params, body))
        if parts[-1] == "get":
            items = {
                "folder-1": {
                    "id": "folder-1",
                    "name": "Project",
                    "mimeType": FOLDER_MIME,
                    "parents": ["old-folder"],
                },
                "child-a": {
                    "id": "child-a",
                    "name": "Child A",
                    "mimeType": FOLDER_MIME,
                    "parents": ["child-b"],
                },
                "child-b": {
                    "id": "child-b",
                    "name": "Child B",
                    "mimeType": FOLDER_MIME,
                    "parents": ["child-a"],
                },
            }
            return items[params["fileId"]]
        if parts[-1] == "list":
            return {"files": []}
        raise AssertionError((parts, params, body))

    api_module._run_gws = fake_gws
    args = argparse.Namespace(
        file_id="folder-1",
        destination_id="child-a",
        execute=False,
        allow_cross_drive=False,
    )

    with pytest.raises(SystemExit, match="ancestry contains a cycle"):
        api_module.drive_move(args)

    assert not any(parts[-1] in {"list", "update"} for parts, _, _ in calls)


def test_folder_ancestry_with_multiple_parents_fails_loudly(api_module):
    calls = []

    def fake_gws(parts, *, params=None, body=None):
        calls.append((parts, params, body))
        if parts[-1] == "get":
            assert params is not None
            items = {
                "folder-1": {
                    "id": "folder-1",
                    "name": "Project",
                    "mimeType": FOLDER_MIME,
                    "parents": ["old-folder"],
                },
                "destination": {
                    "id": "destination",
                    "name": "Ambiguous",
                    "mimeType": FOLDER_MIME,
                    "parents": ["parent-a", "parent-b"],
                },
            }
            return items[params["fileId"]]
        raise AssertionError((parts, params, body))

    api_module._run_gws = fake_gws
    args = argparse.Namespace(
        file_id="folder-1",
        destination_id="destination",
        execute=False,
        allow_cross_drive=False,
    )

    with pytest.raises(SystemExit, match="ancestry contains multiple parents"):
        api_module.drive_move(args)

    assert [params["fileId"] for parts, params, _ in calls if parts[-1] == "get"] == [
        "folder-1",
        "destination",
    ]
    assert not any(parts[-1] in {"list", "update"} for parts, _, _ in calls)


@pytest.mark.parametrize(
    ("parent_hops", "should_fail"),
    [(100, False), (101, True)],
)
def test_folder_ancestry_enforces_drive_depth_boundary(
    api_module, capsys, parent_hops, should_fail
):
    calls = []
    chain = {
        f"node-{index}": {
            "id": f"node-{index}",
            "name": f"Node {index}",
            "mimeType": FOLDER_MIME,
            **({"parents": [f"node-{index + 1}"]} if index < parent_hops else {}),
        }
        for index in range(parent_hops + 1)
    }
    chain["folder-1"] = {
        "id": "folder-1",
        "name": "Project",
        "mimeType": FOLDER_MIME,
        "parents": ["old-folder"],
    }

    def fake_gws(parts, *, params=None, body=None):
        calls.append((parts, params, body))
        assert params is not None
        if parts[-1] == "get":
            return chain[params["fileId"]]
        if parts[-1] == "list":
            return {"files": []}
        raise AssertionError((parts, params, body))

    api_module._run_gws = fake_gws
    args = argparse.Namespace(
        file_id="folder-1",
        destination_id="node-0",
        execute=False,
        allow_cross_drive=False,
    )

    if should_fail:
        with pytest.raises(SystemExit, match="exceeded the safe traversal limit"):
            api_module.drive_move(args)
        assert not any(parts[-1] in {"list", "update"} for parts, _, _ in calls)
    else:
        api_module.drive_move(args)
        assert json.loads(capsys.readouterr().out)["status"] == "preview"
        assert sum(parts[-1] == "get" for parts, _, _ in calls) == parent_hops + 2


def test_cli_exposes_preview_execute_and_cross_drive_flags():
    result = subprocess.run(
        [sys.executable, str(API_PATH), "drive", "move", "--help"],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "FILE_ID" in result.stdout
    assert "--to DESTINATION_ID" in result.stdout
    assert "--execute" in result.stdout
    assert "--allow-cross-drive" in result.stdout


def test_gws_duplicate_lookup_paginates_and_deduplicates(api_module, capsys):
    list_calls = []

    def fake_gws(parts, *, params=None, body=None):
        if parts[-1] == "get" and params["fileId"] == "file-1":
            return {
                "id": "file-1",
                "name": "Report.pdf",
                "mimeType": "application/pdf",
                "parents": ["old-folder"],
            }
        if parts[-1] == "get" and params["fileId"] == "new-folder":
            return {
                "id": "new-folder",
                "name": "Archive",
                "mimeType": FOLDER_MIME,
                "parents": ["root"],
            }
        if parts[-1] == "list":
            list_calls.append(params)
            if "pageToken" not in params:
                return {
                    "files": [
                        {"id": "file-1", "name": "Report.pdf"},
                        {"id": "duplicate-1", "name": "Report.pdf"},
                    ],
                    "nextPageToken": "page-2",
                }
            assert params["pageToken"] == "page-2"
            return {
                "files": [
                    {"id": "duplicate-1", "name": "Report.pdf"},
                    {"id": "duplicate-2", "name": "Report.pdf"},
                ],
            }
        raise AssertionError((parts, params, body))

    api_module._run_gws = fake_gws
    args = argparse.Namespace(
        file_id="file-1",
        destination_id="new-folder",
        execute=False,
        allow_cross_drive=False,
    )

    api_module.drive_move(args)

    result = json.loads(capsys.readouterr().out)
    assert result["duplicateNameWarning"] == [
        {"id": "duplicate-1", "name": "Report.pdf"},
        {"id": "duplicate-2", "name": "Report.pdf"},
    ]
    assert len(list_calls) == 2
    assert "pageToken" not in list_calls[0]
    assert list_calls[1]["pageToken"] == "page-2"
    assert all("nextPageToken" in call["fields"] for call in list_calls)


def test_python_duplicate_lookup_paginates_and_deduplicates(api_module, capsys):
    api_module._gws_binary = lambda: None
    list_calls = []

    class Request:
        def __init__(self, payload):
            self.payload = payload

        def execute(self):
            return self.payload

    class Files:
        def get(self, **params):
            if params["fileId"] == "file-1":
                return Request({
                    "id": "file-1",
                    "name": "Report.pdf",
                    "mimeType": "application/pdf",
                    "parents": ["old-folder"],
                })
            return Request({
                "id": "new-folder",
                "name": "Archive",
                "mimeType": FOLDER_MIME,
                "parents": ["root"],
            })

        def list(self, **params):
            list_calls.append(params)
            if "pageToken" not in params:
                return Request({
                    "files": [
                        {"id": "file-1", "name": "Report.pdf"},
                        {"id": "duplicate-1", "name": "Report.pdf"},
                    ],
                    "nextPageToken": "page-2",
                })
            assert params["pageToken"] == "page-2"
            return Request({
                "files": [
                    {"id": "duplicate-1", "name": "Report.pdf"},
                    {"id": "duplicate-2", "name": "Report.pdf"},
                ],
            })

    class Service:
        def __init__(self):
            self._files = Files()

        def files(self):
            return self._files

    service = Service()
    api_module.build_service = lambda api, version: service
    args = argparse.Namespace(
        file_id="file-1",
        destination_id="new-folder",
        execute=False,
        allow_cross_drive=False,
    )

    api_module.drive_move(args)

    result = json.loads(capsys.readouterr().out)
    assert result["duplicateNameWarning"] == [
        {"id": "duplicate-1", "name": "Report.pdf"},
        {"id": "duplicate-2", "name": "Report.pdf"},
    ]
    assert len(list_calls) == 2
    assert "pageToken" not in list_calls[0]
    assert list_calls[1]["pageToken"] == "page-2"
    assert all("nextPageToken" in call["fields"] for call in list_calls)


def test_python_client_fallback_executes_and_verifies(api_module, capsys):
    api_module._gws_binary = lambda: None
    calls = []
    source_reads = 0

    class Request:
        def __init__(self, payload):
            self.payload = payload

        def execute(self):
            return self.payload

    class Files:
        def get(self, **params):
            nonlocal source_reads
            calls.append(("get", params))
            if params["fileId"] == "file-1":
                source_reads += 1
                return Request({
                    "id": "file-1",
                    "name": "Report.pdf",
                    "mimeType": "application/pdf",
                    "parents": ["old-folder"] if source_reads == 1 else ["new-folder"],
                })
            return Request({
                "id": "new-folder",
                "name": "Archive",
                "mimeType": FOLDER_MIME,
                "parents": ["root"],
            })

        def list(self, **params):
            calls.append(("list", params))
            return Request({"files": []})

        def update(self, **params):
            calls.append(("update", params))
            return Request({"id": "file-1", "parents": ["new-folder"]})

    class Service:
        def __init__(self):
            self._files = Files()

        def files(self):
            return self._files

    service = Service()
    api_module.build_service = lambda api, version: service
    args = argparse.Namespace(
        file_id="file-1",
        destination_id="new-folder",
        execute=True,
        allow_cross_drive=False,
    )

    api_module.drive_move(args)

    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "moved"
    assert result["verified"] is True
    update = next(params for name, params in calls if name == "update")
    assert update["addParents"] == "new-folder"
    assert update["removeParents"] == "old-folder"
    assert update["supportsAllDrives"] is True
    assert source_reads == 2


def test_python_parentless_source_omits_remove_parents(api_module, capsys):
    api_module._gws_binary = lambda: None
    calls = []
    source_reads = 0

    class Request:
        def __init__(self, payload):
            self.payload = payload

        def execute(self):
            return self.payload

    class Files:
        def get(self, **params):
            nonlocal source_reads
            if params["fileId"] == "file-1":
                source_reads += 1
                return Request({
                    "id": "file-1",
                    "name": "Report.pdf",
                    "mimeType": "application/pdf",
                    **({} if source_reads == 1 else {"parents": ["new-folder"]}),
                })
            return Request({
                "id": "new-folder",
                "name": "Archive",
                "mimeType": FOLDER_MIME,
                "parents": ["root"],
            })

        def list(self, **params):
            return Request({"files": []})

        def update(self, **params):
            calls.append(params)
            return Request({"id": "file-1", "parents": ["new-folder"]})

    class Service:
        def __init__(self):
            self._files = Files()

        def files(self):
            return self._files

    service = Service()
    api_module.build_service = lambda api, version: service
    args = argparse.Namespace(
        file_id="file-1",
        destination_id="new-folder",
        execute=True,
        allow_cross_drive=False,
    )

    api_module.drive_move(args)

    result = json.loads(capsys.readouterr().out)
    assert result["verified"] is True
    assert result["from"] is None
    assert result["rollback"] is None
    assert len(calls) == 1
    assert "removeParents" not in calls[0]
    assert calls[0]["addParents"] == "new-folder"


def test_execute_fails_loudly_when_parent_readback_disagrees(api_module):
    reads = iter([
        {
            "id": "file-1",
            "name": "Report.pdf",
            "mimeType": "application/pdf",
            "parents": ["old-folder"],
        },
        {
            "id": "new-folder",
            "name": "Archive",
            "mimeType": FOLDER_MIME,
            "parents": ["root"],
        },
        {
            "id": "file-1",
            "name": "Report.pdf",
            "mimeType": "application/pdf",
            "parents": ["old-folder"],
        },
    ])
    api_module._drive_move_get = lambda file_id: next(reads)
    api_module._drive_move_duplicates = lambda *args: []
    api_module._drive_move_update = lambda *args: {
        "id": "file-1",
        "parents": ["new-folder"],
    }
    args = argparse.Namespace(
        file_id="file-1",
        destination_id="new-folder",
        execute=True,
        allow_cross_drive=False,
    )

    with pytest.raises(SystemExit, match="parent verification failed"):
        api_module.drive_move(args)
