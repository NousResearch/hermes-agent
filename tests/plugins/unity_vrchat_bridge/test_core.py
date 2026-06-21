from __future__ import annotations

import json
import tarfile
import threading
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from plugins import unity_vrchat_bridge
from plugins.unity_vrchat_bridge import core


def make_unity_project(root: Path, unity_version: str = "2022.3.22f1") -> Path:
    (root / "Assets").mkdir(parents=True)
    (root / "ProjectSettings").mkdir()
    (root / "Packages").mkdir()
    (root / "ProjectSettings" / "ProjectVersion.txt").write_text(
        f"m_EditorVersion: {unity_version}\n",
        encoding="utf-8",
    )
    (root / "Packages" / "manifest.json").write_text(
        json.dumps(
            {
                "dependencies": {
                    "com.vrchat.base": "3.10.4",
                    "com.vrchat.avatars": "3.10.4",
                    "jp.lilxyzw.liltoon": "1.9.0",
                    "nadena.dev.modular-avatar": "1.12.0",
                }
            }
        ),
        encoding="utf-8",
    )
    (root / "vpm-manifest.json").write_text(
        json.dumps(
            {
                "locked": {
                    "com.vrchat.avatars": {"version": "3.10.4"},
                    "jp.lilxyzw.liltoon": {"version": "1.9.0"},
                }
            }
        ),
        encoding="utf-8",
    )
    return root


def test_project_profile_detects_vrchat_avatar_project(tmp_path: Path) -> None:
    project = make_unity_project(tmp_path / "AvatarProject")

    profile = core.project_profile(str(project))

    assert profile["ok"] is True
    assert profile["unity_project"] is True
    assert profile["vrchat_project_kind"] == "avatar"
    assert profile["vrchat_unity_version_match"] is True
    assert {pkg["id"] for pkg in profile["detected_packages"]} >= {
        "com.vrchat.avatars",
        "jp.lilxyzw.liltoon",
        "nadena.dev.modular-avatar",
    }


def test_vrchat_project_health_blocks_upload_and_import(tmp_path: Path) -> None:
    project = make_unity_project(tmp_path / "AvatarProject", unity_version="2021.3.1f1")

    health = core.vrchat_project_health(str(project))

    assert "sdk_upload" in health["blocked_actions"]
    assert "package_import" in health["blocked_actions"]
    assert health["dry_run_required"] is True
    assert any("expected 2022.3.22f1" in warning for warning in health["warnings"])


def test_plugin_register_exposes_plan_apply_tool() -> None:
    calls: dict[str, list[str]] = {"tools": [], "commands": [], "cli": []}

    class Context:
        def register_tool(self, **kwargs: object) -> None:
            calls["tools"].append(str(kwargs["name"]))

        def register_command(self, name: str, **_: object) -> None:
            calls["commands"].append(name)

        def register_cli_command(self, name: str, **_: object) -> None:
            calls["cli"].append(name)

    unity_vrchat_bridge.register(Context())

    assert "unity_bridge_plan_apply" in calls["tools"]
    assert "unity_bridge_operation_plan" in calls["tools"]
    assert "unity_bridge_asset_info" in calls["tools"]
    assert "unity_bridge_scene_hierarchy" in calls["tools"]
    assert "unity-vrchat-bridge" in calls["commands"]
    assert "unity-vrchat-bridge" in calls["cli"]


def test_bridge_health_reports_missing_session(tmp_path: Path) -> None:
    project = make_unity_project(tmp_path / "AvatarProject")

    payload = json.loads(core.handle_health({"project_path": str(project)}))

    assert payload["ok"] is False
    assert "session" in payload["error"]


def test_menu_execute_rejects_live_before_http(tmp_path: Path) -> None:
    project = make_unity_project(tmp_path / "AvatarProject")

    payload = json.loads(
        core.handle_menu_execute(
            {
                "project_path": str(project),
                "menu_path": "VRChat SDK/Show Control Panel",
                "dry_run": False,
            }
        )
    )

    assert payload["ok"] is False
    assert payload["blocked_action"] == "menu_execute_live"


def test_plan_apply_rejects_live_before_http(tmp_path: Path) -> None:
    project = make_unity_project(tmp_path / "AvatarProject")

    payload = json.loads(
        core.handle_plan_apply(
            {
                "project_path": str(project),
                "operation": "avatar_preflight",
                "dry_run": False,
            }
        )
    )

    assert payload["ok"] is False
    assert payload["blocked_action"] == "plan_apply_live"


def test_asset_search_posts_json_to_bridge(tmp_path: Path) -> None:
    project = make_unity_project(tmp_path / "AvatarProject")
    seen: dict[str, object] = {}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            seen["path"] = self.path
            seen["token"] = self.headers.get("X-Hermes-Bridge-Token")
            size = int(self.headers.get("Content-Length") or "0")
            seen["body"] = json.loads(self.rfile.read(size).decode("utf-8"))
            data = b'{"ok":true,"assets":[{"path":"Assets/A.prefab"}],"truncated":false}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, format: str, *args: object) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        session_dir = project / "Library" / "HermesUnityBridge"
        session_dir.mkdir(parents=True)
        (session_dir / "session.json").write_text(
            json.dumps(
                {
                    "port": server.server_port,
                    "token": "secret",
                    "projectHash": core._project_hash(project),
                }
            ),
            encoding="utf-8",
        )

        payload = json.loads(
            core.handle_asset_search(
                {
                    "project_path": str(project),
                    "filter": "t:Prefab",
                    "folders": ["Assets"],
                    "limit": 7,
                }
            )
        )
    finally:
        server.shutdown()
        server.server_close()

    assert payload["ok"] is True
    assert seen["path"] == "/assets/search"
    assert seen["token"] == "secret"
    assert seen["body"] == {"filter": "t:Prefab", "folders": ["Assets"], "limit": 7}


def test_asset_info_posts_paths_to_bridge(tmp_path: Path) -> None:
    project = make_unity_project(tmp_path / "AvatarProject")
    seen: dict[str, object] = {}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            seen["path"] = self.path
            size = int(self.headers.get("Content-Length") or "0")
            seen["body"] = json.loads(self.rfile.read(size).decode("utf-8"))
            data = b'{"ok":true,"assets":[{"path":"Assets/A.prefab","exists":true}]}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, format: str, *args: object) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        session_dir = project / "Library" / "HermesUnityBridge"
        session_dir.mkdir(parents=True)
        (session_dir / "session.json").write_text(
            json.dumps(
                {
                    "port": server.server_port,
                    "token": "secret",
                    "projectHash": core._project_hash(project),
                }
            ),
            encoding="utf-8",
        )

        payload = json.loads(
            core.handle_asset_info(
                {
                    "project_path": str(project),
                    "paths": ["Assets/A.prefab"],
                    "include_dependencies": True,
                }
            )
        )
    finally:
        server.shutdown()
        server.server_close()

    assert payload["ok"] is True
    assert seen["path"] == "/assets/info"
    assert seen["body"] == {"paths": ["Assets/A.prefab"], "includeDependencies": True}


def test_operation_plan_rejects_live_before_http(tmp_path: Path) -> None:
    project = make_unity_project(tmp_path / "AvatarProject")

    payload = json.loads(
        core.handle_operation_plan(
            {
                "project_path": str(project),
                "operation": "asset_create",
                "target_path": "Assets/New.asset",
                "dry_run": False,
            }
        )
    )

    assert payload["ok"] is False
    assert payload["blocked_action"] == "operation_execute_live"


def test_plan_apply_posts_dry_run_to_bridge(tmp_path: Path) -> None:
    project = make_unity_project(tmp_path / "AvatarProject")
    seen: dict[str, object] = {}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            seen["path"] = self.path
            seen["token"] = self.headers.get("X-Hermes-Bridge-Token")
            size = int(self.headers.get("Content-Length") or "0")
            seen["body"] = json.loads(self.rfile.read(size).decode("utf-8"))
            data = b'{"ok":true,"dryRun":true,"willApply":false}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, format: str, *args: object) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        session_dir = project / "Library" / "HermesUnityBridge"
        session_dir.mkdir(parents=True)
        (session_dir / "session.json").write_text(
            json.dumps(
                {
                    "port": server.server_port,
                    "token": "secret",
                    "projectHash": core._project_hash(project),
                }
            ),
            encoding="utf-8",
        )

        payload = json.loads(
            core.handle_plan_apply(
                {
                    "project_path": str(project),
                    "operation": "avatar_preflight",
                }
            )
        )
    finally:
        server.shutdown()
        server.server_close()

    assert payload["ok"] is True
    assert payload["willApply"] is False
    assert seen["path"] == "/plan/apply"
    assert seen["token"] == "secret"
    assert seen["body"] == {"operation": "avatar_preflight", "dryRun": True}


def test_commercial_asset_zip_inspection_is_read_only(tmp_path: Path) -> None:
    project = make_unity_project(tmp_path / "AvatarProject")
    archive = tmp_path / "booth_avatar.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("Avatar/README.txt", "local terms")
        zf.writestr("Avatar/Materials/body_liltoon.mat", "%YAML")
        zf.writestr("Avatar/Prefab/avatar.prefab", "%YAML")

    payload = core.inspect_commercial_asset_archive(str(archive), str(project))

    assert payload["ok"] is True
    assert payload["archive_type"] == "zip"
    assert payload["will_modify_files"] is False
    assert "automatic_import" in payload["blocked_actions"]
    assert payload["kinds"]["prefab"] == 1
    assert any(dep["name"] == "lilToon" and dep["status"] == "installed" for dep in payload["dependencies"])


def test_commercial_asset_unitypackage_without_meta_warns(tmp_path: Path) -> None:
    archive = tmp_path / "outfit.unitypackage"
    payload_file = tmp_path / "asset"
    payload_file.write_text("payload", encoding="utf-8")
    with tarfile.open(archive, "w") as tf:
        tf.add(payload_file, arcname="guid/asset")
        tf.add(payload_file, arcname="guid/pathname")

    payload = core.inspect_commercial_asset_archive(str(archive))

    assert payload["ok"] is True
    assert payload["archive_type"] == "unitypackage"
    assert any(".meta" in risk for risk in payload["risks"])
