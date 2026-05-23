#!/usr/bin/env python3
"""Prove Hermes-backed tools can register inside jcode's native registry."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_JCODE = ROOT / ".codex-research" / "jcode"
DEFAULT_PATCH = ROOT / "patches" / "jcode" / "register-external-toolset.patch"
NATIVE_TOOL_DIR = ROOT / "bridges" / "jcode-native-hermes-tool"


REGISTRY_TEST = r'''use async_trait::async_trait;
use jcode::message::{Message, ToolDefinition};
use jcode::provider::{EventStream, Provider};
use jcode::tool::{Registry, ToolContext, ToolExecutionMode};
use jcode_native_hermes_tool::{default_hermes_toolset, HermesToolConfig};
use serde_json::json;
use std::sync::Arc;

struct MockProvider;

#[async_trait]
impl Provider for MockProvider {
    async fn complete(
        &self,
        _messages: &[Message],
        _tools: &[ToolDefinition],
        _system: &str,
        _resume_session_id: Option<&str>,
    ) -> anyhow::Result<EventStream> {
        Err(anyhow::anyhow!("registry smoke does not call the model"))
    }

    fn name(&self) -> &str {
        "mock"
    }

    fn fork(&self) -> Arc<dyn Provider> {
        Arc::new(MockProvider)
    }
}

fn fake_hermes_service_command() -> Vec<String> {
    let script = std::env::temp_dir().join(format!(
        "fake-hermes-service-{}-{}.py",
        std::process::id(),
        std::thread::current().name().unwrap_or("test")
    ));
    std::fs::write(
        &script,
        r#"import json
import sys

line = sys.stdin.readline().strip()
request = json.loads(line)
print(json.dumps({
    "type": "hermes_service_response",
    "contract_version": "hermes-service.v1",
    "id": request.get("id"),
    "ok": True,
    "tool": request.get("tool"),
    "result": {
        "executed_by": "fake_hermes_service",
        "tool": request.get("tool"),
        "args": request.get("args", {}),
    },
    "duration_ms": 1,
}))
"#,
    )
    .expect("write fake Hermes service");
    vec!["python3".to_string(), script.display().to_string()]
}

#[tokio::test]
async fn hermes_native_tools_register_and_execute_in_jcode_registry() {
    let provider: Arc<dyn Provider> = Arc::new(MockProvider);
    let registry = Registry::new(provider).await;
    let config = HermesToolConfig {
        service_command: fake_hermes_service_command(),
        timeout_ms: 1_000,
    };

    let registered = registry
        .register_toolset("hermes", default_hermes_toolset(config))
        .await;
    assert_eq!(
        registered,
        vec![
            "hermes_memory",
            "hermes_session_search",
            "hermes_web_extract",
            "hermes_web_search",
        ]
    );

    let names = registry.tool_names().await;
    assert!(names.contains(&"hermes_memory".to_string()));
    assert!(names.contains(&"hermes_session_search".to_string()));
    assert!(names.contains(&"hermes_web_search".to_string()));
    assert!(names.contains(&"hermes_web_extract".to_string()));

    let definitions = registry.definitions(None).await;
    let definition_names = definitions
        .iter()
        .map(|definition| definition.name.as_str())
        .collect::<Vec<_>>();
    assert!(definition_names.contains(&"hermes_memory"));
    assert!(definition_names.contains(&"hermes_session_search"));
    assert!(definition_names.contains(&"hermes_web_search"));
    assert!(definition_names.contains(&"hermes_web_extract"));

    let output = registry
        .execute(
            "hermes_web_search",
            json!({"query": "bridge", "limit": 1}),
            ToolContext {
                session_id: "session_smoke".to_string(),
                message_id: "message_smoke".to_string(),
                tool_call_id: "call_smoke".to_string(),
                working_dir: None,
                stdin_request_tx: None,
                graceful_shutdown_signal: None,
                execution_mode: ToolExecutionMode::Direct,
            },
        )
        .await
        .expect("execute Hermes-backed jcode tool");
    assert_eq!(output.title.as_deref(), Some("hermes:web_search"));
    assert!(output.output.contains("fake_hermes_service"));
    assert_eq!(
        output.metadata.as_ref().and_then(|value| value.get("tool")),
        Some(&json!("web_search"))
    );
}
'''


def _check(name: str, ok: bool, **details: Any) -> dict[str, Any]:
    result: dict[str, Any] = {"name": name, "ok": bool(ok)}
    result.update(details)
    return result


def _run(
    cmd: list[str],
    *,
    cwd: Path,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=str(cwd),
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def _copy_jcode(source: Path, destination: Path) -> None:
    shutil.copytree(
        source,
        destination,
        ignore=shutil.ignore_patterns(".git", "target", "graphify-out"),
    )


def _prepare_jcode_worktree(
    jcode_path: Path,
    temp_dir: Path,
) -> tuple[Path, bool, subprocess.CompletedProcess[str] | None]:
    worktree = temp_dir / "jcode"
    if (jcode_path / ".git").exists():
        completed = _run(
            [
                "git",
                "-C",
                str(jcode_path),
                "worktree",
                "add",
                "--detach",
                str(worktree),
                "HEAD",
            ],
            cwd=temp_dir,
        )
        if completed.returncode == 0:
            return worktree, True, completed
        _copy_jcode(jcode_path, worktree)
        return worktree, False, completed

    _copy_jcode(jcode_path, worktree)
    return worktree, False, None


def _cleanup_worktree(jcode_path: Path, worktree: Path, used_git_worktree: bool) -> None:
    if not used_git_worktree:
        return
    _run(
        ["git", "-C", str(jcode_path), "worktree", "remove", "--force", str(worktree)],
        cwd=jcode_path,
    )


def _copy_native_tool(worktree: Path) -> Path:
    destination = worktree / "bridges" / "jcode-native-hermes-tool"
    shutil.copytree(
        NATIVE_TOOL_DIR,
        destination,
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns("target", "__pycache__", "*.pyc"),
    )
    cargo_toml = destination / "Cargo.toml"
    text = cargo_toml.read_text(encoding="utf-8")
    text = text.replace(
        "../../upstreams/jcode/crates/jcode-tool-core",
        "../../crates/jcode-tool-core",
    )
    text = text.replace(
        "../../upstreams/jcode/crates/jcode-tool-types",
        "../../crates/jcode-tool-types",
    )
    cargo_toml.write_text(text, encoding="utf-8")
    return destination


def _add_jcode_dev_dependency(worktree: Path) -> None:
    cargo_toml = worktree / "Cargo.toml"
    text = cargo_toml.read_text(encoding="utf-8")
    dependency = (
        'jcode-native-hermes-tool = { path = "bridges/jcode-native-hermes-tool" }\n'
    )
    if dependency in text:
        return
    marker = "[dev-dependencies]\n"
    if marker not in text:
        text += f"\n{marker}"
    text = text.replace(marker, marker + dependency, 1)
    cargo_toml.write_text(text, encoding="utf-8")


def _write_registry_test(worktree: Path) -> Path:
    test_path = worktree / "tests" / "hermes_native_tool_registry.rs"
    test_path.parent.mkdir(parents=True, exist_ok=True)
    test_path.write_text(REGISTRY_TEST, encoding="utf-8")
    return test_path


def run_supertool_registry_smoke(
    jcode_path: Path,
    patch_path: Path,
    *,
    cargo: bool,
    target_dir: Path,
    keep_worktree: bool,
) -> dict[str, Any]:
    jcode_path = jcode_path.expanduser().resolve()
    patch_path = patch_path.expanduser().resolve()
    target_dir = target_dir.expanduser().resolve()
    checks: list[dict[str, Any]] = [
        _check("jcode_checkout:exists", jcode_path.is_dir(), path=str(jcode_path)),
        _check("patch:exists", patch_path.is_file(), path=str(patch_path)),
        _check("native_tool:exists", NATIVE_TOOL_DIR.is_dir(), path=str(NATIVE_TOOL_DIR)),
    ]
    if not jcode_path.is_dir() or not patch_path.is_file() or not NATIVE_TOOL_DIR.is_dir():
        return {
            "success": False,
            "checks": checks,
            "jcode_path": str(jcode_path),
            "patch_path": str(patch_path),
        }

    temp: tempfile.TemporaryDirectory[str] | None = None
    if keep_worktree:
        temp_path = Path(tempfile.mkdtemp(prefix="jcode-supertool-registry-"))
    else:
        temp = tempfile.TemporaryDirectory(prefix="jcode-supertool-registry-")
        temp_path = Path(temp.name)
    worktree: Path | None = None
    used_git_worktree = False
    try:
        worktree, used_git_worktree, worktree_result = _prepare_jcode_worktree(
            jcode_path,
            temp_path,
        )
        checks.append(_check(
            "jcode_worktree:prepared",
            worktree.is_dir(),
            path=str(worktree),
            mode="git_worktree" if used_git_worktree else "copy",
            fallback_stderr=(worktree_result.stderr[-4000:] if worktree_result else ""),
        ))

        patch_completed = _run(
            ["git", "apply", "--unidiff-zero", str(patch_path)],
            cwd=worktree,
        )
        checks.append(_check(
            "jcode_patch:applied",
            patch_completed.returncode == 0,
            returncode=patch_completed.returncode,
            stdout=patch_completed.stdout[-4000:],
            stderr=patch_completed.stderr[-4000:],
        ))
        if patch_completed.returncode != 0:
            return {
                "success": False,
                "checks": checks,
                "jcode_path": str(jcode_path),
                "patch_path": str(patch_path),
                "worktree": str(worktree),
            }

        native_destination = _copy_native_tool(worktree)
        _add_jcode_dev_dependency(worktree)
        test_path = _write_registry_test(worktree)
        checks.extend([
            _check(
                "native_tool:copied_into_jcode",
                (native_destination / "Cargo.toml").exists()
                and (native_destination / "src" / "lib.rs").exists(),
                path=str(native_destination),
            ),
            _check(
                "native_tool:exports_toolset",
                "default_hermes_toolset"
                in (native_destination / "src" / "lib.rs").read_text(encoding="utf-8"),
            ),
            _check(
                "jcode_test:writes_native_registry_test",
                test_path.exists()
                and "register_toolset" in test_path.read_text(encoding="utf-8"),
                path=str(test_path),
            ),
        ])

        if cargo:
            env = os.environ.copy()
            env["CARGO_TARGET_DIR"] = str(target_dir)
            cargo_completed = _run(
                [
                    "cargo",
                    "test",
                    "--no-default-features",
                    "--test",
                    "hermes_native_tool_registry",
                ],
                cwd=worktree,
                env=env,
            )
            checks.append(_check(
                "cargo:test_jcode_registry_with_hermes_tools",
                cargo_completed.returncode == 0,
                returncode=cargo_completed.returncode,
                stdout=cargo_completed.stdout[-6000:],
                stderr=cargo_completed.stderr[-6000:],
                target_dir=str(target_dir),
            ))

        success = all(item["ok"] for item in checks)
        return {
            "success": success,
            "checks": checks,
            "jcode_path": str(jcode_path),
            "patch_path": str(patch_path),
            "worktree": str(worktree),
            "kept_worktree": keep_worktree,
        }
    finally:
        if worktree is not None and not keep_worktree:
            _cleanup_worktree(jcode_path, worktree, used_git_worktree)
        if temp is not None:
            temp.cleanup()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--jcode",
        default=str(DEFAULT_JCODE),
        help="Path to a jcode checkout.",
    )
    parser.add_argument(
        "--patch",
        default=str(DEFAULT_PATCH),
        help="Registration patch to apply.",
    )
    parser.add_argument(
        "--skip-cargo",
        action="store_true",
        help="Prepare the temp jcode smoke without running cargo.",
    )
    parser.add_argument(
        "--target-dir",
        default=str(Path(tempfile.gettempdir()) / "jcode-supertool-registry-smoke-target"),
        help="Cargo target directory for the full registry smoke.",
    )
    parser.add_argument(
        "--keep-worktree",
        action="store_true",
        help="Do not delete the temporary jcode worktree.",
    )
    ns = parser.parse_args(argv)

    report = run_supertool_registry_smoke(
        Path(ns.jcode),
        Path(ns.patch),
        cargo=not ns.skip_cargo,
        target_dir=Path(ns.target_dir),
        keep_worktree=bool(ns.keep_worktree),
    )
    print(json.dumps(report, indent=2, ensure_ascii=True, sort_keys=True))
    return 0 if report.get("success") else 1


if __name__ == "__main__":
    raise SystemExit(main())
