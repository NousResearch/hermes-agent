"""Tests for BSL Language Server registry entries."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from agent.lsp.servers import (
    ServerContext,
    _BSL_DIAGNOSTICS_DOCUMENT_WAIT,
    _BSL_INITIALIZE_TIMEOUT,
    _find_bsl_jar,
    _root_bsl,
    _spawn_bsl_ls,
    find_server_for_file,
    language_id_for,
)


def test_bsl_server_def_uses_extended_timeouts():
    srv = find_server_for_file("/x/module.bsl")
    assert srv is not None
    assert srv.initialize_timeout == _BSL_INITIALIZE_TIMEOUT
    assert srv.diagnostics_document_wait == _BSL_DIAGNOSTICS_DOCUMENT_WAIT


def test_find_server_for_bsl_and_os_extensions():
    bsl_srv = find_server_for_file("/proj/CommonModules/Foo/Module.bsl")
    assert bsl_srv is not None
    assert bsl_srv.server_id == "bsl-language-server"
    os_srv = find_server_for_file("/proj/src/main.os")
    assert os_srv is not None
    assert os_srv.server_id == "bsl-language-server"


def test_language_id_for_bsl_files():
    assert language_id_for("/x/module.bsl") == "bsl"
    assert language_id_for("/x/script.os") == "bsl"


def test_root_bsl_finds_configuration_xml(tmp_path):
    proj = tmp_path / "myconf"
    proj.mkdir()
    (proj / "Configuration.xml").write_text("<Configuration/>", encoding="utf-8")
    module = proj / "CommonModules" / "Test"
    module.mkdir(parents=True)
    mod_file = module / "Module.bsl"
    mod_file.write_text("", encoding="utf-8")
    ws = str(tmp_path)
    assert _root_bsl(str(mod_file), ws) == str(proj)


def test_root_bsl_finds_dt_inf_directory(tmp_path):
    proj = tmp_path / "edt-project"
    proj.mkdir()
    (proj / "DT-INF").mkdir()
    src = proj / "src" / "CommonModules" / "Foo"
    src.mkdir(parents=True)
    mod_file = src / "Module.bsl"
    mod_file.write_text("", encoding="utf-8")
    assert _root_bsl(str(mod_file), str(tmp_path)) == str(proj)


def test_root_bsl_falls_back_to_workspace(tmp_path):
    mod = tmp_path / "orphan.bsl"
    mod.write_text("", encoding="utf-8")
    assert _root_bsl(str(mod), str(tmp_path)) == str(tmp_path)


def test_spawn_bsl_ls_builds_java_jar_command(tmp_path, monkeypatch):
    jar = tmp_path / "bsl-language-server.jar"
    jar.write_bytes(b"")
    monkeypatch.setenv("BSL_LANGUAGE_SERVER_JAR", str(jar))
    monkeypatch.setattr(
        "agent.lsp.servers._which",
        lambda *names: "/usr/bin/java" if names == ("java",) else None,
    )
    ctx = ServerContext(workspace_root=str(tmp_path))
    spec = _spawn_bsl_ls(str(tmp_path), ctx)
    assert spec is not None
    assert spec.command == ["/usr/bin/java", "-jar", str(jar), "--lsp"]
    assert spec.initialize_timeout == _BSL_INITIALIZE_TIMEOUT
    assert spec.diagnostics_document_wait == _BSL_DIAGNOSTICS_DOCUMENT_WAIT


def test_spawn_bsl_ls_uses_full_command_override(tmp_path, monkeypatch):
    jar = tmp_path / "custom.jar"
    jar.write_bytes(b"")
    monkeypatch.setattr(
        "agent.lsp.servers._which",
        lambda *names: "/usr/bin/java" if names == ("java",) else None,
    )
    ctx = ServerContext(
        workspace_root=str(tmp_path),
        binary_overrides={
            "bsl-language-server": [
                "/usr/bin/java",
                "-jar",
                str(jar),
                "--lsp",
                "-c",
                str(tmp_path / "bsl.json"),
            ],
        },
    )
    spec = _spawn_bsl_ls(str(tmp_path), ctx)
    assert spec is not None
    assert spec.command[0] == "/usr/bin/java"
    assert str(jar) in spec.command


def test_spawn_bsl_ls_returns_none_without_java(tmp_path, monkeypatch):
    jar = tmp_path / "bsl-language-server.jar"
    jar.write_bytes(b"")
    monkeypatch.setenv("BSL_LANGUAGE_SERVER_JAR", str(jar))
    monkeypatch.setattr("agent.lsp.servers._which", lambda *names: None)
    ctx = ServerContext(workspace_root=str(tmp_path))
    assert _spawn_bsl_ls(str(tmp_path), ctx) is None


def test_find_bsl_jar_from_env(tmp_path, monkeypatch):
    jar = tmp_path / "bsl-language-server.jar"
    jar.write_bytes(b"")
    monkeypatch.setenv("BSL_LANGUAGE_SERVER_JAR", str(jar))
    monkeypatch.delenv("PATH", raising=False)
    assert _find_bsl_jar() == str(jar)
