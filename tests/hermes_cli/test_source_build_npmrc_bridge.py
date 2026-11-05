"""npmrc settings reach dependency install scripts from the file npm itself would read.

pnpm reads an npmrc only for its registry and auth; npm also exported every other key to
install scripts as ``npm_config_<key>``, which is how node-pre-gyp finds a binary mirror.
``source_build_env`` keeps that bridge. It must read the npmrc npm would read (an explicit
``NPM_CONFIG_USERCONFIG`` wins over Hermes's own), and decode quoted values as npm does: a
quoted ``registry=`` exported with its quotes is a malformed URL that breaks every install.
"""
from __future__ import annotations

import os

import pytest

import pm
from pm.package import Runner


@pytest.fixture
def bridge(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    for name in list(os.environ):
        if name.upper().startswith("NPM_CONFIG_"):
            monkeypatch.delenv(name)
    monkeypatch.setattr(pm, "ensure", lambda name, *, base_env, **_: Runner(name, dict(base_env)))
    return home


def test_an_explicit_userconfig_is_the_file_bridged(bridge, tmp_path, monkeypatch):
    from hermes_cli.source_build import source_build_env

    (bridge / "npmrc").write_text("registry=https://hermes-file.invalid/\nhermes-only=1\n", encoding="utf-8")
    selected = tmp_path / "selected-npmrc"
    selected.write_text("electron-mirror=https://selected.invalid/electron/\n", encoding="utf-8")
    monkeypatch.setenv("NPM_CONFIG_USERCONFIG", str(selected))

    env = source_build_env()

    assert env["NPM_CONFIG_USERCONFIG"] == str(selected)
    assert env["npm_config_electron_mirror"] == "https://selected.invalid/electron/"
    assert "npm_config_registry" not in env and "npm_config_hermes_only" not in env


def test_quoted_values_are_exported_without_their_quotes(bridge):
    from hermes_cli.source_build import source_build_env

    (bridge / "npmrc").write_text(
        "registry=\"https://mirror.invalid/\"\nelectron-mirror='https://mirror.invalid/electron/'\n", encoding="utf-8")

    env = source_build_env()

    assert env["NPM_CONFIG_USERCONFIG"] == str(bridge / "npmrc")
    assert env["npm_config_registry"] == "https://mirror.invalid/"
    assert env["npm_config_electron_mirror"] == "https://mirror.invalid/electron/"
