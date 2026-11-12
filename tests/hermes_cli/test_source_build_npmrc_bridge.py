"""npmrc settings reach dependency install scripts from the file npm itself would read.

pnpm reads an npmrc only for its registry and auth; npm also exported every other key to
install scripts as ``npm_config_<key>``, which is how node-pre-gyp finds a binary mirror.
``source_build_env`` keeps that bridge. It must read the npmrc npm would read (an explicit
``NPM_CONFIG_USERCONFIG``, else Hermes's own, else ``~/.npmrc``), and decode quoted values as
npm does: a quoted ``registry=`` exported with its quotes is a malformed URL that breaks every
install.
"""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

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
    user = tmp_path / "user"
    user.mkdir()
    monkeypatch.setattr(Path, "home", lambda: user)
    monkeypatch.setattr(pm, "ensure", lambda name, *, base_env, **_: Runner(name, dict(base_env)))
    return home


def test_an_explicit_userconfig_is_the_file_bridged(bridge, tmp_path, monkeypatch):
    from hermes_agent.hermes_cli.source_build import source_build_env

    (bridge / "npmrc").write_text("registry=https://hermes-file.invalid/\nhermes-only=1\n", encoding="utf-8")
    selected = tmp_path / "selected-npmrc"
    selected.write_text("electron-mirror=https://selected.invalid/electron/\n", encoding="utf-8")
    monkeypatch.setenv("NPM_CONFIG_USERCONFIG", str(selected))

    env = source_build_env()

    assert env["NPM_CONFIG_USERCONFIG"] == str(selected)
    assert env["npm_config_electron_mirror"] == "https://selected.invalid/electron/"
    assert "npm_config_registry" not in env and "npm_config_hermes_only" not in env


def test_quoted_values_are_exported_without_their_quotes(bridge):
    from hermes_agent.hermes_cli.source_build import source_build_env

    (bridge / "npmrc").write_text(
        "registry=\"https://mirror.invalid/\"\nelectron-mirror='https://mirror.invalid/electron/'\n", encoding="utf-8")

    env = source_build_env()

    assert env["NPM_CONFIG_USERCONFIG"] == str(bridge / "npmrc")
    assert env["npm_config_registry"] == "https://mirror.invalid/"
    assert env["npm_config_electron_mirror"] == "https://mirror.invalid/electron/"


def test_without_a_userconfig_the_home_npmrc_is_bridged(bridge, tmp_path):
    from hermes_agent.hermes_cli.source_build import source_build_env

    (tmp_path / "user" / ".npmrc").write_text("disturl=https://mirror.invalid/node\n", encoding="utf-8")

    env = source_build_env()

    assert "NPM_CONFIG_USERCONFIG" not in env
    assert env["npm_config_disturl"] == "https://mirror.invalid/node"


# Each case below was checked against npm 12.0.2 (`npm config get`, `npm run` printing the
# script environment): the registry lookup and the bridge read the file as npm does.

def test_a_utf16_npmrc_reads_as_npm_reads_it_instead_of_failing_the_build(bridge):
    """Windows PowerShell 5.1's ``>`` writes UTF-16. npm decodes the file as UTF-8 and finds no
    usable setting in it; it never fails, and neither may a source build."""
    from hermes_agent.hermes_cli.source_build import source_build_env
    from pm.index_config import NPM_PUBLIC_REGISTRY, npm_registry

    npmrc = bridge / "npmrc"
    npmrc.write_bytes("registry=https://utf16.invalid/\r\ndisturl=https://utf16.invalid/node\r\n".encode("utf-16"))

    env = source_build_env()

    assert not any(value.startswith("https://utf16.invalid") for value in env.values())
    assert npm_registry({"NPM_CONFIG_USERCONFIG": str(npmrc)}) == NPM_PUBLIC_REGISTRY


def test_inline_comments_are_not_part_of_the_value(bridge):
    from hermes_agent.hermes_cli.source_build import source_build_env
    from pm.index_config import npm_registry

    npmrc = bridge / "npmrc"
    npmrc.write_text("registry=https://mirror.invalid/npm/ # corp mirror\n"
                     "disturl=https://mirror.invalid/node ; headers\nesc=a\\#b\n", encoding="utf-8")

    env = source_build_env()

    assert env["npm_config_disturl"] == "https://mirror.invalid/node"
    assert env["npm_config_esc"] == "a#b"
    assert npm_registry({"NPM_CONFIG_USERCONFIG": str(npmrc)}) == "https://mirror.invalid/npm/"


def test_keys_under_an_ini_section_are_not_settings(bridge):
    from hermes_agent.hermes_cli.source_build import source_build_env
    from pm.index_config import npm_registry

    npmrc = bridge / "npmrc"
    npmrc.write_text("registry=https://top.invalid/\n[corp]\nregistry=https://section.invalid/\n"
                     "disturl=https://section.invalid/node\n", encoding="utf-8")

    env = source_build_env()

    assert "npm_config_disturl" not in env and env["npm_config_registry"] == "https://top.invalid/"
    assert npm_registry({"NPM_CONFIG_USERCONFIG": str(npmrc)}) == "https://top.invalid/"


def test_an_optional_variable_reference_expands_to_empty(bridge, monkeypatch):
    from hermes_agent.hermes_cli.source_build import source_build_env
    from pm.index_config import npm_registry

    monkeypatch.delenv("UNSET_NPMRC_VAR", raising=False)
    monkeypatch.setenv("NPMRC_MIRROR", "https://env.invalid")
    npmrc = bridge / "npmrc"
    npmrc.write_text("registry=${NPMRC_MIRROR?}/npm/\nelectron-mirror=${UNSET_NPMRC_VAR?}\n", encoding="utf-8")

    env = source_build_env()

    assert env["npm_config_electron_mirror"] == ""
    assert npm_registry({"NPM_CONFIG_USERCONFIG": str(npmrc), **os.environ}) == "https://env.invalid/npm/"


def test_one_time_passwords_and_inline_tls_keys_never_reach_install_scripts(bridge):
    from hermes_agent.hermes_cli.source_build import source_build_env

    (bridge / "npmrc").write_text("otp=123456\nkey=-----BEGIN PRIVATE KEY-----\ncert=CLIENT-CERT\n"
                                  "OTP=654321\nfund=false\n", encoding="utf-8")

    env = source_build_env()

    assert not {"NPM_CONFIG_OTP", "NPM_CONFIG_KEY", "NPM_CONFIG_CERT"} & {name.upper() for name in env}
    assert env["npm_config_fund"] == ""  # npm exports a false setting as empty


def test_values_no_child_environment_can_carry_are_skipped(bridge):
    """A JSON-unquoted ``\\ud800`` (lone surrogate), ``\\u0000`` or a raw NUL byte would make
    every install spawn fail; npm survives them, and so must a source build."""
    from hermes_agent.hermes_cli.source_build import source_build_env
    from pm.index_config import npm_registry

    npmrc = bridge / "npmrc"
    npmrc.write_bytes(b'disturl="\\ud800"\nelectron-mirror="a\\u0000b"\nsass-binary-site=raw\x00nul\n'
                      b'registry=https://ok.invalid/\nfund=false\n')

    env = source_build_env()

    assert not {"npm_config_disturl", "npm_config_electron_mirror", "npm_config_sass_binary_site"} & set(env)
    assert env["npm_config_registry"] == "https://ok.invalid/" and env["npm_config_fund"] == ""
    assert npm_registry({"NPM_CONFIG_USERCONFIG": str(npmrc)}) == "https://ok.invalid/"
    subprocess.run([sys.executable, "-c", ""], env=env, check=True)


def test_hermes_home_npmrc_steers_pm_downloads_as_it_steers_builds(bridge, monkeypatch):
    """The npm keys documented for ``$HERMES_HOME/npmrc`` reach the source build, so PM's own
    npm downloads (pinned artifacts, dist-tags) must resolve against the same file."""
    from pm.artifact_mirror import pinned_source
    from pm.index_config import npm_registry

    (bridge / "npmrc").write_text("registry=https://hermes-home.invalid/npm/\n", encoding="utf-8")
    (bridge.parent / "user" / ".npmrc").write_text("registry=https://home-dir.invalid/\n", encoding="utf-8")

    assert npm_registry({}) == "https://hermes-home.invalid/npm/"
    source = pinned_source("https://registry.npmjs.org/pnpm/-/pnpm-12.15.0.tgz", bridge / "pnpm.tgz", "0" * 64)
    assert source.url == "https://hermes-home.invalid/npm/pnpm/-/pnpm-12.15.0.tgz"

    explicit = bridge.parent / "explicit-npmrc"
    explicit.write_text("registry=https://explicit.invalid/\n", encoding="utf-8")
    assert npm_registry({"npm_config_userconfig": str(explicit)}) == "https://explicit.invalid/"
