"""Tests for resolve_session_source() — HERMES_SESSION_SOURCE override semantics."""

from __future__ import annotations

import os

import pytest

from hermes_state import resolve_session_source


def test_env_override_wins_over_platform_cli(monkeypatch):
    monkeypatch.setenv("HERMES_SESSION_SOURCE", "tool")
    assert resolve_session_source("cli") == "tool"


def test_platform_used_when_env_unset(monkeypatch):
    monkeypatch.delenv("HERMES_SESSION_SOURCE", raising=False)
    assert resolve_session_source("telegram") == "telegram"


def test_defaults_to_cli(monkeypatch):
    monkeypatch.delenv("HERMES_SESSION_SOURCE", raising=False)
    assert resolve_session_source(None) == "cli"
    assert resolve_session_source("") == "cli"


def test_blank_env_falls_back_to_platform(monkeypatch):
    monkeypatch.setenv("HERMES_SESSION_SOURCE", "   ")
    assert resolve_session_source("discord") == "discord"
