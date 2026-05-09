"""Helpers for reporting Vercel Sandbox authentication state."""

from __future__ import annotations

import os
from dataclasses import dataclass

from agent.i18n import t


_TOKEN_TUPLE_VARS = ("VERCEL_TOKEN", "VERCEL_PROJECT_ID", "VERCEL_TEAM_ID")


@dataclass(frozen=True)
class VercelAuthStatus:
    ok: bool
    label: str
    detail_lines: tuple[str, ...]


def _present(name: str) -> bool:
    return bool(os.getenv(name))


def _vercel_t(key: str, default: str, **kwargs) -> str:
    return t(f"vercel_auth.{key}", default=default, **kwargs)


def describe_vercel_auth() -> VercelAuthStatus:
    """Return Vercel auth status without exposing secret values."""

    has_oidc = _present("VERCEL_OIDC_TOKEN")
    token_states = {name: _present(name) for name in _TOKEN_TUPLE_VARS}
    present_token_vars = tuple(name for name, present in token_states.items() if present)
    missing_token_vars = tuple(name for name, present in token_states.items() if not present)

    if has_oidc:
        details = [
            _vercel_t("oidc_detail_mode", "mode: OIDC"),
            _vercel_t("oidc_detail_active_env", "active env: VERCEL_OIDC_TOKEN"),
            _vercel_t("oidc_detail_note", "note: OIDC tokens are development-only; use access-token auth for deployments and long-running processes"),
        ]
        if present_token_vars:
            details.append(_vercel_t("oidc_detail_also_present", "also present: {vars}", vars=", ".join(present_token_vars)))
        return VercelAuthStatus(True, _vercel_t("oidc_label", "OIDC token via VERCEL_OIDC_TOKEN"), tuple(details))

    if not missing_token_vars:
        return VercelAuthStatus(
            True,
            _vercel_t("access_token_label", "access token + project/team via VERCEL_TOKEN, VERCEL_PROJECT_ID, VERCEL_TEAM_ID"),
            (
                _vercel_t("access_token_detail_mode", "mode: access token"),
                _vercel_t("access_token_detail_active_env", "active env: VERCEL_TOKEN, VERCEL_PROJECT_ID, VERCEL_TEAM_ID"),
            ),
        )

    if present_token_vars:
        return VercelAuthStatus(
            False,
            _vercel_t("partial_label", "partial access-token auth (missing {vars})", vars=", ".join(missing_token_vars)),
            (
                _vercel_t("partial_detail_mode", "mode: incomplete access token"),
                _vercel_t("partial_detail_present_env", "present env: {vars}", vars=", ".join(present_token_vars)),
                _vercel_t("partial_detail_missing_env", "missing env: {vars}", vars=", ".join(missing_token_vars)),
                _vercel_t("partial_detail_recommended", "recommended: set VERCEL_TOKEN, VERCEL_PROJECT_ID, and VERCEL_TEAM_ID together"),
            ),
        )

    return VercelAuthStatus(
        False,
        _vercel_t("not_configured_label", "not configured"),
        (
            _vercel_t("not_configured_detail_recommended", "recommended: set VERCEL_TOKEN, VERCEL_PROJECT_ID, and VERCEL_TEAM_ID"),
            _vercel_t("not_configured_detail_alt", "development-only alternative: set VERCEL_OIDC_TOKEN"),
        ),
    )
