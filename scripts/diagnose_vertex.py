#!/usr/bin/env python3
"""Vertex AI credential diagnostic for Hermes.

Run this ON the machine where `hermes chat` fails (e.g. the Google Cloud VM)
using the SAME Python environment Hermes runs in:

    python scripts/diagnose_vertex.py

It reports exactly which link in the credential chain is broken, so we stop
guessing between "google-auth not installed", "token minted but no project",
and "attached service account missing scope". It makes real network calls to
the GCE metadata server and the OAuth token endpoint, but prints NO secret
material (tokens are shown only as a length + prefix).
"""

from __future__ import annotations

import os
import sys

# Ensure the repo root (parent of scripts/) is importable so `agent.*` resolves
# when this script is run as `python scripts/diagnose_vertex.py`.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def _line(label: str, value: object) -> None:
    print(f"  {label:<32} {value}")


def main() -> int:
    print("=== Hermes Vertex AI credential diagnostic ===\n")
    print(f"python: {sys.executable}")
    print(f"version: {sys.version.split()[0]}\n")

    # 1. Is google-auth importable in THIS environment?
    print("[1] google-auth import")
    try:
        import google.auth  # noqa: F401
        import google.auth.transport.requests as gart
        from google.auth import compute_engine  # noqa: F401

        ver = getattr(getattr(google, "auth", None), "__version__", "?")
        _line("google-auth installed", f"YES (version {ver})")
    except Exception as e:  # pragma: no cover - environment probe
        _line("google-auth installed", f"NO -> {type(e).__name__}: {e}")
        print("\nDIAGNOSIS: google-auth is not available in this environment.")
        print("This alone produces the 'credentials could not be resolved' error.")
        print(
            "Fix: pip install 'google-auth' (or reinstall hermes with the vertex extra)."
        )
        return 1

    # 2. Relevant environment variables (paths only, never contents).
    print("\n[2] relevant environment variables")
    for var in (
        "GOOGLE_APPLICATION_CREDENTIALS",
        "VERTEX_CREDENTIALS_PATH",
        "VERTEX_PROJECT_ID",
        "GOOGLE_CLOUD_PROJECT",
        "VERTEX_REGION",
    ):
        val = os.environ.get(var)
        if val and "CREDENTIAL" in var:
            val = f"{val} (exists={os.path.exists(val)})"
        _line(var, val if val else "(unset)")

    # 3. Are we on GCE? Can we read the attached-SA project from metadata?
    print("\n[3] GCE metadata server (attached service account)")
    req = gart.Request()
    try:
        from google.auth.compute_engine import _metadata

        on_gce = _metadata.ping(req)
        _line("on GCE (metadata reachable)", on_gce)
        if on_gce:
            try:
                proj = _metadata.get_project_id(req)
                _line("metadata project-id", proj)
            except Exception as e:
                _line("metadata project-id", f"ERROR {type(e).__name__}: {e}")
            try:
                sa = _metadata.get(req, "instance/service-accounts/default/email")
                _line("attached SA email", sa)
                scopes = _metadata.get(req, "instance/service-accounts/default/scopes")
                _line("attached SA scopes", str(scopes).replace("\n", ","))
            except Exception as e:
                _line("attached SA info", f"ERROR {type(e).__name__}: {e}")
    except Exception as e:  # pragma: no cover - environment probe
        _line("metadata probe", f"ERROR {type(e).__name__}: {e}")

    # 4. What does google.auth.default() actually return here?
    print("\n[4] google.auth.default(scopes=[cloud-platform])")
    scopes = ["https://www.googleapis.com/auth/cloud-platform"]
    try:
        import google.auth

        creds, project = google.auth.default(scopes=scopes)
        _line("credentials type", type(creds).__name__)
        _line("project returned", project if project else "(None)")
        try:
            creds.refresh(req)
            tok = getattr(creds, "token", None) or ""
            _line("token minted", f"YES (len={len(tok)}, prefix={tok[:6]}...)")
        except Exception as e:
            _line("token minted", f"NO -> {type(e).__name__}: {e}")
    except Exception as e:  # pragma: no cover - environment probe
        _line("default() result", f"ERROR {type(e).__name__}: {e}")
        creds, project = None, None

    # 5. Exercise the exact path Hermes uses.
    print("\n[5] Hermes resolution path (agent.vertex_adapter)")
    try:
        from agent.vertex_adapter import get_vertex_config, get_vertex_credentials

        tok, proj = get_vertex_credentials()
        _line("get_vertex_credentials token", "present" if tok else "MISSING")
        _line("get_vertex_credentials project", proj if proj else "(None)")
        tok2, base = get_vertex_config()
        _line("get_vertex_config token", "present" if tok2 else "MISSING")
        _line(
            "get_vertex_config base_url",
            base if base else "(None) <- this is the failure",
        )
        if tok and not proj:
            print(
                "\nDIAGNOSIS: a token WAS minted but project_id is None, so "
                "get_vertex_config() returns (None, None) and Hermes reports "
                "'credentials could not be resolved'. Set VERTEX_PROJECT_ID / "
                "vertex.project_id, or use a build that resolves the project "
                "from the metadata server."
            )
    except Exception as e:  # pragma: no cover - environment probe
        _line("vertex_adapter path", f"ERROR {type(e).__name__}: {e}")

    print("\n=== done ===")
    print("Send this full output back (it contains no secrets).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
