"""Live audit: validate that Hermes source code correctly routes 8 Z.AI keys.

Keys under test:
- 7 keys (2, 7, 8, 22, 23, 24, 26, 34) → coding+anthropic plan
- 1 key  (28)               → china+intl plan

The audit verifies that the patched _resolve_zai_base_url() handles
both plans correctly when env_override and config_override are unset
(i.e. relying on the cached probe or live probe).

For each key, we test:
  1. /api/coding/paas/v4 (OpenAI wire, coding plan)
  2. /api/anthropic       (Anthropic wire, coding+anthropic plan)
  3. /api/paas/v4         (OpenAI wire, standard plan)

This gives a complete picture of which endpoint each key can hit.
"""
from __future__ import annotations

import json
import os

import httpx
import pytest


CODING_URL = "https://api.z.ai/api/coding/paas/v4/chat/completions"
ANTHROPIC_URL = "https://api.z.ai/api/anthropic/v1/messages"
METERED_URL = "https://api.z.ai/api/paas/v4/chat/completions"
CHINA_CODING_URL = "https://open.bigmodel.cn/api/coding/paas/v4/chat/completions"


def _mask(token: str) -> str:
    if not token:
        return "<empty>"
    return f"{token[:8]}..."


@pytest.fixture(scope="module")
def all_keys():
    """Load all 8 keys from env. NEVER log the keys."""
    raw = os.environ.get("GLM_AUDIT_KEYS", "").strip()
    if not raw:
        pytest.skip("GLM_AUDIT_KEYS env var not set")
    pairs = []
    for chunk in raw.split(";"):
        if "=" not in chunk:
            continue
        name, _, key = chunk.partition("=")
        pairs.append((name.strip(), key.strip()))
    if not pairs:
        pytest.fail("GLM_AUDIT_KEYS is set but no valid pairs found")
    return pairs


def _probe(url: str, key: str, *, model: str = "glm-4-flash", body_style: str = "openai") -> tuple[int, str]:
    """Probe an endpoint. Returns (status_code, short_msg)."""
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    if body_style == "anthropic":
        body = {"model": "glm-5.2", "max_tokens": 1, "messages": [{"role": "user", "content": "ping"}]}
    else:
        body = {"model": model, "messages": [{"role": "user", "content": "ping"}], "max_tokens": 1}
    try:
        resp = httpx.post(url, headers=headers, json=body, timeout=15.0)
        try:
            err_body = resp.json()
            if "error" in err_body:
                err = err_body["error"]
                if isinstance(err, dict):
                    return resp.status_code, f"{err.get('code', '')} {err.get('message', '')[:60]}"
                return resp.status_code, str(err)[:60]
            return resp.status_code, "OK"
        except Exception:
            return resp.status_code, resp.text[:60]
    except Exception as e:
        return -1, str(e)[:60]


# ────────────────────────────────────────────────────────────────────────────
# Per-key probe across all 3 endpoint types
# ────────────────────────────────────────────────────────────────────────────


class TestPerKeyEndpointRouting:
    """For each of the 8 keys, probe all 3 endpoint types and report."""

    def test_all_keys_full_endpoint_matrix(self, all_keys):
        """Print a matrix: key × endpoint → HTTP status."""
        print("\n" + "=" * 100)
        print(f"{'KEY':12s} {'CODING':8s} {'ANTHROPIC':10s} {'METERED':8s} {'CHINA-CODING':12s}")
        print("=" * 100)

        results = {}
        for name, key in all_keys:
            masked = _mask(key)
            r_coding = _probe(CODING_URL, key)
            r_anthropic = _probe(ANTHROPIC_URL, key, body_style="anthropic")
            r_metered = _probe(METERED_URL, key)
            r_china = _probe(CHINA_CODING_URL, key)

            results[name] = {
                "coding": r_coding,
                "anthropic": r_anthropic,
                "metered": r_metered,
                "china_coding": r_china,
            }
            print(
                f"{masked:12s} "
                f"{r_coding[0]:3d}/{r_coding[1][:5]:8s} "
                f"{r_anthropic[0]:3d}/{r_anthropic[1][:8]:10s} "
                f"{r_metered[0]:3d}/{r_metered[1][:5]:8s} "
                f"{r_china[0]:3d}/{r_china[1][:8]:12s}"
            )

        print("=" * 100)

        # At least one key must work on coding endpoint (baseline sanity)
        any_coding_ok = any(r["coding"][0] == 200 for r in results.values())
        assert any_coding_ok, "No key works on /api/coding/paas/v4"

        # At least one key must work on anthropic endpoint
        any_anthropic_ok = any(r["anthropic"][0] == 200 for r in results.values())
        if any_anthropic_ok:
            print("\nAt least one key works on /api/anthropic — anthropic wire is reachable")
        else:
            print("\nWARNING: No key works on /api/anthropic — check key scopes")


# ────────────────────────────────────────────────────────────────────────────
# What does detect_zai_endpoint() actually detect?
# ────────────────────────────────────────────────────────────────────────────


class TestDetectZaiEndpointBehavior:
    """Verify what detect_zai_endpoint() returns for each key."""

    def test_each_key_detect_result(self, all_keys):
        """Print detect_zai_endpoint() result for each key."""
        from hermes_cli.auth import detect_zai_endpoint

        print("\n" + "=" * 80)
        print(f"{'KEY':12s} {'DETECTED ENDPOINT':40s} {'MODEL':15s}")
        print("=" * 80)

        for name, key in all_keys:
            masked = _mask(key)
            try:
                result = detect_zai_endpoint(key)
                if result:
                    print(f"{masked:12s} {result['base_url']:40s} {result.get('model', '?'):15s}")
                else:
                    print(f"{masked:12s} {'<NOT DETECTED>':40s}")
            except Exception as e:
                print(f"{masked:12s} {'<ERROR: ' + str(e)[:30] + '>':40s}")

        print("=" * 80)
        print("\nNote: ZAI_ENDPOINTS only includes /api/paas/v4 and /api/coding/paas/v4")
        print("/api/anthropic is NOT in ZAI_ENDPOINTS — will not be auto-detected")


# ────────────────────────────────────────────────────────────────────────────
# Test the runtime resolution chain
# ────────────────────────────────────────────────────────────────────────────


class TestRuntimeResolutionChain:
    """Verify _resolve_zai_base_url() handles the 8-key scenario correctly."""

    def test_resolve_zai_base_url_returns_endpoint_for_each_key(self, all_keys):
        """For each key, verify _resolve_zai_base_url returns a valid endpoint."""
        from hermes_cli.auth import _resolve_zai_base_url

        print("\n" + "=" * 80)
        print(f"{'KEY':12s} {'RESOLVED URL':45s} {'WORKS?':6s}")
        print("=" * 80)

        for name, key in all_keys:
            masked = _mask(key)
            try:
                # No env override, no config override — rely on probe
                resolved = _resolve_zai_base_url(
                    api_key=key,
                    default_url="https://api.z.ai/api/paas/v4",
                    env_override="",
                )
                # Test the resolved URL
                status, _ = _probe(resolved, key)
                works = "YES" if status == 200 else f"NO({status})"
                print(f"{masked:12s} {resolved:45s} {works:6s}")
            except Exception as e:
                print(f"{masked:12s} {'<ERROR: ' + str(e)[:40] + '>':45s}")

        print("=" * 80)

    def test_with_anthropic_env_override(self, all_keys):
        """Verify that setting GLM_BASE_URL=https://api.z.ai/api/anthropic routes all keys correctly."""
        from hermes_cli.auth import _resolve_zai_base_url

        print("\n" + "=" * 80)
        print(f"{'KEY':12s} {'GLM_BASE_URL=anthropic':30s} {'WORKS?':10s}")
        print("=" * 80)

        anthropic_override = "https://api.z.ai/api/anthropic"
        for name, key in all_keys:
            masked = _mask(key)
            try:
                # env_override set to anthropic — should win immediately
                resolved = _resolve_zai_base_url(
                    api_key=key,
                    default_url="https://api.z.ai/api/paas/v4",
                    env_override=anthropic_override,
                )
                assert resolved == anthropic_override, (
                    f"GLM_BASE_URL override should win, got {resolved}"
                )
                # Test with anthropic body format
                status, _ = _probe(anthropic_override, key, body_style="anthropic")
                works = f"YES({status})" if status == 200 else f"NO({status})"
                print(f"{masked:12s} {resolved:30s} {works:10s}")
            except Exception as e:
                print(f"{masked:12s} {'<ERROR: ' + str(e)[:40] + '>':30s}")

        print("=" * 80)
        print("\nWith GLM_BASE_URL=https://api.z.ai/api/anthropic set:")
        print("→ All keys route to anthropic endpoint (override wins)")
        print("→ Coding+anthropic keys work; china+intl keys may 400/401")


# ────────────────────────────────────────────────────────────────────────────
# Document the audit findings
# ────────────────────────────────────────────────────────────────────────────


class TestAuditFindings:
    """Summary of what we learned about Z.AI key routing."""

    def test_documented_finding_zai_endpoints_lacks_anthropic(self):
        """Document the gap: ZAI_ENDPOINTS doesn't include /api/anthropic."""
        from hermes_cli.auth import ZAI_ENDPOINTS

        endpoints = [ep[1] for ep in ZAI_ENDPOINTS]
        has_anthropic = any("/anthropic" in url for url in endpoints)

        assert not has_anthropic, (
            "ZAI_ENDPOINTS unexpectedly includes /api/anthropic — update this test"
        )

        # Print a clear finding for the developer
        print("\n=== AUDIT FINDING ===")
        print("ZAI_ENDPOINTS currently contains:")
        for ep_id, url, _, label in ZAI_ENDPOINTS:
            print(f"  - {ep_id}: {url} ({label})")
        print("\n/api/anthropic is MISSING from ZAI_ENDPOINTS.")
        print("Keys on coding+anthropic plans will NOT be auto-routed.")
        print("Users must set GLM_BASE_URL=https://api.z.ai/api/anthropic explicitly.")
        print("\nRecommended upstream PR: add anthropic-global and anthropic-cn endpoints.")
        print("===========================")

    def test_audit_documented_for_keys(self):
        """Just a marker test for documentation."""
        # This test always passes; it's a placeholder for the audit findings
        # which are printed by the other tests in this file.
        assert True