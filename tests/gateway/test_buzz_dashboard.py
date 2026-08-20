"""Production-shaped Dashboard contracts for the bundled Buzz plugin."""

from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DASHBOARD = PROJECT_ROOT / "plugins" / "platforms" / "buzz" / "dashboard"
VALID_NPUB = "npub1nl2u0wnd8mezfknc74q7pl9ec58h9nrrakce4tnk434qgaxl4psqe5twr6"
VALID_HEX = "a" * 64


def test_stock_dashboard_discovers_serves_and_safely_mounts_bundled_buzz(monkeypatch):
    from hermes_cli import web_server

    monkeypatch.setattr(web_server, "_dashboard_plugins_cache", None)
    plugins = web_server._discover_dashboard_plugins()
    buzz = next(plugin for plugin in plugins if plugin["name"] == "buzz-platform")

    assert buzz["source"] == "bundled"
    assert Path(buzz["_dir"]).resolve() == DASHBOARD.resolve()
    assert buzz["_api_file"] == "plugin_api.py"
    assert buzz["has_api"] is True
    assert "/api/plugins/buzz-platform/policy" in web_server.app.openapi()["paths"]

    response = asyncio.run(
        web_server.serve_plugin_asset("buzz-platform", "manifest.json")
    )
    assert Path(response.path).resolve() == (DASHBOARD / "manifest.json").resolve()

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(web_server.serve_plugin_asset("buzz-platform", "../settings.py"))
    assert exc_info.value.status_code == 403

    manifest = json.loads((DASHBOARD / "manifest.json").read_text(encoding="utf-8"))
    assert manifest == {
        "name": "buzz-platform",
        "label": "Buzz",
        "description": "Manage profile-scoped Buzz access and mention policy.",
        "icon": "MessageSquare",
        "version": "1.0.0",
        "tab": {"path": "/buzz", "hidden": True},
        "slots": ["config:section:buzz"],
        "entry": "dist/index.js",
        "css": "dist/style.css",
        "api": "plugin_api.py",
    }


def _run_dashboard_node(expression: str) -> dict[str, Any]:
    script = (
        "const dashboard = require('./plugins/platforms/buzz/dashboard/src/index.js');\n"
        f"const result = {expression};\n"
        "process.stdout.write(JSON.stringify(result));\n"
    )
    completed = subprocess.run(
        ["node", "-e", script],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    result = json.loads(completed.stdout)
    assert isinstance(result, dict)
    return result


def test_dashboard_owns_exact_approved_mark_and_reproducible_assets():
    source = DASHBOARD / "src" / "index.js"
    dist = DASHBOARD / "dist" / "index.js"
    mark = DASHBOARD / "assets" / "BuzzLogo24px.svg"

    assert source.read_bytes() == dist.read_bytes()
    assert hashlib.sha256(mark.read_bytes()).hexdigest() == (
        "6efb8bf616e0febd3940f411927d42cccddfd798112c9fc53e3f7b9ae46f4ce0"
    )
    subprocess.run(
        ["python", str(DASHBOARD / "build.py"), "--check"],
        cwd=PROJECT_ROOT,
        check=True,
    )


def test_dashboard_bundle_uses_policy_api_profile_and_stale_response_guards():
    source = (DASHBOARD / "src" / "index.js").read_text(encoding="utf-8")

    assert '"/api/plugins/buzz-platform/policy"' in source
    assert "SDK.fetchJSON" in source
    assert 'method: "PUT"' in source
    assert '"Content-Type": "application/json"' in source
    assert "SDK.api.getManagementProfile" in source
    assert "URLSearchParams" not in source
    assert "encodeURIComponent(profile)" in source
    api_source = (PROJECT_ROOT / "web" / "src" / "lib" / "api.ts").read_text(
        encoding="utf-8"
    )
    assert "export const api = {\n  getManagementProfile," in api_source
    assert "requestSequence" in source
    assert "if (sequence !== requestSequence.current)" in source
    assert "const readyPair" in source
    assert "loading || saving || !ready || locked" in source
    assert "useEffect" in source and "[profile]" in source
    assert 'registry.registerSlot(PLUGIN_NAME, "config:section:buzz", BuzzPolicyPanel)' in source
    assert "registry.register(PLUGIN_NAME" not in source


def test_dashboard_identity_validation_preserves_multiline_draft_and_builds_exact_body():
    result = _run_dashboard_node(
        "({"
        "valid: dashboard.validateAllowedUsers('"
        + VALID_NPUB
        + "\\n"
        + VALID_HEX.upper()
        + "'),"
        "invalid: dashboard.validateAllowedUsers('not-valid\\n" + VALID_HEX + "'),"
        "draft: dashboard.updateAllowedUsersDraft('not-valid\\n"
        + VALID_HEX
        + "').draft,"
        "body: dashboard.buildPolicyBody({allowedUsersText: '"
        + VALID_NPUB
        + "\\n"
        + VALID_HEX.upper()
        + "', allowAllUsers: true, requireMention: false, threadRequireMention: true})"
        "})"
    )

    assert result["valid"] is None
    assert "item 1" in result["invalid"]
    assert result["draft"] == f"not-valid\n{VALID_HEX}"
    assert result["body"] == {
        "policy": {
            "allowed_users": [VALID_NPUB, VALID_HEX.upper()],
            "allow_all_users": True,
            "require_mention": False,
            "thread_require_mention": True,
        }
    }


def test_dashboard_copy_covers_live_managed_environment_legacy_and_save_states():
    source = (DASHBOARD / "src" / "index.js").read_text(encoding="utf-8")

    for required in (
        "Policy changes apply immediately. No Gateway restart is required.",
        "Saving…",
        "Saved",
        "managed policy",
        "managed fields",
        "Active environment value is hidden",
        "Non-overridden policy changes apply immediately",
        "Managed policy unavailable",
        "could not be inspected safely",
        "Legacy policy keys will be cleaned up when you save.",
        "Legacy policy keys were cleaned up by this save.",
        "Buzz-specific policy",
        "Gateway or pairing grants can broaden access",
    ):
        assert required in source


def test_dashboard_never_reads_or_renders_secret_values_or_config_paths():
    source = (DASHBOARD / "src" / "index.js").read_text(encoding="utf-8")

    for forbidden in (
        "__HERMES_SESSION_TOKEN__",
        "HERMES_HOME",
        "HERMES_MANAGED_DIR",
        "config.yaml",
        '".env"',
        "environment_values",
        "environment_paths",
    ):
        assert forbidden not in source


def test_dashboard_styles_present_an_obvious_spacious_config_section():
    css = (DASHBOARD / "dist" / "style.css").read_text(encoding="utf-8")

    assert "../assets/BuzzLogo24px.svg" in css
    assert ".buzz-policy-section" in css
    assert "padding:" in css
    assert "gap:" in css
    assert "@media" in css


def test_get_policy_reports_profile_config_and_private_precedence_metadata(
    monkeypatch, tmp_path
):
    from hermes_cli import managed_scope, web_server

    root = tmp_path / "hermes"
    profile_home = root / "profiles" / "work"
    profile_home.mkdir(parents=True)
    (profile_home / "config.yaml").write_text(
        "unrelated: keep\n"
        "gateway:\n"
        "  platforms:\n"
        "    buzz:\n"
        "      extra:\n"
        f"        allowed_users: ['{VALID_HEX}']\n"
        "        allow_all_users: false\n"
        "        require_mention: false\n"
        "        thread_require_mention: true\n"
        "buzz:\n"
        f"  allowed_users: ['{VALID_HEX}']\n",
        encoding="utf-8",
    )
    (profile_home / ".env").write_text(
        "BUZZ_ALLOW_ALL_USERS=top-secret-environment-value\n",
        encoding="utf-8",
    )
    managed = tmp_path / "managed"
    managed.mkdir()
    (managed / "config.yaml").write_text(
        "gateway:\n"
        "  platforms:\n"
        "    buzz:\n"
        "      extra:\n"
        "        require_mention: true\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(managed))
    managed_scope.invalidate_managed_cache()

    client = TestClient(web_server.app)
    response = client.get(
        "/api/plugins/buzz-platform/policy",
        params={"profile": "work"},
        headers={"X-Hermes-Session-Token": web_server._SESSION_TOKEN},
    )

    assert response.status_code == 200
    assert response.json() == {
        "profile": "work",
        "policy": {
            "allowed_users": [VALID_HEX],
            "allow_all_users": None,
            "require_mention": True,
            "thread_require_mention": True,
        },
        "environment_overrides": ["allow_all_users"],
        "indeterminate_fields": ["allow_all_users"],
        "ineffective_fields": ["allow_all_users"],
        "managed_fields": ["require_mention"],
        "locked": True,
        "managed_error": False,
        "user_policy_unavailable": False,
        "legacy_fields": ["allowed_users"],
        "legacy_cleanup_required": True,
        "additional_global_grants_active": False,
        "additional_pairing_grants_active": False,
    }
    serialized = response.text
    assert "top-secret-environment-value" not in serialized
    assert str(profile_home) not in serialized
    assert str(managed) not in serialized


def test_valid_environment_overrides_are_redacted_under_managed_lock(monkeypatch, tmp_path):
    from hermes_cli import managed_scope, web_server

    home = tmp_path / "hermes"
    home.mkdir()
    configured_identity = "b" * 64
    environment_identity = "c" * 64
    (home / "config.yaml").write_text(
        "gateway:\n  platforms:\n    buzz:\n      extra:\n"
        f"        allowed_users: ['{configured_identity}']\n"
        "        allow_all_users: false\n"
        "        require_mention: true\n"
        "        thread_require_mention: true\n",
        encoding="utf-8",
    )
    for name in (
        "BUZZ_ALLOWED_USERS", "BUZZ_ALLOW_ALL_USERS", "BUZZ_REQUIRE_MENTION",
        "BUZZ_THREAD_REQUIRE_MENTION", "GATEWAY_ALLOWED_USERS",
        "GATEWAY_ALLOW_ALL_USERS",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("BUZZ_ALLOWED_USERS", environment_identity)
    monkeypatch.setenv("BUZZ_ALLOW_ALL_USERS", "true")
    monkeypatch.setenv("BUZZ_REQUIRE_MENTION", "false")
    managed = tmp_path / "managed"
    managed.mkdir()
    (managed / "config.yaml").write_text(
        "gateway:\n  platforms:\n    buzz:\n      extra:\n"
        "        thread_require_mention: false\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(managed))
    managed_scope.invalidate_managed_cache()

    response = TestClient(web_server.app).get(
        "/api/plugins/buzz-platform/policy",
        headers={"X-Hermes-Session-Token": web_server._SESSION_TOKEN},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["locked"] is True
    assert payload["policy"] == {
        "allowed_users": None,
        "allow_all_users": None,
        "require_mention": None,
        "thread_require_mention": False,
    }
    assert payload["indeterminate_fields"] == [
        "allow_all_users", "allowed_users", "require_mention"
    ]
    assert environment_identity not in response.text
    assert configured_identity not in response.text


def test_dashboard_contract_renders_overrides_indeterminate_and_scopes_additive_grants():
    source = (DASHBOARD / "src" / "index.js").read_text(encoding="utf-8")
    result = _run_dashboard_node(
        "dashboard.settingsFromPayload({policy: {allowed_users: null, "
        "allow_all_users: null, require_mention: null, thread_require_mention: true}, "
        "indeterminate_fields: ['allowed_users', 'allow_all_users', 'require_mention']})"
    )

    assert "allowedUsersText" not in result
    assert result["indeterminateFields"] == [
        "allowed_users", "allow_all_users", "require_mention"
    ]
    assert '"aria-checked": props.indeterminate ? "mixed"' in source
    assert 'fieldDisabled("allowed_users")' in source
    assert 'fieldDisabled("allow_all_users")' in source
    assert 'fieldDisabled("require_mention")' in source
    assert 'fieldDisabled("thread_require_mention")' in source
    assert "Active environment value is hidden" in source
    assert "Buzz-specific policy" in source
    assert "Gateway or pairing grants can broaden access" in source


def test_dashboard_partial_body_omits_indeterminate_values_and_drafts():
    result = _run_dashboard_node(
        "(() => { const settings = dashboard.settingsFromPayload({policy: {"
        "allowed_users: null, allow_all_users: true, require_mention: null, "
        "thread_require_mention: false}, indeterminate_fields: "
        "['allowed_users', 'require_mention']}); return {settings, "
        "body: dashboard.buildPolicyBody(settings)}; })()"
    )

    assert "allowedUsersText" not in result["settings"]
    assert "requireMention" not in result["settings"]
    assert result["body"] == {
        "policy": {"allow_all_users": True, "thread_require_mention": False}
    }


def test_dashboard_reports_additive_grants_without_serializing_identities(
    monkeypatch, tmp_path
):
    from hermes_cli import web_server

    home = tmp_path / "hermes"
    pairing = home / "platforms" / "pairing"
    pairing.mkdir(parents=True)
    global_identity = "d" * 64
    paired_identity = "e" * 64
    (pairing / "buzz-approved.json").write_text(
        json.dumps({paired_identity: {"user_name": "private"}}), encoding="utf-8"
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("GATEWAY_ALLOWED_USERS", global_identity)

    response = TestClient(web_server.app).get(
        "/api/plugins/buzz-platform/policy",
        headers={"X-Hermes-Session-Token": web_server._SESSION_TOKEN},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["additional_global_grants_active"] is True
    assert payload["additional_pairing_grants_active"] is True
    assert global_identity not in response.text
    assert paired_identity not in response.text
    assert "private" not in response.text


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"policy": {}, "unexpected": True},
        {"policy": {"unexpected": True}},
        {"policy": {"require_mention": None}},
        {
            "policy": {
                "allowed_users": [],
                "allow_all_users": "false",
                "require_mention": True,
                "thread_require_mention": True,
            }
        },
        {
            "policy": {
                "allowed_users": "not-a-list",
                "allow_all_users": False,
                "require_mention": True,
                "thread_require_mention": True,
            }
        },
        {
            "policy": {
                "allowed_users": ["not-a-valid-public-key"],
                "allow_all_users": False,
                "require_mention": True,
                "thread_require_mention": True,
            }
        },
    ],
)
def test_put_rejects_invalid_shape_boolean_list_or_identity_without_writing(
    monkeypatch, tmp_path, body
):
    from hermes_cli import web_server

    home = tmp_path / "hermes"
    home.mkdir()
    config_path = home / "config.yaml"
    original = b"unrelated: keep\n"
    config_path.write_bytes(original)
    monkeypatch.setenv("HERMES_HOME", str(home))

    response = TestClient(web_server.app).put(
        "/api/plugins/buzz-platform/policy",
        json=body,
        headers={"X-Hermes-Session-Token": web_server._SESSION_TOKEN},
    )

    assert response.status_code == 422
    assert config_path.read_bytes() == original


def test_put_saves_canonical_profile_policy_and_cleans_only_policy_aliases(
    monkeypatch, tmp_path
):
    from hermes_cli import web_server

    root = tmp_path / "hermes"
    work = root / "profiles" / "work"
    other = root / "profiles" / "other"
    work.mkdir(parents=True)
    other.mkdir(parents=True)
    original = {
        "unrelated": {"keep": "yes"},
        "gateway": {
            "keep": "gateway",
            "platforms": {
                "buzz": {
                    "allowed_users": ["b" * 64],
                    "direct_transport": "keep",
                    "extra": {
                        "allowed_users": ["c" * 64],
                        "allow_all_users": True,
                        "require_mention": False,
                        "thread_require_mention": False,
                        "relay_url": "keep-canonical",
                    },
                }
            },
            "buzz": {
                "allow_all_users": True,
                "legacy_gateway": "keep",
                "extra": {
                    "require_mention": False,
                    "legacy_gateway_extra": "keep",
                },
            },
        },
        "platforms": {
            "buzz": {
                "thread_require_mention": False,
                "legacy_platform": "keep",
                "extra": {
                    "allowed_users": ["d" * 64],
                    "legacy_platform_extra": "keep",
                },
            }
        },
        "buzz": {
            "allow_all_users": True,
            "legacy_root": "keep",
            "extra": {
                "require_mention": False,
                "legacy_root_extra": "keep",
            },
        },
    }
    import yaml

    (work / "config.yaml").write_text(yaml.safe_dump(original), encoding="utf-8")
    other_bytes = b"sentinel: other-profile\n"
    (other / "config.yaml").write_bytes(other_bytes)
    (work / ".env").write_text("BUZZ_REQUIRE_MENTION=false\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(root))

    body = {
        "policy": {
            "allowed_users": [VALID_NPUB, VALID_HEX.upper(), VALID_HEX],
            "allow_all_users": False,
            "require_mention": True,
            "thread_require_mention": True,
        }
    }
    response = TestClient(web_server.app).put(
        "/api/plugins/buzz-platform/policy",
        params={"profile": "work"},
        json=body,
        headers={"X-Hermes-Session-Token": web_server._SESSION_TOKEN},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["policy"] == {
        "allowed_users": [
            "9fd5c7ba6d3ef224da78f541e0fcb9c50f72cc63edb19aae76ac6a0474dfa860",
            VALID_HEX,
        ],
        "allow_all_users": False,
        "require_mention": None,
        "thread_require_mention": True,
    }
    assert payload["environment_overrides"] == ["require_mention"]
    assert payload["ineffective_fields"] == ["require_mention"]
    assert payload["legacy_cleanup_required"] is False

    saved = yaml.safe_load((work / "config.yaml").read_text(encoding="utf-8"))
    canonical = saved["gateway"]["platforms"]["buzz"]
    assert canonical["extra"] == {
        "allowed_users": [
            "9fd5c7ba6d3ef224da78f541e0fcb9c50f72cc63edb19aae76ac6a0474dfa860",
            VALID_HEX,
        ],
        "allow_all_users": False,
        "require_mention": True,
        "thread_require_mention": True,
        "relay_url": "keep-canonical",
    }
    assert canonical["direct_transport"] == "keep"
    for field in (
        "allowed_users",
        "allow_all_users",
        "require_mention",
        "thread_require_mention",
    ):
        assert field not in canonical
        for candidate in (
            saved["platforms"]["buzz"],
            saved["platforms"]["buzz"]["extra"],
            saved["gateway"]["buzz"],
            saved["gateway"]["buzz"]["extra"],
            saved["buzz"],
            saved["buzz"]["extra"],
        ):
            assert field not in candidate
    assert saved["unrelated"] == {"keep": "yes"}
    assert saved["gateway"]["keep"] == "gateway"
    assert saved["gateway"]["buzz"] == {
        "legacy_gateway": "keep",
        "extra": {"legacy_gateway_extra": "keep"},
    }
    assert saved["platforms"]["buzz"] == {
        "legacy_platform": "keep",
        "extra": {"legacy_platform_extra": "keep"},
    }
    assert saved["buzz"] == {
        "legacy_root": "keep",
        "extra": {"legacy_root_extra": "keep"},
    }
    assert (other / "config.yaml").read_bytes() == other_bytes


def test_partial_put_does_not_synthesize_omitted_policy_defaults(monkeypatch, tmp_path):
    from hermes_cli import web_server
    import yaml

    home = tmp_path / "hermes"
    home.mkdir()
    config_path = home / "config.yaml"
    config_path.write_text("unrelated: keep\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))

    response = TestClient(web_server.app).put(
        "/api/plugins/buzz-platform/policy",
        json={"policy": {"allow_all_users": True}},
        headers={"X-Hermes-Session-Token": web_server._SESSION_TOKEN},
    )

    assert response.status_code == 200
    saved = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert saved["unrelated"] == "keep"
    assert saved["gateway"]["platforms"]["buzz"]["extra"] == {
        "allow_all_users": True
    }


@pytest.mark.parametrize(
    ("overridden_field", "configured_value", "environment_value", "edit"),
    [
        ("allowed_users", [VALID_HEX], "b" * 64, {"allow_all_users": True}),
        ("allow_all_users", True, "false", {"require_mention": True}),
        ("require_mention", False, "true", {"thread_require_mention": True}),
        ("thread_require_mention", False, "true", {"allow_all_users": True}),
    ],
)
def test_partial_put_preserves_each_overridden_underlying_policy_value(
    monkeypatch, tmp_path, overridden_field, configured_value, environment_value, edit
):
    from hermes_cli import web_server
    import yaml

    root = tmp_path / "hermes"
    home = root / "profiles" / "work"
    home.mkdir(parents=True)
    config_path = home / "config.yaml"
    original_policy = {
        "allowed_users": [VALID_HEX],
        "allow_all_users": True,
        "require_mention": False,
        "thread_require_mention": False,
    }
    original_policy[overridden_field] = configured_value
    config_path.write_text(
        yaml.safe_dump({
            "unrelated": {"keep": True},
            "buzz": {"extra": {**original_policy, "legacy_keep": "yes"}},
        }),
        encoding="utf-8",
    )
    env_name = {
        "allowed_users": "BUZZ_ALLOWED_USERS",
        "allow_all_users": "BUZZ_ALLOW_ALL_USERS",
        "require_mention": "BUZZ_REQUIRE_MENTION",
        "thread_require_mention": "BUZZ_THREAD_REQUIRE_MENTION",
    }[overridden_field]
    env_path = home / ".env"
    env_path.write_text(f"{env_name}={environment_value}\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(root))
    client = TestClient(web_server.app)
    headers = {"X-Hermes-Session-Token": web_server._SESSION_TOKEN}

    response = client.put(
        "/api/plugins/buzz-platform/policy",
        params={"profile": "work"},
        json={"policy": edit},
        headers=headers,
    )
    assert response.status_code == 200
    assert response.json()["policy"][overridden_field] is None

    env_path.unlink()
    revealed = client.get(
        "/api/plugins/buzz-platform/policy",
        params={"profile": "work"},
        headers=headers,
    )
    assert revealed.status_code == 200
    assert revealed.json()["policy"][overridden_field] == configured_value
    saved = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert saved["unrelated"] == {"keep": True}
    assert saved["buzz"]["extra"] == {"legacy_keep": "yes"}


@pytest.mark.parametrize(
    "original",
    [
        b"gateway: [\n",
        b"gateway: \xff\n",
        b"- not\n- a\n- mapping\n",
    ],
    ids=["malformed-yaml", "invalid-utf8", "nonmapping-root"],
)
def test_unavailable_user_policy_locks_get_refuses_put_and_preserves_bytes(
    monkeypatch, tmp_path, original
):
    from hermes_cli import web_server

    home = tmp_path / "hermes"
    home.mkdir()
    config_path = home / "config.yaml"
    config_path.write_bytes(original)
    monkeypatch.setenv("HERMES_HOME", str(home))
    client = TestClient(web_server.app)
    headers = {"X-Hermes-Session-Token": web_server._SESSION_TOKEN}

    loaded = client.get("/api/plugins/buzz-platform/policy", headers=headers)
    assert loaded.status_code == 200
    payload = loaded.json()
    assert payload["locked"] is True
    assert payload["user_policy_unavailable"] is True
    assert payload["policy"] == dict.fromkeys((
        "allowed_users", "allow_all_users", "require_mention",
        "thread_require_mention",
    ))
    assert str(config_path) not in loaded.text

    refused = client.put(
        "/api/plugins/buzz-platform/policy",
        json={"policy": {"allow_all_users": True}},
        headers=headers,
    )
    assert refused.status_code == 409
    assert refused.json()["detail"] == {"error": "user_policy_unavailable"}
    assert config_path.read_bytes() == original


def test_user_policy_io_error_is_privately_unavailable_and_write_safe(
    monkeypatch, tmp_path
):
    from hermes_cli import web_server
    import sys

    home = tmp_path / "hermes"
    home.mkdir()
    config_path = home / "config.yaml"
    original = b"unrelated: keep\n"
    config_path.write_bytes(original)
    monkeypatch.setenv("HERMES_HOME", str(home))
    mounted_api = sys.modules["hermes_dashboard_plugin_buzz-platform"]
    monkeypatch.setattr(
        mounted_api, "_read_user_config_strict",
        lambda: (_ for _ in ()).throw(OSError("private path detail")),
    )
    client = TestClient(web_server.app)
    headers = {"X-Hermes-Session-Token": web_server._SESSION_TOKEN}

    loaded = client.get("/api/plugins/buzz-platform/policy", headers=headers)
    assert loaded.status_code == 200
    assert loaded.json()["user_policy_unavailable"] is True
    assert "private path detail" not in loaded.text
    refused = client.put(
        "/api/plugins/buzz-platform/policy",
        json={"policy": {"allow_all_users": True}}, headers=headers,
    )
    assert refused.status_code == 409
    assert refused.json()["detail"] == {"error": "user_policy_unavailable"}
    assert config_path.read_bytes() == original


@pytest.mark.parametrize("reference", ["${BUZZ_POLICY_VALUE}", "${env:BUZZ_POLICY_VALUE}"])
@pytest.mark.parametrize("field", [
    "allowed_users", "allow_all_users", "require_mention", "thread_require_mention",
])
@pytest.mark.parametrize("environment_value", [None, "malformed-private-value"])
def test_unlocked_environment_references_are_indeterminate_before_typed_parsing(
    monkeypatch, tmp_path, reference, field, environment_value
):
    from hermes_cli import web_server
    import yaml

    home = tmp_path / "hermes"
    home.mkdir()
    policy = {
        "allowed_users": [VALID_HEX],
        "allow_all_users": False,
        "require_mention": True,
        "thread_require_mention": True,
    }
    policy[field] = reference
    (home / "config.yaml").write_text(
        yaml.safe_dump({"gateway": {"platforms": {"buzz": {"extra": policy}}}}),
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    if environment_value is None:
        monkeypatch.delenv("BUZZ_POLICY_VALUE", raising=False)
    else:
        monkeypatch.setenv("BUZZ_POLICY_VALUE", environment_value)

    response = TestClient(web_server.app).get(
        "/api/plugins/buzz-platform/policy",
        headers={"X-Hermes-Session-Token": web_server._SESSION_TOKEN},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["locked"] is False
    assert payload["policy"][field] is None
    assert field in payload["indeterminate_fields"]
    assert "BUZZ_POLICY_VALUE" not in response.text
    assert "malformed-private-value" not in response.text


def test_put_refuses_managed_policy_fields_without_writing(monkeypatch, tmp_path):
    from hermes_cli import managed_scope, web_server

    home = tmp_path / "hermes"
    home.mkdir()
    config_path = home / "config.yaml"
    original = (
        "gateway:\n"
        "  platforms:\n"
        "    buzz:\n"
        "      extra:\n"
        "        allow_all_users: false\n"
    ).encode()
    config_path.write_bytes(original)
    managed = tmp_path / "managed"
    managed.mkdir()
    (managed / "config.yaml").write_text(
        "gateway:\n"
        "  platforms:\n"
        "    buzz:\n"
        "      extra:\n"
        "        allow_all_users: false\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(managed))
    managed_scope.invalidate_managed_cache()

    response = TestClient(web_server.app).put(
        "/api/plugins/buzz-platform/policy",
        json={
            "policy": {
                "allowed_users": [],
                "allow_all_users": True,
                "require_mention": True,
                "thread_require_mention": True,
            }
        },
        headers={"X-Hermes-Session-Token": web_server._SESSION_TOKEN},
    )

    assert response.status_code == 409
    assert response.json()["detail"] == {
        "error": "managed_policy",
        "managed_fields": ["allow_all_users"],
    }
    assert config_path.read_bytes() == original


def test_malformed_managed_policy_locks_get_and_refuses_put(monkeypatch, tmp_path):
    from hermes_cli import managed_scope, web_server

    home = tmp_path / "hermes"
    home.mkdir()
    config_path = home / "config.yaml"
    original = (
        "gateway:\n"
        "  platforms:\n"
        "    buzz:\n"
        "      extra:\n"
        "        allow_all_users: true\n"
    ).encode()
    config_path.write_bytes(original)
    managed = tmp_path / "managed"
    managed.mkdir()
    managed_path = managed / "config.yaml"
    managed_path.write_text(
        "gateway:\n"
        "  platforms:\n"
        "    buzz:\n"
        "      extra:\n"
        "        allow_all_users: false\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(managed))
    managed_scope.invalidate_managed_cache()
    client = TestClient(web_server.app)
    headers = {"X-Hermes-Session-Token": web_server._SESSION_TOKEN}

    warm = client.get("/api/plugins/buzz-platform/policy", headers=headers)
    assert warm.status_code == 200
    assert warm.json()["policy"]["allow_all_users"] is False

    managed_path.write_text("gateway: [\n", encoding="utf-8")
    managed_scope.invalidate_managed_cache()
    locked = client.get("/api/plugins/buzz-platform/policy", headers=headers)
    assert locked.status_code == 200
    assert locked.json()["locked"] is True
    assert locked.json()["managed_error"] is True
    assert locked.json()["policy"]["allow_all_users"] is None
    assert locked.json()["indeterminate_fields"] == sorted((
        "allowed_users", "allow_all_users", "require_mention",
        "thread_require_mention",
    ))

    response = client.put(
        "/api/plugins/buzz-platform/policy",
        json={
            "policy": {
                "allowed_users": [],
                "allow_all_users": True,
                "require_mention": True,
                "thread_require_mention": True,
            }
        },
        headers=headers,
    )
    assert response.status_code == 409
    assert response.json()["detail"] == {
        "error": "managed_policy_unavailable",
        "managed_fields": [],
    }
    assert config_path.read_bytes() == original


@pytest.mark.parametrize("managed_signal", ["environment", "marker"])
def test_put_refuses_coarse_managed_install_without_writing(
    monkeypatch, tmp_path, managed_signal
):
    from hermes_cli import web_server

    home = tmp_path / "hermes"
    home.mkdir()
    config_path = home / "config.yaml"
    original = b"unrelated: keep\n"
    config_path.write_bytes(original)
    monkeypatch.setenv("HERMES_HOME", str(home))
    if managed_signal == "environment":
        monkeypatch.setenv("HERMES_MANAGED", "package-manager")
    else:
        (home / ".managed").write_text("package-manager\n", encoding="utf-8")

    response = TestClient(web_server.app).put(
        "/api/plugins/buzz-platform/policy",
        json={
            "policy": {
                "allowed_users": [],
                "allow_all_users": False,
                "require_mention": True,
                "thread_require_mention": True,
            }
        },
        headers={"X-Hermes-Session-Token": web_server._SESSION_TOKEN},
    )
    assert response.status_code == 409
    assert response.json()["detail"]["error"] == "managed_install"
    assert config_path.read_bytes() == original


def test_policy_api_requires_dashboard_authentication(tmp_path, monkeypatch):
    from hermes_cli import web_server

    home = tmp_path / "hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    client = TestClient(web_server.app)
    body = {
        "policy": {
            "allowed_users": [],
            "allow_all_users": False,
            "require_mention": True,
            "thread_require_mention": True,
        }
    }

    assert client.get("/api/plugins/buzz-platform/policy").status_code in {401, 403}
    assert client.put(
        "/api/plugins/buzz-platform/policy", json=body
    ).status_code in {401, 403}
