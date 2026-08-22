"""Tests for plugins/memory/openviking/__init__.py — URI normalization and payload handling."""

import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, cast
from urllib.parse import parse_qs, urlparse

import plugins.memory.openviking as openviking_plugin
from plugins.memory.openviking import OpenVikingMemoryProvider


def _write_skill(skills_dir, name, body="Do the thing."):
    skill_dir = skills_dir / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: Description for {name}\n---\n\n# {name}\n\n{body}\n",
        encoding="utf-8",
    )
    return skill_dir


def _write_bundle(bundles_dir, slug, skills):
    bundles_dir.mkdir(parents=True, exist_ok=True)
    lines = [f"name: {slug}", "skills:"]
    lines.extend(f"  - {skill}" for skill in skills)
    (bundles_dir / f"{slug}.yaml").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


class FakeVikingClient:
    def __init__(self, responses):
        self.responses = responses
        self.calls = []

    def get(self, path, params=None, **kwargs):
        self.calls.append((path, params or {}))
        response = self.responses[(path, tuple(sorted((params or {}).items())))]
        if isinstance(response, Exception):
            raise response
        return response

    def post(self, path, payload=None, **kwargs):
        self.calls.append((path, payload or {}))
        response = self.responses.get((path, tuple(sorted((payload or {}).items()))), {})
        if isinstance(response, Exception):
            raise response
        return response


class RecordingVikingClient:
    calls = []

    def __init__(self, *args, **kwargs):
        pass

    def post(self, path, payload=None, **kwargs):
        self.calls.append((path, payload or {}))
        return {"result": {"memories": [], "resources": []}}


def _recall_context_key(value):
    if isinstance(value, list):
        return tuple(value)
    return value


class FakeRecallClient:
    calls = []
    responses = {}

    def __init__(self, *args, **kwargs):
        pass

    def post(self, path, payload=None, **kwargs):
        payload = payload or {}
        self.__class__.calls.append(("post", path, dict(payload)))
        context_type = _recall_context_key(payload.get("context_type"))
        key = (path, context_type, payload.get("query"), payload.get("session_id"))
        if key not in self.__class__.responses:
            key = (path, context_type, payload.get("query"))
        if key not in self.__class__.responses:
            key = (path, context_type)
        response = self.__class__.responses[key]
        if isinstance(response, Exception):
            raise response
        return response

    def get(self, path, params=None, **kwargs):
        params = params or {}
        self.__class__.calls.append(("get", path, dict(params)))
        response = self.__class__.responses[(path, params.get("uri"))]
        if isinstance(response, Exception):
            raise response
        return response


def make_prefetch_provider(monkeypatch, responses, **env):
    monkeypatch.setattr(openviking_plugin, "_VikingClient", FakeRecallClient)
    FakeRecallClient.calls = []
    FakeRecallClient.responses = responses
    for key in (
        "OPENVIKING_RECALL_LIMIT",
        "OPENVIKING_RECALL_SCORE_THRESHOLD",
        "OPENVIKING_RECALL_MAX_INJECTED_CHARS",
        "OPENVIKING_RECALL_TIMEOUT_SECONDS",
        "OPENVIKING_RECALL_REQUEST_TIMEOUT_SECONDS",
        "OPENVIKING_RECALL_FULL_READ_LIMIT",
        "OPENVIKING_RECALL_PREFER_ABSTRACT",
        "OPENVIKING_RECALL_RESOURCES",
        "OPENVIKING_PROFILE_TOKEN_BUDGET",
    ):
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, str(value))

    provider = OpenVikingMemoryProvider()
    provider._client = object()
    provider._endpoint = "http://openviking.test"
    provider._account = "default"
    provider._user = "default"
    provider._agent = "hermes"
    provider._session_id = "session-test"
    return provider


def wait_prefetch(provider, query="What should we recall?", session_id="session-test"):
    return provider.prefetch(query, session_id=session_id)


class TestOpenVikingSummaryUriNormalization:
    def test_normalize_summary_uri_maps_pseudo_files_to_parent_directory(self):
        assert OpenVikingMemoryProvider._normalize_summary_uri("viking://user/hermes/.overview.md") == "viking://user/hermes"
        assert OpenVikingMemoryProvider._normalize_summary_uri("viking://resources/.abstract.md") == "viking://resources"
        assert OpenVikingMemoryProvider._normalize_summary_uri("viking://") == "viking://"
        assert OpenVikingMemoryProvider._normalize_summary_uri("viking://user/hermes/memories/profile.md") == "viking://user/hermes/memories/profile.md"

class TestOpenVikingSkillQuerySafety:
    def test_derive_returns_empty_string_for_non_string_input(self):
        assert openviking_plugin._derive_openviking_user_text(None) == ""
        assert openviking_plugin._derive_openviking_user_text(123) == ""
        assert openviking_plugin._derive_openviking_user_text([{"text": "hi"}]) == ""


    def test_skill_markers_match_hermes_scaffolding(self, tmp_path, monkeypatch):
        import agent.skill_bundles as skill_bundles
        import agent.skill_commands as skill_commands
        import tools.skills_tool as skills_tool

        skills_dir = tmp_path / "skills"
        bundles_dir = tmp_path / "skill-bundles"
        _write_skill(skills_dir, "example")
        _write_bundle(bundles_dir, "demo", ["example"])

        monkeypatch.setattr(skills_tool, "SKILLS_DIR", skills_dir)
        monkeypatch.setenv("HERMES_BUNDLES_DIR", str(bundles_dir))
        monkeypatch.setattr(skill_commands, "_skill_commands", {})
        monkeypatch.setattr(skill_commands, "_skill_commands_platform", None)
        monkeypatch.setattr(skill_bundles, "_bundles_cache", {})
        monkeypatch.setattr(skill_bundles, "_bundles_cache_mtime", None)

        skill_commands.scan_skill_commands()
        single = skill_commands.build_skill_invocation_message(
            "/example",
            user_instruction="hello",
            runtime_note="runtime detail",
        )
        assert single is not None
        assert skill_commands._SKILL_INVOCATION_PREFIX in single
        assert skill_commands._SINGLE_SKILL_MARKER in single
        assert skill_commands._SINGLE_SKILL_INSTRUCTION in single
        assert skill_commands._RUNTIME_NOTE in single

        skill_bundles.scan_bundles()
        bundle_result = skill_bundles.build_bundle_invocation_message(
            "/demo",
            user_instruction="hello",
        )
        assert bundle_result is not None
        bundle, _, _ = bundle_result
        assert skill_commands._BUNDLE_MARKER in bundle
        assert skill_commands._BUNDLE_USER_INSTRUCTION in bundle
        assert skill_commands._BUNDLE_FIRST_SKILL_BLOCK in bundle

    def test_prefetch_searches_only_slash_skill_user_instruction(self, monkeypatch):
        RecordingVikingClient.calls = []
        monkeypatch.setattr(openviking_plugin, "_VikingClient", RecordingVikingClient)
        provider = OpenVikingMemoryProvider()
        provider._client = cast(Any, object())
        provider._endpoint = "http://openviking.test"
        provider._api_key = ""
        provider._account = "default"
        provider._user = "default"
        provider._agent = "hermes"
        skill_message = (
            '[IMPORTANT: The user has invoked the "skill-creator" skill, indicating they want '
            "you to follow its instructions. The full skill content is loaded below.]\n\n"
            "# Skill Creator\n\n"
            "Large skill body that must not be searched or embedded.\n\n"
            "The user has provided the following instruction alongside the skill invocation: "
            "make a skill for release triage"
        )

        provider.prefetch(skill_message)

        assert RecordingVikingClient.calls == [
            (
                "/api/v1/search/find",
                {
                    "query": "make a skill for release triage",
                    "limit": 24,
                    "score_threshold": 0,
                    "context_type": "memory",
                },
            ),
        ]


    def test_sync_turn_stores_only_slash_skill_user_instruction(self, monkeypatch):
        RecordingVikingClient.calls = []
        monkeypatch.setattr(openviking_plugin, "_VikingClient", RecordingVikingClient)
        provider = OpenVikingMemoryProvider()
        provider._client = cast(Any, object())
        provider._endpoint = "http://openviking.test"
        provider._api_key = ""
        provider._account = "default"
        provider._user = "default"
        provider._agent = "hermes"
        provider._session_id = "session-1"
        skill_message = (
            '[IMPORTANT: The user has invoked the "skill-creator" skill, indicating they want '
            "you to follow its instructions. The full skill content is loaded below.]\n\n"
            "# Skill Creator\n\n"
            "Large skill body that must not be stored as user content.\n\n"
            "The user has provided the following instruction alongside the skill invocation: "
            "make a skill for release triage"
        )

        provider.sync_turn(skill_message, "Done.")
        assert provider._drain_writers("session-1", timeout=5.0)

        assert RecordingVikingClient.calls == [
            (
                "/api/v1/sessions/session-1/messages/batch",
                {
                    "messages": [
                        {
                            "role": "user",
                            "parts": [
                                {"type": "text", "text": "make a skill for release triage"},
                            ],
                        },
                        {
                            "role": "assistant",
                            "parts": [{"type": "text", "text": "Done."}],
                            "peer_id": "hermes",
                        },
                    ]
                },
            ),
        ]


class TestOpenVikingConfigSchema:
    def test_recall_policy_options_are_exposed_in_setup_schema(self):
        provider = OpenVikingMemoryProvider()

        schema = provider.get_config_schema()
        fields = {entry["key"]: entry for entry in schema}
        env_vars = {entry.get("env_var") for entry in schema}

        assert "OPENVIKING_RECALL_LIMIT" in env_vars
        assert "OPENVIKING_RECALL_SCORE_THRESHOLD" in env_vars
        assert "OPENVIKING_RECALL_MAX_INJECTED_CHARS" in env_vars
        assert "OPENVIKING_RECALL_TIMEOUT_SECONDS" in env_vars
        assert "OPENVIKING_RECALL_REQUEST_TIMEOUT_SECONDS" in env_vars
        assert "OPENVIKING_RECALL_FULL_READ_LIMIT" in env_vars
        assert "OPENVIKING_RECALL_PREFER_ABSTRACT" in env_vars
        assert "OPENVIKING_RECALL_RESOURCES" in env_vars
        assert fields["recall_limit"]["type"] == "integer"
        assert fields["recall_limit"]["minimum"] == 1
        assert fields["recall_limit"]["maximum"] == 100
        assert fields["recall_score_threshold"]["type"] == "number"
        assert fields["recall_prefer_abstract"]["type"] == "boolean"
        assert provider._recall_config() == {
            "limit": 6,
            "score_threshold": 0.15,
            "max_injected_chars": 4000,
            "timeout_seconds": 4.0,
            "request_timeout_seconds": 3.0,
            "full_read_limit": 2,
            "prefer_abstract": False,
            "resources": False,
        }

    def test_recall_config_reads_from_config_yaml(self, monkeypatch, tmp_path):
        """_recall_config() reads memory.openviking values from config.yaml when
        the corresponding OPENVIKING_RECALL_* env vars are not set."""
        # Populate config.yaml in the temp HERMES_HOME
        hermes_home = tmp_path / "hermes_test"
        hermes_home.mkdir(exist_ok=True)
        config_yaml = hermes_home / "config.yaml"
        config_yaml.write_text(
            """\
memory:
  provider: openviking
  openviking:
    recall_limit: 12
    recall_score_threshold: 0.42
    recall_max_injected_chars: 8000
    profile_token_budget: 7000
    recall_timeout_seconds: 2.0
    recall_request_timeout_seconds: 1.5
    recall_full_read_limit: 5
    recall_prefer_abstract: true
    recall_resources: true
""",
            encoding="utf-8",
        )
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        # Clear any OPENVIKING_RECALL_* env vars so config.yaml prevails
        for key in list(os.environ):
            if key.startswith("OPENVIKING_RECALL_"):
                monkeypatch.delenv(key, raising=False)

        provider = OpenVikingMemoryProvider()
        cfg = provider._recall_config()

        assert cfg["limit"] == 12
        assert cfg["score_threshold"] == 0.42
        assert cfg["max_injected_chars"] == 8000
        assert cfg["timeout_seconds"] == 2.0
        assert cfg["request_timeout_seconds"] == 1.5
        assert cfg["full_read_limit"] == 5
        assert cfg["prefer_abstract"] is True
        assert cfg["resources"] is True
        assert provider._profile_token_budget() == 7000

    def test_recall_config_env_overrides_config_yaml(self, monkeypatch, tmp_path):
        """Env vars OPENVIKING_RECALL_* take precedence over config.yaml values
        when both are present."""
        hermes_home = tmp_path / "hermes_test"
        hermes_home.mkdir(exist_ok=True)
        config_yaml = hermes_home / "config.yaml"
        config_yaml.write_text(
            """\
memory:
  provider: openviking
  openviking:
    recall_limit: 12
    recall_resources: true
""",
            encoding="utf-8",
        )
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        # Override config.yaml via env
        monkeypatch.setenv("OPENVIKING_RECALL_LIMIT", "6")
        monkeypatch.setenv("OPENVIKING_RECALL_RESOURCES", "false")

        provider = OpenVikingMemoryProvider()
        cfg = provider._recall_config()

        assert cfg["limit"] == 6, "env var should override config.yaml"
        assert cfg["resources"] is False, "env var false should override config.yaml true"

    def test_recall_config_partial_config_yaml(self, monkeypatch, tmp_path):
        """Partially populated config.yaml falls back to defaults for omitted keys
        and env vars can override individual fields."""
        hermes_home = tmp_path / "hermes_test"
        hermes_home.mkdir(exist_ok=True)
        config_yaml = hermes_home / "config.yaml"
        config_yaml.write_text(
            """\
memory:
  provider: openviking
  openviking:
    recall_limit: 3
    # No recall_resources set — should use default (False)
""",
            encoding="utf-8",
        )
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        for key in list(os.environ):
            if key.startswith("OPENVIKING_RECALL_"):
                monkeypatch.delenv(key, raising=False)

        provider = OpenVikingMemoryProvider()
        cfg = provider._recall_config()

        assert cfg["limit"] == 3, "config.yaml value should be picked up"
        assert cfg["resources"] is False, "omitted key should use default"
        assert cfg["timeout_seconds"] == 4.0, "omitted key should use built-in default"

    def test_dashboard_shaped_string_values_are_typed(self, monkeypatch):
        for key in list(os.environ):
            if key.startswith("OPENVIKING_RECALL_") or key == "OPENVIKING_PROFILE_TOKEN_BUDGET":
                monkeypatch.delenv(key, raising=False)
        monkeypatch.setattr(
            openviking_plugin,
            "_load_hermes_openviking_config",
            lambda: {
                "recall_limit": "12",
                "recall_score_threshold": "0.42",
                "recall_prefer_abstract": "false",
                "recall_resources": "true",
                "profile_token_budget": "7500",
            },
        )
        provider = OpenVikingMemoryProvider()

        cfg = provider._recall_config()

        assert cfg["limit"] == 12
        assert cfg["score_threshold"] == 0.42
        assert cfg["prefer_abstract"] is False
        assert cfg["resources"] is True
        assert provider._profile_token_budget() == 7500

    def test_invalid_recall_values_fall_back_without_type_errors(self, monkeypatch):
        for key in list(os.environ):
            if key.startswith("OPENVIKING_RECALL_") or key == "OPENVIKING_PROFILE_TOKEN_BUDGET":
                monkeypatch.delenv(key, raising=False)
        monkeypatch.setattr(
            openviking_plugin,
            "_load_hermes_openviking_config",
            lambda: {
                "recall_limit": "many",
                "recall_score_threshold": True,
                "recall_prefer_abstract": "sometimes",
                "profile_token_budget": "7.5",
            },
        )
        provider = OpenVikingMemoryProvider()

        cfg = provider._recall_config()

        assert cfg["limit"] == 6
        assert cfg["score_threshold"] == 0.15
        assert cfg["prefer_abstract"] is False
        assert provider._profile_token_budget() == 6000

    def test_recall_env_overrides_string_config_with_native_types(self, monkeypatch):
        monkeypatch.setattr(
            openviking_plugin,
            "_load_hermes_openviking_config",
            lambda: {"recall_limit": "12", "recall_resources": "false"},
        )
        monkeypatch.setenv("OPENVIKING_RECALL_LIMIT", "4")
        monkeypatch.setenv("OPENVIKING_RECALL_RESOURCES", "true")

        cfg = OpenVikingMemoryProvider()._recall_config()

        assert cfg["limit"] == 4
        assert cfg["resources"] is True


class TestOpenVikingTurnConversion:
    def test_extract_current_turn_anchors_on_latest_matching_user_and_assistant(self):
        messages = [
            {"role": "user", "content": "Please inspect the repository for assemble hooks."},
            {"role": "assistant", "content": "Earlier answer."},
            {"role": "user", "content": "Please inspect the repository for assemble hooks."},
            {
                "role": "assistant",
                "content": "I will search the codebase.",
                "tool_calls": [
                    {
                        "id": "call_rg_1",
                        "type": "function",
                        "function": {
                            "name": "shell_command",
                            "arguments": json.dumps({"command": "rg assemble"}),
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_rg_1",
                "name": "shell_command",
                "content": "agent/context_engine.py: no preassemble hook",
            },
            {"role": "assistant", "content": "The current main does not expose assemble."},
        ]

        turn = OpenVikingMemoryProvider._extract_current_turn_messages(
            messages,
            "Please inspect the repository for assemble hooks.",
            "The current main does not expose assemble.",
        )

        assert turn == messages[2:]

    def test_messages_to_openviking_batch_coalesces_tool_results(self):
        turn = [
            {"role": "user", "content": "Please inspect the repository for assemble hooks."},
            {
                "role": "assistant",
                "content": "I will search the codebase.",
                "tool_calls": [
                    {
                        "id": "call_rg_1",
                        "type": "function",
                        "function": {
                            "name": "shell_command",
                            "arguments": json.dumps({"command": "rg assemble"}),
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_rg_1",
                "name": "shell_command",
                "content": "agent/context_engine.py: no preassemble hook",
            },
            {"role": "assistant", "content": "The current main does not expose assemble."},
        ]

        batch = OpenVikingMemoryProvider._messages_to_openviking_batch(turn)

        assert [message["role"] for message in batch] == ["user", "assistant", "assistant", "assistant"]
        assert batch[0]["parts"] == [
            {"type": "text", "text": "Please inspect the repository for assemble hooks."}
        ]
        assert batch[1]["parts"] == [
            {"type": "text", "text": "I will search the codebase."}
        ]
        assert batch[2]["parts"] == [
            {
                "type": "tool",
                "tool_id": "call_rg_1",
                "tool_name": "shell_command",
                "tool_input": {"command": "rg assemble"},
                "tool_output": "agent/context_engine.py: no preassemble hook",
                "tool_status": "completed",
            }
        ]
        assert batch[3]["parts"] == [
            {"type": "text", "text": "The current main does not expose assemble."}
        ]


class TestOpenVikingRead:
    def test_overview_read_normalizes_uri_and_unwraps_result(self):
        provider = OpenVikingMemoryProvider()
        provider._client = FakeVikingClient(
            {
                (
                    "/api/v1/content/overview",
                    (("uri", "viking://user/hermes"),),
                ): {"result": {"content": "overview text"}},
            }
        )

        result = json.loads(provider._tool_read({"uri": "viking://user/hermes/.overview.md", "level": "overview"}))

        assert result["uri"] == "viking://user/hermes/.overview.md"
        assert result["resolved_uri"] == "viking://user/hermes"
        assert result["level"] == "overview"
        assert result["content"] == "overview text"
        assert provider._client.calls == [(
            "/api/v1/content/overview",
            {"uri": "viking://user/hermes"},
        )]


    def test_read_accepts_uri_batch_and_caps_batch_full_content(self):
        provider = OpenVikingMemoryProvider()
        uris = [
            "viking://user/hermes/memories/a.md",
            "viking://user/hermes/memories/b.md",
            "viking://user/hermes/memories/c.md",
            "viking://user/hermes/memories/d.md",
        ]
        provider._client = FakeVikingClient(
            {
                (
                    "/api/v1/content/read",
                    (("uri", uris[0]),),
                ): {"result": {"content": "a" * 3000}},
                (
                    "/api/v1/content/read",
                    (("uri", uris[1]),),
                ): {"result": {"content": "b content"}},
                (
                    "/api/v1/content/read",
                    (("uri", uris[2]),),
                ): {"result": {"content": "c content"}},
            }
        )

        result = json.loads(provider._tool_read({"uris": uris, "level": "full"}))

        assert result["requested"] == 4
        assert result["returned"] == 3
        assert result["truncated"] is True
        assert [entry["uri"] for entry in result["results"]] == uris[:3]
        assert result["results"][0]["content"].endswith(
            "[... truncated, use a more specific URI or full level]"
        )
        assert len(result["results"][0]["content"]) < 2700
        assert provider._client.calls == [
            ("/api/v1/content/read", {"uri": uris[0]}),
            ("/api/v1/content/read", {"uri": uris[1]}),
            ("/api/v1/content/read", {"uri": uris[2]}),
        ]


    def test_summary_uri_error_does_not_fallback_and_raises(self):
        provider = OpenVikingMemoryProvider()
        provider._client = FakeVikingClient(
            {
                (
                    "/api/v1/content/overview",
                    (("uri", "viking://user/hermes"),),
                ): RuntimeError("500 Internal Server Error"),
            }
        )

        try:
            provider._tool_read({"uri": "viking://user/hermes/.overview.md", "level": "overview"})
            assert False, "Expected summary endpoint error to be raised"
        except RuntimeError:
            pass

        assert provider._client.calls == [
            ("/api/v1/content/overview", {"uri": "viking://user/hermes"}),
        ]


class TestOpenVikingAutoRecallPrefetch:
    def test_prefetch_e2e_sends_limit_and_reads_l2_content(self, monkeypatch):
        records = {"searches": [], "reads": [], "listings": [], "headers": []}

        class Handler(BaseHTTPRequestHandler):
            def _send_json(self, payload):
                body = json.dumps(payload).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

            def do_GET(self):
                parsed = urlparse(self.path)
                if parsed.path == "/health":
                    self._send_json({"status": "ok", "healthy": True, "version": "test"})
                    return
                if parsed.path == "/api/v1/content/read":
                    query = parse_qs(parsed.query)
                    uri = query.get("uri", [""])[0]
                    if uri == "viking://~/memories/profile.md":
                        self._send_json({"result": "E2E user profile."})
                        return
                    records["reads"].append(uri)
                    self._send_json({"result": {"content": "E2E full L2 memory content."}})
                    return
                if parsed.path == "/api/v1/fs/ls":
                    query = {key: values[0] for key, values in parse_qs(parsed.query).items()}
                    records["listings"].append(query)
                    uri = query.get("uri")
                    if uri == "viking://~/memories/preferences":
                        self._send_json({
                            "result": [
                                {"isDir": True, "rel_path": "owner", "abstract": "ignored"},
                                {
                                    "isDir": False,
                                    "rel_path": "owner/answers.md",
                                    "abstract": "Prefers source-backed answers.",
                                },
                            ]
                        })
                        return
                    if uri == "viking://~/memories/entities":
                        self._send_json({
                            "result": [
                                {
                                    "isDir": False,
                                    "rel_path": "people/ada.md",
                                    "abstract": "Ada is the project owner.",
                                }
                            ]
                        })
                        return
                    self.send_error(404)
                    return
                self.send_error(404)

            def do_POST(self):
                length = int(self.headers.get("Content-Length", "0") or "0")
                payload = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
                records["headers"].append(dict(self.headers))
                if self.path == "/api/v1/search/search":
                    records["searches"].append(payload)
                    if payload.get("context_type") == "memory":
                        self._send_json({
                            "result": {
                                "memories": [
                                    {
                                        "uri": "viking://user/peers/hermes/memories/e2e-full.md",
                                        "score": 0.9,
                                        "level": 2,
                                        "category": "events",
                                        "abstract": "E2E abstract should not be injected.",
                                    }
                                ],
                                "resources": [],
                            }
                        })
                    else:
                        self._send_json({"result": {"memories": [], "resources": []}})
                    return
                self.send_error(404)

        server = HTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        endpoint = f"http://127.0.0.1:{server.server_port}"

        for key in (
            "OPENVIKING_RECALL_LIMIT",
            "OPENVIKING_RECALL_SCORE_THRESHOLD",
            "OPENVIKING_RECALL_MAX_INJECTED_CHARS",
            "OPENVIKING_RECALL_PREFER_ABSTRACT",
            "OPENVIKING_RECALL_RESOURCES",
            "OPENVIKING_PROFILE_TOKEN_BUDGET",
            "OPENVIKING_API_KEY",
        ):
            monkeypatch.delenv(key, raising=False)
        monkeypatch.setenv("OPENVIKING_ENDPOINT", endpoint)
        monkeypatch.setenv("OPENVIKING_ACCOUNT", "acct")
        monkeypatch.setenv("OPENVIKING_USER", "user")
        monkeypatch.setenv("OPENVIKING_AGENT", "hermes")

        provider = OpenVikingMemoryProvider()
        try:
            provider.initialize("e2e-session")
            block = provider.prefetch("What should we recall?", session_id="e2e-session")
        finally:
            provider.shutdown()
            server.shutdown()
            server.server_close()
            thread.join(timeout=3.0)

        assert block.startswith("## OpenViking Context\n")
        assert "E2E user profile." in block
        assert "owner/answers.md — Prefers source-backed answers." in block
        assert "people/ada.md — Ada is the project owner." in block
        assert "E2E full L2 memory content." in block
        assert "E2E abstract should not be injected." not in block
        assert records["reads"] == ["viking://user/peers/hermes/memories/e2e-full.md"]
        assert [listing["uri"] for listing in records["listings"]] == [
            "viking://~/memories/preferences",
            "viking://~/memories/entities",
        ]
        assert all(listing["output"] == "agent" for listing in records["listings"])
        assert all(listing["recursive"].lower() == "true" for listing in records["listings"])
        assert all(listing["abs_limit"] == "512" for listing in records["listings"])
        assert all(listing["node_limit"] == "512" for listing in records["listings"])
        assert len(records["searches"]) == 1
        assert records["searches"][0]["context_type"] == "memory"
        assert records["searches"][0]["session_id"] == "e2e-session"
        assert "target_uri" not in records["searches"][0]
        assert all(payload["limit"] == 24 for payload in records["searches"])
        assert all("top_k" not in payload for payload in records["searches"])
        assert all("mode" not in payload for payload in records["searches"])
        assert all(payload["score_threshold"] == 0 for payload in records["searches"])
        normalized_headers = [
            {key.lower(): value for key, value in headers.items()}
            for headers in records["headers"]
        ]
        assert all(headers.get("x-openviking-actor-peer") == "hermes" for headers in normalized_headers)
        assert all(headers.get("x-openviking-account") == "acct" for headers in normalized_headers)
        assert all(headers.get("x-openviking-user") == "user" for headers in normalized_headers)


class TestOpenVikingBrowse:
    def test_list_browse_unwraps_and_normalizes_entry_shapes(self):
        provider = OpenVikingMemoryProvider()
        provider._client = FakeVikingClient(
            {
                (
                    "/api/v1/fs/ls",
                    (("uri", "viking://user/hermes"),),
                ): {
                    "result": {
                        "entries": [
                            {"name": "memories", "uri": "viking://user/hermes/memories", "type": "dir"},
                            {"rel_path": "profile.md", "uri": "viking://user/hermes/memories/profile.md", "isDir": False, "abstract": "Profile"},
                        ]
                    }
                },
            }
        )

        result = json.loads(provider._tool_browse({"action": "list", "path": "viking://user/hermes"}))

        assert result["path"] == "viking://user/hermes"
        assert result["entries"] == [
            {"name": "memories", "uri": "viking://user/hermes/memories", "type": "dir", "abstract": ""},
            {"name": "profile.md", "uri": "viking://user/hermes/memories/profile.md", "type": "file", "abstract": "Profile"},
        ]
        assert provider._client.calls == [(
            "/api/v1/fs/ls",
            {"uri": "viking://user/hermes"},
        )]


class TestOpenVikingMemoryUriBuilder:
    """Regression tests for _build_memory_uri — fixes #36969.

    OpenViking's current memory layout stores peer-scoped memories under
    viking://~/peers/{peer_id}/...
    """

    def _make_provider(self, user="alice", agent="coder"):
        p = OpenVikingMemoryProvider.__new__(OpenVikingMemoryProvider)
        p._user = user
        p._agent = agent
        return p

    def test_uri_layout_includes_peer_segment(self):
        """URI must contain /peers/{peer_id}/ between user and memories."""
        p = self._make_provider(user="alice", agent="coder")
        uri = p._build_memory_uri("preferences")
        assert uri.startswith("viking://~/peers/coder/memories/preferences/mem_")
        assert uri.endswith(".md")


# ===================================================================
# Issue #21130 — OPENVIKING_* not reloaded after /reload
# ===================================================================


class TestEnsureClientReloadsEnv:
    """Verify /reload picks up new OPENVIKING_* values without a restart (#21130)."""

    def test_ensure_client_rebuilds_when_api_key_changes(self, monkeypatch):
        constructions = []

        class _StubClient:
            def __init__(self, endpoint, api_key, account="", user="", agent="hermes"):
                constructions.append({"endpoint": endpoint, "api_key": api_key,
                                      "account": account, "user": user, "agent": agent})
                self.endpoint, self.api_key = endpoint, api_key
                self.account, self.user, self.agent = account, user, agent

            def health(self):
                return True

        monkeypatch.setattr("plugins.memory.openviking._VikingClient", _StubClient)
        monkeypatch.setenv("OPENVIKING_ENDPOINT", "http://srv:31933")
        monkeypatch.setenv("OPENVIKING_API_KEY", "")

        provider = OpenVikingMemoryProvider()
        provider._env_refresh_enabled = True
        first = provider._ensure_client()
        assert first is not None
        assert first.api_key == ""
        assert len(constructions) == 1

        # Same env on second call — must reuse cached client (no rebuild).
        assert provider._ensure_client() is first
        assert len(constructions) == 1

        # Simulate /reload: env now carries the new API key.
        monkeypatch.setenv("OPENVIKING_API_KEY", "sk-fresh")
        rebuilt = provider._ensure_client()
        assert rebuilt is not None
        assert rebuilt is not first
        assert rebuilt.api_key == "sk-fresh"
        assert len(constructions) == 2


    def test_handle_tool_call_reconnects_after_startup_health_failure(self, monkeypatch):
        instances = []

        class _StubClient:
            def __init__(self, endpoint, api_key="", account="", user="", agent=""):
                self.endpoint = endpoint
                self.api_key = api_key
                self.account = account
                self.user = user
                self.agent = agent
                self.index = len(instances)
                self.posts = []
                instances.append(self)

            def health(self):
                return self.index > 0

            def post(self, path, payload=None, **kwargs):
                self.posts.append((path, payload or {}))
                return {"result": {"written_bytes": 11}}

        monkeypatch.setattr("plugins.memory.openviking._VikingClient", _StubClient)
        monkeypatch.setenv("OPENVIKING_ENDPOINT", "https://openviking.example")
        monkeypatch.setenv("OPENVIKING_API_KEY", "sk-test")

        provider = OpenVikingMemoryProvider()
        provider.initialize("session-1")

        assert provider._client is None

        out = json.loads(provider.handle_tool_call(
            "viking_remember",
            {"content": "stable fact"},
        ))

        assert out["status"] == "stored"
        assert len(instances) == 2
        assert instances[1].posts[0][0] == "/api/v1/content/write"
        assert instances[1].posts[0][1]["content"] == "stable fact"
        assert instances[1].posts[0][1]["mode"] == "create"
        assert instances[1].posts[0][1]["uri"].startswith(
            "viking://~/peers/hermes/memories/"
        )

    def test_concurrent_refresh_does_not_return_stale_client(self, monkeypatch):
        refresh_entered = threading.Event()
        release_refresh = threading.Event()

        class _StubClient:
            def __init__(self, endpoint, api_key="", account="", user="", agent=""):
                self.endpoint = endpoint
                self.api_key = api_key

            def health(self):
                if self.endpoint == "https://new.example":
                    refresh_entered.set()
                    assert release_refresh.wait(2.0)
                    return False
                return True

        monkeypatch.setattr(openviking_plugin, "_VikingClient", _StubClient)
        monkeypatch.setenv("OPENVIKING_ENDPOINT", "https://new.example")
        monkeypatch.setenv("OPENVIKING_API_KEY", "sk-new")
        monkeypatch.delenv("OPENVIKING_ACCOUNT", raising=False)
        monkeypatch.delenv("OPENVIKING_USER", raising=False)
        monkeypatch.delenv("OPENVIKING_AGENT", raising=False)

        provider = OpenVikingMemoryProvider()
        stale_client = _StubClient("https://old.example", "sk-old")
        provider._endpoint = stale_client.endpoint
        provider._api_key = stale_client.api_key
        provider._account = ""
        provider._user = ""
        provider._agent = "hermes"
        provider._client = stale_client
        provider._env_refresh_enabled = True

        results = {}
        errors = []
        second_started = threading.Event()
        second_done = threading.Event()

        def refresh_client(name, *, started=None, done=None):
            if started is not None:
                started.set()
            try:
                results[name] = provider._ensure_client()
            except BaseException as exc:  # pragma: no cover - surfaced below
                errors.append(exc)
            finally:
                if done is not None:
                    done.set()

        first = threading.Thread(target=refresh_client, args=("first",))
        first.start()
        assert refresh_entered.wait(2.0)

        second = threading.Thread(
            target=refresh_client,
            args=("second",),
            kwargs={"started": second_started, "done": second_done},
        )
        second.start()
        assert second_started.wait(2.0)
        completed_during_refresh = second_done.wait(0.2)

        release_refresh.set()
        first.join(timeout=2.0)
        second.join(timeout=2.0)

        assert not first.is_alive()
        assert not second.is_alive()
        assert errors == []
        assert completed_during_refresh is False
        assert results == {"first": None, "second": None}
        assert all(client is not stale_client for client in results.values())


    def test_concurrent_local_runtime_recovery_starts_once(self, monkeypatch):
        class _AliveThread:
            def is_alive(self):
                return True

        provider = OpenVikingMemoryProvider()
        provider._endpoint = "http://127.0.0.1:31933"
        start_calls = []
        start_lock = threading.Lock()
        first_start_entered = threading.Event()
        release_start = threading.Event()
        second_started = threading.Event()

        def start_local(endpoint):
            with start_lock:
                start_calls.append(endpoint)
            first_start_entered.set()
            release_start.wait(timeout=2)
            return openviking_plugin._LOCAL_SERVER_STARTED, "started"

        monkeypatch.setattr(openviking_plugin, "_start_local_openviking_server", start_local)
        monkeypatch.setattr(
            provider,
            "_start_runtime_openviking_waiter",
            lambda **kwargs: setattr(provider, "_runtime_start_thread", _AliveThread()),
            raising=False,
        )

        errors = []

        def recover(*, started=None):
            if started is not None:
                started.set()
            try:
                provider._handle_runtime_openviking_unreachable()
            except BaseException as exc:  # pragma: no cover - surfaced below
                errors.append(exc)

        threads = [
            threading.Thread(target=recover, name="openviking-recovery-1"),
            threading.Thread(
                target=recover,
                kwargs={"started": second_started},
                name="openviking-recovery-2",
            ),
        ]
        for thread in threads:
            thread.start()

        assert first_start_entered.wait(timeout=2)
        assert second_started.wait(timeout=2)
        release_start.set()
        for thread in threads:
            thread.join(timeout=2)
            assert not thread.is_alive()

        assert errors == []
        assert start_calls == ["http://127.0.0.1:31933"]


class TestEnsureClientFailureHardening:
    """Follow-up hardening on top of #21130: failed-config cooldown and
    torn-identity-safe connection snapshots."""

    def test_failed_config_probes_once_then_cools_down(self, monkeypatch):
        """A down endpoint must not pay a health probe on every access."""
        probes = []

        class _StubClient:
            def __init__(self, endpoint, api_key="", account="", user="", agent=""):
                self.endpoint = endpoint

            def health(self):
                probes.append(self.endpoint)
                return False

        monkeypatch.setattr(openviking_plugin, "_VikingClient", _StubClient)
        monkeypatch.setenv("OPENVIKING_ENDPOINT", "https://down.example")
        monkeypatch.setenv("OPENVIKING_API_KEY", "sk-x")

        provider = OpenVikingMemoryProvider()
        provider._env_refresh_enabled = True

        assert provider._ensure_client() is None
        assert provider._ensure_client() is None
        assert provider._ensure_client() is None
        # Only the first access probes; the rest hit the cooldown.
        assert len(probes) == 1


    def test_conn_snapshot_published_only_on_healthy(self, monkeypatch):
        class _StubClient:
            def __init__(self, endpoint, api_key="", account="", user="", agent=""):
                self.endpoint = endpoint

            def health(self):
                return self.endpoint == "https://up.example"

        monkeypatch.setattr(openviking_plugin, "_VikingClient", _StubClient)
        monkeypatch.setenv("OPENVIKING_ENDPOINT", "https://up.example")
        monkeypatch.setenv("OPENVIKING_API_KEY", "sk-good")
        monkeypatch.delenv("OPENVIKING_ACCOUNT", raising=False)
        monkeypatch.delenv("OPENVIKING_USER", raising=False)
        monkeypatch.delenv("OPENVIKING_AGENT", raising=False)

        provider = OpenVikingMemoryProvider()
        provider._env_refresh_enabled = True
        assert provider._ensure_client() is not None
        healthy_snapshot = provider._conn_snapshot
        assert healthy_snapshot is not None
        assert healthy_snapshot[0] == "https://up.example"
        assert healthy_snapshot[1] == "sk-good"

        # A failed refresh must NOT advance the snapshot: background writers
        # (_new_client) keep targeting the last identity that passed health.
        monkeypatch.setenv("OPENVIKING_ENDPOINT", "https://down.example")
        assert provider._ensure_client() is None
        assert provider._conn_snapshot == healthy_snapshot
        built = provider._new_client()
        assert built.endpoint == "https://up.example"


class TestUnavailableWarningsPromiseRetry:
    """Every "OpenViking is unavailable" warning must describe what actually
    happens next.

    ``_ensure_client()`` rebuilds and re-probes the client whenever the
    resolved config changes or the failed-config cooldown has elapsed, so no
    warning may tell the user memory is off for the rest of the run — that
    reads as "it never recovers" and sends people restarting hermes for
    nothing (#5721).
    """

    @staticmethod
    def _assert_promises_retry(message: str) -> None:
        assert "for this Hermes run" not in message, message
        assert "will retry on a later access" in message, message
        assert "when the config changes" in message, message

    @staticmethod
    def _stub_client(health_result):
        class _StubClient:
            def __init__(self, endpoint, api_key="", account="", user="", agent=""):
                self.endpoint = endpoint

            def health(self):
                return health_result

        return _StubClient

    def test_local_autostart_timeout_warning(self):
        self._assert_promises_retry(
            openviking_plugin._runtime_openviking_timeout_message("http://127.0.0.1:1934")
        )

    def test_remote_unreachable_warning(self):
        provider = OpenVikingMemoryProvider()
        provider._endpoint = "https://remote.example"
        warnings: list[str] = []

        provider._handle_runtime_openviking_unreachable(warning_callback=warnings.append)

        assert provider._client is None
        assert len(warnings) == 1
        self._assert_promises_retry(warnings[0])

    def test_local_autostart_refused_warning(self, monkeypatch):
        monkeypatch.setattr(
            openviking_plugin,
            "_start_local_openviking_server",
            lambda endpoint: (
                openviking_plugin._LOCAL_SERVER_FAILED,
                "openviking-server was not found on PATH.",
            ),
        )
        provider = OpenVikingMemoryProvider()
        provider._endpoint = "http://127.0.0.1:1934"
        warnings: list[str] = []

        provider._handle_runtime_openviking_unreachable(warning_callback=warnings.append)

        assert provider._client is None
        assert len(warnings) == 1
        self._assert_promises_retry(warnings[0])

    def test_still_unhealthy_after_autostart_warning(self, monkeypatch):
        monkeypatch.setattr(openviking_plugin, "_VikingClient", self._stub_client(False))
        monkeypatch.setattr(
            openviking_plugin, "_wait_for_openviking_health", lambda endpoint, **kwargs: True
        )
        provider = OpenVikingMemoryProvider()
        provider._endpoint = "http://127.0.0.1:1934"
        warnings: list[str] = []

        provider._finish_runtime_openviking_start(warning_callback=warnings.append)

        assert provider._client is None
        assert len(warnings) == 1
        self._assert_promises_retry(warnings[0])

    def test_attach_failure_after_autostart_warning(self, monkeypatch):
        def _explode(*args, **kwargs):
            raise RuntimeError("connection reset by peer")

        monkeypatch.setattr(openviking_plugin, "_VikingClient", _explode)
        monkeypatch.setattr(
            openviking_plugin, "_wait_for_openviking_health", lambda endpoint, **kwargs: True
        )
        provider = OpenVikingMemoryProvider()
        provider._endpoint = "http://127.0.0.1:1934"
        warnings: list[str] = []

        provider._finish_runtime_openviking_start(warning_callback=warnings.append)

        assert provider._client is None
        assert len(warnings) == 1
        self._assert_promises_retry(warnings[0])

    def test_initialize_responded_unhealthy_warning(self, monkeypatch, tmp_path):
        class _UnhealthyClient:
            def __init__(self, endpoint, api_key="", account="", user="", agent=""):
                self.endpoint = endpoint

            def health_payload(self):
                return {"healthy": False}

            def health(self):
                return False

        monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
        monkeypatch.setenv("OPENVIKING_ENDPOINT", "https://sick.example")
        monkeypatch.setattr(openviking_plugin, "_VikingClient", _UnhealthyClient)
        provider = OpenVikingMemoryProvider()
        warnings: list[str] = []

        provider.initialize("session-1", platform="cli", warning_callback=warnings.append)

        assert provider._client is None
        assert len(warnings) == 1
        self._assert_promises_retry(warnings[0])

    def test_ensure_client_responded_unhealthy_warning(self, monkeypatch, caplog):
        class _UnhealthyClient:
            def __init__(self, endpoint, api_key="", account="", user="", agent=""):
                self.endpoint = endpoint

            def health_payload(self):
                return {"healthy": False}

        monkeypatch.setenv("OPENVIKING_ENDPOINT", "https://sick.example")
        monkeypatch.setattr(openviking_plugin, "_VikingClient", _UnhealthyClient)
        provider = OpenVikingMemoryProvider()
        provider._env_refresh_enabled = True

        with caplog.at_level("WARNING", logger=openviking_plugin.__name__):
            assert provider._ensure_client() is None

        self._assert_promises_retry(caplog.text)

    def test_startup_failure_really_does_reconnect_on_a_later_access(
        self, monkeypatch, tmp_path
    ):
        """The warnings promise a retry — prove the provider delivers one."""
        probes: list[str] = []

        class _FlakyClient:
            def __init__(self, endpoint, api_key="", account="", user="", agent=""):
                self.endpoint = endpoint

            def health(self):
                probes.append(self.endpoint)
                return len(probes) > 1  # down at startup, up on the next access

        monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
        monkeypatch.setenv("OPENVIKING_ENDPOINT", "https://remote.example")
        monkeypatch.setattr(openviking_plugin, "_VikingClient", _FlakyClient)
        provider = OpenVikingMemoryProvider()
        warnings: list[str] = []

        provider.initialize("session-1", platform="cli", warning_callback=warnings.append)
        assert provider._client is None
        assert len(warnings) == 1
        self._assert_promises_retry(warnings[0])

        # A startup failure arms no cooldown, so the very next access re-probes.
        client = provider._ensure_client()
        assert client is not None
        assert client.endpoint == "https://remote.example"
        assert len(probes) == 2
