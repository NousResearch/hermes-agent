"""OpenRouter video_gen plugin — live-catalog shape, per-model clamping, and the submit→poll→download flow."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace

from agent import video_gen_registry
from plugins.video_gen.openrouter import OpenRouterVideoGenProvider, _build_payload

_VEO = {"id": "google/veo-3.1", "name": "Google: Veo 3.1", "supported_durations": [4, 6, 8],
        "supported_resolutions": ["720p", "1080p", "4K"], "supported_aspect_ratios": ["16:9", "9:16"],
        "supported_frame_images": ["first_frame", "last_frame"], "generate_audio": True, "seed": True,
        "pricing_skus": {"duration_seconds_with_audio": "0.40", "duration_seconds_without_audio": "0.20"}}
_HAILUO = {"id": "minimax/hailuo-3-max", "name": "MiniMax: Hailuo 3 Max", "supported_durations": list(range(5, 16)),
           "supported_resolutions": ["768p", "480p"], "supported_aspect_ratios": ["21:9", "16:9", "4:3", "1:1", "3:4", "9:16"],
           "supported_frame_images": ["first_frame"], "generate_audio": False, "seed": False,
           "pricing_skus": {"duration_seconds_480p": "0.05", "duration_seconds_768p": "0.08"}}
_EDIT = {"id": "black-forest-labs/flux-video-edit", "name": "FLUX Video Edit", "supported_durations": None,
         "supported_resolutions": None, "supported_aspect_ratios": None, "supported_frame_images": None,
         "generate_audio": False, "seed": False, "pricing_skus": {"cents_per_second_output": "3"}}


def _provider(monkeypatch, catalog, configured="google/veo-3.1"):
    provider = OpenRouterVideoGenProvider()
    monkeypatch.setattr(provider, "_catalog", lambda: catalog)
    monkeypatch.setattr(provider, "_configured_model", lambda: configured)
    return provider


def test_catalog_drives_picker_rows_and_selected_model_capabilities(monkeypatch):
    """Rows come from the live catalog minus edit/upscale models; capabilities() follows the CONFIGURED
    model (Veo: audio+seed; Hailuo: neither) so the dynamic schema never advertises a dead toggle."""
    provider = _provider(monkeypatch, [_VEO, _HAILUO, _EDIT], configured="google/veo-3.1")
    rows = provider.list_models()
    assert [r["id"] for r in rows] == ["google/veo-3.1", "minimax/hailuo-3-max"]
    assert rows[0]["price"] == "$0.20–0.40/s" and rows[1]["max_duration"] == 15

    veo = provider.capabilities()
    assert veo["supports_audio"] and veo["supports_seed"] and veo["resolutions"] == ["720p", "1080p", "4K"]
    monkeypatch.setattr(provider, "_configured_model", lambda: "minimax/hailuo-3-max")
    hailuo = provider.capabilities()
    assert not hailuo["supports_audio"] and not hailuo["supports_seed"] and hailuo["max_duration"] == 15


def test_payload_clamps_to_model_limits_and_drops_unsupported_toggles():
    payload = _build_payload(_HAILUO, model=_HAILUO["id"], prompt="p", image_url="https://x/a.png",
                             reference_image_urls=["https://x/r.png"], duration=99, aspect_ratio="2:3",
                             resolution="720p", audio=True, seed=7)
    assert payload["duration"] == 15 and payload["resolution"] == "768p" and payload["aspect_ratio"] == "3:4"
    assert payload["frame_images"][0]["frame_type"] == "first_frame"
    assert payload["input_references"] == [{"type": "image_url", "image_url": {"url": "https://x/r.png"}}]
    assert "generate_audio" not in payload and "seed" not in payload  # Hailuo lacks both → would 400

    veo = _build_payload(_VEO, model=_VEO["id"], prompt="p", image_url=None, reference_image_urls=None,
                         duration=5, aspect_ratio="16:9", resolution="1080p", audio=False, seed=7)
    assert veo["duration"] == 4 and veo["generate_audio"] is False and veo["seed"] == 7


@dataclass
class _Response:
    payload: dict
    status_code: int = 200
    text: str = ""

    def json(self):
        return self.payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


@dataclass
class _Session:
    posts: list = field(default_factory=list)
    gets: list = field(default_factory=list)
    polls: list = field(default_factory=lambda: [
        _Response({"id": "job-1", "status": "in_progress"}),
        _Response({"id": "job-1", "status": "completed", "unsigned_urls": ["https://evil.example/steal"],
                   "usage": {"cost": 0.4}})])

    def post(self, url, **kwargs):
        self.posts.append((url, kwargs))
        return _Response({"id": "job-1", "polling_url": f"{url}/job-1", "status": "pending"}, status_code=202)

    def get(self, url, **kwargs):
        self.gets.append((url, kwargs))
        return self.polls.pop(0)

    def close(self):
        pass


def test_generate_submits_polls_and_downloads_from_configured_origin(monkeypatch, tmp_path):
    """The bearer key goes to the poll URL and to ``{base}/videos/{id}/content`` derived from OUR base URL,
    never to a provider-supplied ``unsigned_urls`` host."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.delenv("OPENROUTER_BASE_URL", raising=False)
    provider = _provider(monkeypatch, [_VEO], configured="google/veo-3.1")
    session = _Session()
    monkeypatch.setattr(provider, "_session", lambda: session)
    monkeypatch.setattr("plugins.video_gen.openrouter.time.sleep", lambda s: None)
    saved = []

    def fake_save(url, **kwargs):
        saved.append((url, kwargs))
        return tmp_path / "clip.mp4"
    monkeypatch.setattr("plugins.video_gen.openrouter.save_url_video", fake_save)

    result = provider.generate("a fox", duration=6, resolution="1080p", aspect_ratio="16:9", audio=True)

    assert result["success"], result
    assert result["video"] == str(tmp_path / "clip.mp4") and result["cost"] == 0.4 and result["duration"] == 6
    assert session.posts[0][0] == "https://openrouter.ai/api/v1/videos"
    assert session.posts[0][1]["json"]["model"] == "google/veo-3.1" and session.posts[0][1]["json"]["generate_audio"] is True
    assert session.posts[0][1]["headers"]["Authorization"] == "Bearer sk-or-test"
    assert [g[0] for g in session.gets] == ["https://openrouter.ai/api/v1/videos/job-1"] * 2
    assert saved[0][0] == "https://openrouter.ai/api/v1/videos/job-1/content"
    assert saved[0][1]["headers"]["Authorization"] == "Bearer sk-or-test" and saved[0][1]["require_video_content_type"]
    # Operator-configured origin: a LAN relay must not be refused as SSRF on the first hop.
    assert saved[0][1]["trusted_origin"] is True


def _generate_capturing(monkeypatch, tmp_path, provider):
    """``generate()`` over the fake transport; returns the result and every Authorization header sent."""
    session = _Session()
    monkeypatch.setattr(provider, "_session", lambda: session)
    monkeypatch.setattr("plugins.video_gen.openrouter.time.sleep", lambda s: None)
    saved = []

    def fake_save(url, **kwargs):
        saved.append((url, kwargs))
        return tmp_path / "clip.mp4"
    monkeypatch.setattr("plugins.video_gen.openrouter.save_url_video", fake_save)
    result = provider.generate("a fox")
    bearers = [kwargs["headers"]["Authorization"] for _, kwargs in session.posts + session.gets + saved]
    return result, session, bearers


def _add_pooled_key(key, label):
    from hermes_cli.auth_commands import auth_add_command
    auth_add_command(SimpleNamespace(provider="openrouter", auth_type="api-key", api_key=key, label=label))


def test_pooled_credential_enables_the_backend(monkeypatch, tmp_path):
    """A key added only with ``hermes auth add openrouter`` serves chat and image_gen; video must use it too."""
    _add_pooled_key("sk-or-pool", "pool")
    provider = _provider(monkeypatch, [_VEO])
    assert provider.is_available() is True
    result, _, bearers = _generate_capturing(monkeypatch, tmp_path, provider)
    assert result["success"], result
    assert bearers == ["Bearer sk-or-pool"] * 4  # submit, two polls, download


def test_one_job_keeps_one_credential_while_the_pool_rotates(monkeypatch, tmp_path):
    """A job created under one account is only visible to that account: poll and download must reuse the
    submit key even when a round-robin pool would hand out the other one next."""
    from hermes_constants import get_hermes_home
    (get_hermes_home() / "config.yaml").write_text("credential_pool_strategies:\n  openrouter: round_robin\n")
    _add_pooled_key("sk-or-one", "one")
    _add_pooled_key("sk-or-two", "two")
    result, _, bearers = _generate_capturing(monkeypatch, tmp_path, _provider(monkeypatch, [_VEO]))
    assert result["success"], result
    assert len(bearers) == 4 and len(set(bearers)) == 1 and bearers[0] in {"Bearer sk-or-one", "Bearer sk-or-two"}


def test_multiplexed_profile_spends_its_own_key_not_the_launch_profiles(monkeypatch, tmp_path):
    """On a multiplexed gateway os.environ is the launch profile's .env; a routed turn must sign every request
    with its own profile's key and send it only to its own profile's base URL."""
    from agent.secret_scope import build_profile_secret_scope, reset_secret_scope, set_multiplex_active, set_secret_scope
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-launch-profile")
    monkeypatch.setenv("OPENROUTER_BASE_URL", "https://launch.example/api/v1")
    profile_home = tmp_path / "profile-b"
    profile_home.mkdir()
    (profile_home / ".env").write_text("OPENROUTER_API_KEY=sk-or-profile-b\n"
                                       "OPENROUTER_BASE_URL=https://profile-b.example/api/v1\n")
    provider = _provider(monkeypatch, [_VEO])

    set_multiplex_active(True)
    home_token = set_hermes_home_override(str(profile_home))
    secret_token = set_secret_scope(build_profile_secret_scope(profile_home))
    try:
        result, session, bearers = _generate_capturing(monkeypatch, tmp_path, provider)
    finally:
        reset_secret_scope(secret_token)
        reset_hermes_home_override(home_token)
        set_multiplex_active(False)

    assert result["success"], result
    assert bearers == ["Bearer sk-or-profile-b"] * 4
    assert session.posts[0][0] == "https://profile-b.example/api/v1/videos"


def test_generate_rejects_local_image_paths_before_spending(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    provider = _provider(monkeypatch, [_VEO])
    monkeypatch.setattr(provider, "_session", lambda: (_ for _ in ()).throw(AssertionError("must not submit")))
    result = provider.generate("p", image_url="/home/me/frame.png")
    assert not result["success"] and result["error_type"] == "invalid_request"


def test_register_exposes_openrouter_in_the_video_gen_picker(monkeypatch):
    from hermes_cli import plugins as plugin_loader, tools_config
    from plugins.video_gen.openrouter import register

    class _Context:
        def register_video_gen_provider(self, provider):
            video_gen_registry.register_provider(provider)

    video_gen_registry._reset_for_tests()
    try:
        register(_Context())
        monkeypatch.setattr(plugin_loader, "_ensure_plugins_discovered", lambda: None)
        row = next(r for r in tools_config._plugin_video_gen_providers() if r["video_gen_plugin_name"] == "openrouter")
        assert row["name"] == "OpenRouter" and row["env_vars"][0]["key"] == "OPENROUTER_API_KEY"
    finally:
        video_gen_registry._reset_for_tests()
