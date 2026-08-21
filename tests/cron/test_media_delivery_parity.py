"""Cron media-delivery parity — manual runs must not silently drop attachments.

Field report (enterprise, v0.20.0/2026.8.3): cron jobs whose output carries
PDF/image MEDIA attachments deliver text+attachment on scheduled ticks but
text-only on manual ``hermes cron run <job-id>``. Same box, same token, same
scopes — the divergence is process context and error visibility, not
credentials.

Three defects, each pinned here:

1. STANDALONE LANE SWALLOWS WARNINGS: platform standalone senders (Slack,
   Discord, ...) report per-file upload failures in ``result["warnings"]``
   while returning ``success: True``. ``_deliver_result`` only checks
   ``result["error"]`` — an attachment failure vanishes: job marked ok, no
   delivery error, text delivered, file gone.

2. LIVE-ADAPTER LANE SWALLOWS MEDIA FAILURES: ``_send_media_via_adapter``
   logs failed/skipped sends at WARNING and returns None — the caller
   cannot surface them into delivery_errors either.

3. MEDIA-POLICY ENV BRIDGE IS GATEWAY-ONLY: ``gateway.strict`` /
   ``media_delivery_allow_dirs`` / ``trust_recent_files`` from config.yaml
   are exported to the env vars ``validate_media_delivery_path`` reads ONLY
   in gateway startup (gateway/run.py). A CLI-process manual run filters
   MEDIA paths under different policy than the gateway's scheduled tick.
"""

import os
from pathlib import Path

import pytest

from cron.scheduler import _deliver_result, _send_media_via_adapter


@pytest.fixture()
def media_file(tmp_path):
    p = tmp_path / "report.pdf"
    p.write_bytes(b"%PDF-1.4 test")
    return str(p)


@pytest.fixture()
def slack_job():
    return {
        "id": "job-media-1",
        "name": "media-parity",
        "deliver": "slack:D0123456789",
        "origin": {"platform": "slack", "chat_id": "D0123456789"},
    }


def _install_fake_slack_sender(monkeypatch, result_factory):
    """Register a fake slack standalone sender through the platform registry."""
    calls = []

    async def fake_sender(pconfig, chat_id, message, *, thread_id=None,
                          media_files=None, force_document=False, caption=None):
        calls.append({
            "chat_id": chat_id,
            "message": message,
            "media_files": list(media_files or []),
            "caption": caption,
        })
        return result_factory(calls[-1])

    import gateway.platform_registry as reg_mod

    entry = reg_mod.platform_registry.get("slack")
    if entry is None:
        # Populate the registry the same way tools/send_message_tool does.
        import hermes_cli.plugins as hp_boot

        hp_boot.discover_plugins()
        entry = reg_mod.platform_registry.get("slack")
    if entry is None:
        pytest.skip("slack platform entry not registered in this environment")
    monkeypatch.setattr(entry, "standalone_sender_fn", fake_sender)
    # Keep plugin discovery from replacing our fake mid-test.
    import hermes_cli.plugins as hp

    monkeypatch.setattr(hp, "discover_plugins", lambda *a, **k: None)
    return calls


@pytest.fixture()
def slack_platform_config(monkeypatch, tmp_path):
    home = tmp_path / "hermes-home"
    home.mkdir()
    (home / "config.yaml").write_text(
        "platforms:\n  slack:\n    enabled: true\n    token: xoxb-test\n"
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    # Config caches are process-global; clear them so the temp HERMES_HOME wins.
    try:
        from gateway import config as gwconfig

        for attr in ("_config_cache", "_CONFIG_CACHE", "_cached_config"):
            if hasattr(gwconfig, attr):
                monkeypatch.setattr(gwconfig, attr, None)
    except Exception:
        pass
    return home


class TestStandaloneWarningsSurfaced:
    """Defect 1: sender warnings must become delivery errors."""

    def test_upload_warning_becomes_delivery_error(
        self, monkeypatch, slack_platform_config, slack_job, media_file
    ):
        _install_fake_slack_sender(
            monkeypatch,
            lambda call: {
                "success": True,
                "platform": "slack",
                "chat_id": call["chat_id"],
                "message_id": "1.2",
                "warnings": [
                    f"Failed to send media {media_file}: Slack API error: missing_scope"
                ],
            },
        )
        err = _deliver_result(
            slack_job, f"Report ready.\n\nMEDIA:{media_file}", adapters=None, loop=None
        )
        assert err is not None, (
            "an attachment-upload failure reported via warnings must surface "
            "as a delivery error, not vanish"
        )
        assert "missing_scope" in err or "Failed to send media" in err

    def test_clean_delivery_still_returns_none(
        self, monkeypatch, slack_platform_config, slack_job, media_file
    ):
        _install_fake_slack_sender(
            monkeypatch,
            lambda call: {
                "success": True,
                "platform": "slack",
                "chat_id": call["chat_id"],
                "message_id": "1.2",
            },
        )
        err = _deliver_result(
            slack_job, f"Report ready.\n\nMEDIA:{media_file}", adapters=None, loop=None
        )
        assert err is None

    def test_media_actually_reaches_sender(
        self, monkeypatch, slack_platform_config, slack_job, media_file
    ):
        calls = _install_fake_slack_sender(
            monkeypatch,
            lambda call: {"success": True, "chat_id": call["chat_id"], "message_id": "1.2"},
        )
        _deliver_result(
            slack_job, f"Report ready.\n\nMEDIA:{media_file}", adapters=None, loop=None
        )
        sent_media = [m for c in calls for m in c["media_files"]]
        assert any(media_file in m[0] for m in sent_media)


class TestLiveAdapterMediaFailuresSurfaced:
    """Defect 2: _send_media_via_adapter must report failures to its caller."""

    def test_failed_media_send_returns_errors(self, media_file, slack_job):
        import asyncio

        class FailingAdapter:
            platform = None

            async def send_document(self, chat_id, file_path, metadata=None):
                from types import SimpleNamespace

                return SimpleNamespace(success=False, error="upload rejected")

        loop = asyncio.new_event_loop()
        try:
            import threading

            t = threading.Thread(target=loop.run_forever, daemon=True)
            t.start()
            errors = _send_media_via_adapter(
                FailingAdapter(), "D01", [(media_file, False)], None, loop, slack_job
            )
            assert errors, (
                "a failed media send must be returned to the caller, not only logged"
            )
            assert any("upload rejected" in e for e in errors)
        finally:
            loop.call_soon_threadsafe(loop.stop)
            t.join(timeout=5)
            loop.close()

    def test_dropped_unsafe_path_is_reported(self, slack_job):
        import asyncio

        class NeverCalledAdapter:
            platform = None

            async def send_document(self, chat_id, file_path, metadata=None):
                raise AssertionError("should not be called for a dropped path")

        loop = asyncio.new_event_loop()
        try:
            import threading

            t = threading.Thread(target=loop.run_forever, daemon=True)
            t.start()
            errors = _send_media_via_adapter(
                NeverCalledAdapter(), "D01",
                [("/nonexistent/definitely-missing.pdf", False)],
                None, loop, slack_job,
            )
            assert errors, "a filtered-out MEDIA path must be reported, not silent"
        finally:
            loop.call_soon_threadsafe(loop.stop)
            t.join(timeout=5)
            loop.close()


class TestMediaPolicyEnvBridge:
    """Defect 3: media-policy config must apply outside the gateway process."""

    def test_bridge_helper_exists_and_applies_config(self, monkeypatch, tmp_path):
        home = tmp_path / "hermes-home"
        home.mkdir()
        allow_dir = tmp_path / "reports"
        allow_dir.mkdir()
        (home / "config.yaml").write_text(
            "gateway:\n"
            "  strict: true\n"
            f"  media_delivery_allow_dirs: [{str(allow_dir)!r}]\n"
            "  trust_recent_files: false\n"
        )
        monkeypatch.setenv("HERMES_HOME", str(home))
        for var in (
            "HERMES_MEDIA_DELIVERY_STRICT",
            "HERMES_MEDIA_ALLOW_DIRS",
            "HERMES_MEDIA_TRUST_RECENT_FILES",
        ):
            monkeypatch.delenv(var, raising=False)

        from gateway.media_policy import apply_media_policy_env

        apply_media_policy_env()

        assert os.environ.get("HERMES_MEDIA_DELIVERY_STRICT") == "1"
        assert str(allow_dir) in os.environ.get("HERMES_MEDIA_ALLOW_DIRS", "")
        assert os.environ.get("HERMES_MEDIA_TRUST_RECENT_FILES") == "0"

    def test_standalone_filter_honors_bridged_allowlist(self, monkeypatch, tmp_path):
        """End-to-end: strict-mode file inside allow_dirs passes validation
        in a process that never ran gateway startup."""
        home = tmp_path / "hermes-home"
        home.mkdir()
        allow_dir = tmp_path / "reports"
        allow_dir.mkdir()
        media = allow_dir / "report.pdf"
        media.write_bytes(b"%PDF-1.4 x")
        # Make the file OLD so recency-trust cannot save it: only the
        # allowlist can accept it, proving the bridge ran.
        old = 1_600_000_000
        os.utime(media, (old, old))
        (home / "config.yaml").write_text(
            "gateway:\n"
            "  strict: true\n"
            f"  media_delivery_allow_dirs: [{str(allow_dir)!r}]\n"
        )
        monkeypatch.setenv("HERMES_HOME", str(home))
        for var in ("HERMES_MEDIA_DELIVERY_STRICT", "HERMES_MEDIA_ALLOW_DIRS"):
            monkeypatch.delenv(var, raising=False)

        from gateway.media_policy import apply_media_policy_env

        apply_media_policy_env()

        from gateway.platforms.base import BasePlatformAdapter

        kept = BasePlatformAdapter.filter_media_delivery_paths([(str(media), False)])
        assert kept, (
            "with the policy bridged, an allowlisted file must survive strict "
            "filtering in a non-gateway process"
        )

    def test_deliver_result_runs_bridge(self, monkeypatch, tmp_path, media_file):
        """_deliver_result itself must apply the bridge before filtering.

        Realistic enterprise shape: HERMES_MEDIA_DELIVERY_STRICT=1 arrives via
        .env (loaded by BOTH processes), but the allowlist lives in
        config.yaml's gateway block — bridged only at gateway boot. Without
        the bridge, a CLI manual run is strict WITHOUT the allowlist and
        silently drops the attachment the scheduled (gateway) run delivers.
        """
        home = tmp_path / "hermes-home"
        home.mkdir()
        media_dir = str(Path(media_file).parent)
        (home / "config.yaml").write_text(
            "platforms:\n  slack:\n    enabled: true\n    token: xoxb-test\n"
            "gateway:\n"
            "  strict: true\n"
            f"  media_delivery_allow_dirs: [{media_dir!r}]\n"
            "  trust_recent_files: false\n"
        )
        monkeypatch.setenv("HERMES_HOME", str(home))
        # Strict comes from the shared .env in both processes...
        monkeypatch.setenv("HERMES_MEDIA_DELIVERY_STRICT", "1")
        monkeypatch.setenv("HERMES_MEDIA_TRUST_RECENT_FILES", "0")
        # ...but the allowlist is config-only (gateway-boot bridge).
        monkeypatch.delenv("HERMES_MEDIA_ALLOW_DIRS", raising=False)
        old = 1_600_000_000
        os.utime(media_file, (old, old))

        calls = _install_fake_slack_sender(
            monkeypatch,
            lambda call: {"success": True, "chat_id": call["chat_id"], "message_id": "1.2"},
        )
        job = {
            "id": "job-media-3",
            "name": "bridge",
            "deliver": "slack:D0123456789",
            "origin": {"platform": "slack", "chat_id": "D0123456789"},
        }
        _deliver_result(job, f"Report.\n\nMEDIA:{media_file}", adapters=None, loop=None)
        sent_media = [m for c in calls for m in c["media_files"]]
        assert any(media_file in m[0] for m in sent_media), (
            "manual-run delivery must filter media under the same configured "
            "policy as the gateway process (allowlisted file was dropped)"
        )
