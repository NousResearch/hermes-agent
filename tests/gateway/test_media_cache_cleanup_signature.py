"""Every cache cleanup the gateway housekeeping loop iterates accepts the loop's call shape.

The loop calls each entry as ``fn(max_age_hours=24)``; a cleanup renamed to another kwarg
raised ``TypeError`` inside housekeeping on every tick (the terminal temp sweep did, once).
"""
import inspect


def test_every_housekeeping_cache_cleanup_accepts_max_age_hours(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from tools.environments.local import cleanup_terminal_temp_cache
    from tools.bot_mode_dm import cleanup_bot_dm_cache
    from tools.bot_relay import cleanup_bot_relay_artifacts
    from tools.tool_result_storage import cleanup_spillover_cache
    from gateway.platforms.base import (
        cleanup_audio_cache, cleanup_document_cache, cleanup_image_cache,
        cleanup_screenshot_cache, cleanup_video_cache,
    )
    cleanups = (cleanup_image_cache, cleanup_document_cache, cleanup_audio_cache, cleanup_video_cache,
                cleanup_screenshot_cache, cleanup_spillover_cache, cleanup_terminal_temp_cache,
                cleanup_bot_dm_cache, cleanup_bot_relay_artifacts)
    for fn in cleanups:
        inspect.signature(fn).bind(max_age_hours=24)
        assert isinstance(fn(max_age_hours=24), int), fn.__name__
