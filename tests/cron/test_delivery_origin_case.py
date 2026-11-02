"""Fire-time delivery: a bare platform name matches the origin case-insensitively.

`deliver="Telegram"` with origin platform `"telegram"` must take the origin
fallback, not fall through to the home-only path and fail delivery when no
home channel is set.
"""

from cron.scheduler_delivery import _resolve_single_delivery_target


def _job_with_origin(platform="telegram", chat_id="-1001"):
    return {"id": "j1", "origin": {"platform": platform, "chat_id": chat_id}}


class TestBarePlatformOriginCase:
    def test_capitalized_deliver_hits_origin_fallback(self):
        target = _resolve_single_delivery_target(_job_with_origin(), "Telegram")
        assert target is not None
        assert target["chat_id"] == "-1001"

    def test_lowercase_deliver_still_hits_origin_fallback(self):
        target = _resolve_single_delivery_target(_job_with_origin(), "telegram")
        assert target is not None
        assert target["chat_id"] == "-1001"

    def test_unrelated_platform_does_not_steal_origin(self):
        target = _resolve_single_delivery_target(_job_with_origin(), "discord")
        assert target is None or target.get("chat_id") != "-1001"
