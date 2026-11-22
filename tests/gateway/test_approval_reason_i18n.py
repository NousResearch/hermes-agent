"""The gateway renders structured approval reasons in the routed profile's language.

Related to #131450. Both the button card and the typed-text fallback get the same reason string; each
item falls back to its own redacted English text when the catalog has no entry for it, and a payload
without structured reasons keeps the legacy ``description``. Profiles are real temp homes with their
own ``display.language`` and locale overlay (synthetic strings), bound the way a routed turn binds
them.
"""
import asyncio
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from agent import i18n
from gateway.run_turn_runner import TurnRunner
from hermes_constants import reset_hermes_home_override, set_hermes_home_override

_FAKE_GHP = "ghp_" + "X" * 36
INVISIBLE = "invisible or bidirectional Unicode control character (possible obfuscation)"
REASONS = [
    {"slug": "recursive_delete", "text": "recursive delete"},
    {"slug": None, "text": "custom plugin rule for token " + _FAKE_GHP},
    {"slug": "forged", "text": "forged slug text"},
    {"slug": "invisible_unicode", "text": INVISIBLE},
]


def _home(root, name, lang, overlay):
    home = root / name
    (home / "locales").mkdir(parents=True)
    (home / "config.yaml").write_text(f"display:\n  language: {lang}\n", encoding="utf-8")
    (home / "locales" / f"{lang}.yaml").write_text(overlay, encoding="utf-8")
    return home


@pytest.fixture
def profiles(tmp_path, monkeypatch):
    monkeypatch.delenv("HERMES_LANGUAGE", raising=False)
    de = _home(tmp_path, "alpha", "de",
               "gateway:\n  exec_approval:\n    reason_joiner: \" | \"\n    reasons:\n"
               "      recursive_delete: \"rekursives Loeschen (Fixture)\"\n"
               "      forged: \"MUST NOT APPEAR\"\n")
    fr = _home(tmp_path, "beta", "fr",
               "gateway:\n  exec_approval:\n    reasons:\n      recursive_delete: \"suppression recursive (fixture)\"\n")
    i18n.reset_language_cache()
    yield {"de": de, "fr": fr}
    i18n.reset_language_cache()


@contextmanager
def _routed(home):
    token = set_hermes_home_override(home)
    try:
        yield
    finally:
        reset_hermes_home_override(token)


def _render(approval, *, card_success):
    """Drive ``TurnRunner._approval_notify_sync``; return the reason text each surface received."""
    seen = []

    class Adapter:
        def pause_typing_for_chat(self, chat_id):
            pass

        async def send_exec_approval(self, **kwargs):
            seen.append(("card", kwargs["description"]))
            return SimpleNamespace(success=card_success, error=None, message_id=None)

        async def send(self, chat_id, text, **kwargs):
            seen.append(("text", text))
            return SimpleNamespace(success=True, message_id=None)

    async def exercise():
        ctx = SimpleNamespace(_status_adapter=Adapter(), _status_chat_id="fixture", _status_thread_metadata=None,
                              session_key="reason-fixture", _loop_for_step=asyncio.get_running_loop(),
                              stream_consumer_holder=[None])
        await asyncio.to_thread(TurnRunner(SimpleNamespace(), ctx)._approval_notify_sync, approval)

    asyncio.run(exercise())
    return seen


@pytest.mark.parametrize("card_success", [True, False], ids=["card", "text-fallback"])
def test_reasons_render_in_each_routed_profile_language(profiles, card_success):
    approval = {"command": "rm -rf build/", "description": "english description", "reasons": REASONS}
    with _routed(profiles["de"]):
        alpha = _render(approval, card_success=card_success)
    with _routed(profiles["fr"]):
        beta = _render(approval, card_success=card_success)
    with _routed(profiles["de"]):
        alpha_again = _render(approval, card_success=card_success)

    assert [kind for kind, _ in alpha] == (["card"] if card_success else ["card", "text"])
    alpha_reason, beta_reason = alpha[-1][1], beta[-1][1]
    with _routed(profiles["de"]):
        bundled_de = i18n.lookup("gateway.exec_approval.reasons.invisible_unicode")
    # Ordered: overlay translation, the item's own redacted text for an unmapped or forged slug,
    # then the bundled translation, joined with the profile's joiner.
    assert bundled_de and bundled_de != INVISIBLE
    assert "rekursives Loeschen (Fixture) | custom plugin rule for token" in alpha_reason
    assert "forged slug text | " + bundled_de in alpha_reason
    with _routed(profiles["fr"]):
        fr_joiner = i18n.t("gateway.exec_approval.reason_joiner")
    assert f"suppression recursive (fixture){fr_joiner}custom plugin rule" in beta_reason
    assert "rekursives" not in beta_reason
    assert alpha_again == alpha  # profile B's render did not leak into profile A
    for _kind, text in alpha + beta:
        assert _FAKE_GHP not in text
        assert "MUST NOT APPEAR" not in text
        assert "english description" not in text
    assert approval["reasons"] == REASONS  # the payload is display input, never rewritten


def test_payload_without_reasons_keeps_the_legacy_description(profiles):
    with _routed(profiles["de"]):
        seen = _render({"command": "rm -rf build/", "description": "legacy reason text"}, card_success=True)
    assert seen == [("card", "legacy reason text")]


def test_a_language_without_its_own_entry_keeps_the_payload_text(tmp_path, monkeypatch):
    """Codex review: a pack/overlay language that omits a reason must not borrow an English catalog
    value (here an English overlay) in place of the payload's own redacted text."""
    monkeypatch.delenv("HERMES_LANGUAGE", raising=False)
    home = _home(tmp_path, "gamma", "eo",
                 "gateway:\n  exec_approval:\n    reasons:\n      sql_drop: \"SQL DROP (eo fixture)\"\n")
    (home / "locales" / "en.yaml").write_text(
        "gateway:\n  exec_approval:\n    reasons:\n      recursive_delete: \"EN OVERLAY MUST NOT APPEAR\"\n",
        encoding="utf-8")
    i18n.reset_language_cache()
    approval = {"command": "rm -rf build/", "description": "unused",
                "reasons": [{"slug": "recursive_delete", "text": "recursive delete (payload)"},
                            {"slug": "sql_drop", "text": "SQL DROP"}]}
    try:
        with _routed(home):
            assert i18n.get_language() == "eo"
            seen = _render(approval, card_success=True)
    finally:
        i18n.reset_language_cache()
    assert seen == [("card", "recursive delete (payload); SQL DROP (eo fixture)")]
