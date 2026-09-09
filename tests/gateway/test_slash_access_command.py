"""Tests for gateway.slash_commands_access — the ``/access`` command.

Covers the three layers the command touches: reference resolution (adapter resolver →
generic fallbacks), the dual write (live adapter state + routed-profile config.yaml), and
the reply contract (ambiguity asks, never guesses).  The fake adapter mirrors the
WhatsApp-family shape (``_allow_from`` set + ``config.extra``) because that is the
production deployment; the generic fallbacks are exercised with a bare adapter.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

import gateway.run as gateway_run
from gateway.config import Platform
from gateway.platforms.whatsapp_common import WhatsAppBehaviorMixin
from gateway.slash_commands_access import AccessResolution, GatewayAccessCommandsMixin


class _Source:
    def __init__(self, platform: str = "waha", chat_id: str = "6281@s.whatsapp.net"):
        from gateway.config import Platform
        self.platform = Platform(platform)
        self.chat_id = chat_id
        self.user_id = "6285157813352@s.whatsapp.net"


class _Event:
    def __init__(self, args: str = "", source=None, metadata=None, reply_to_author_id=None):
        self._args = args
        self.source = source or _Source()
        self.metadata = metadata or {}
        self.reply_to_author_id = reply_to_author_id

    def get_command_args(self) -> str:
        return self._args


class _Config:
    def __init__(self, extra: dict):
        self.extra = extra


class _FakeAdapter:
    """Plain platform shape (Telegram/Discord-style): live sets + config.extra, no
    WhatsApp matcher — digit strings pass through as canonical ids."""

    def __init__(self, extra: dict):
        self.config = _Config(extra)
        self._allow_from = set(extra.get("allow_from") or [])
        self._group_allow_from = set(extra.get("group_allow_from") or [])
        self._dm_policy = extra.get("dm_policy", "allowlist")
        self._group_policy = extra.get("group_policy", "allowlist")
        self.name = "waha"


class _FakeWhatsAppAdapter(_FakeAdapter, WhatsAppBehaviorMixin):
    """WhatsApp-family shape: the REAL WhatsAppBehaviorMixin supplies the allowlist
    matcher and phone normalization (``_access_phone_jid``), so the generic fallback
    path is exercised exactly as in production."""

    def __init__(self, extra: dict):
        _FakeAdapter.__init__(self, extra)
        self._bot_ids = {"6287784454555@c.us"}


class _PlatformConfig:
    def __init__(self, extra: dict):
        self.extra = extra


class _GatewayConfig:
    def __init__(self, extras: dict):
        self.platforms = {p: _PlatformConfig(e) for p, e in extras.items()}


class _Runner(GatewayAccessCommandsMixin):
    def __init__(self, adapters: dict, admins: bool = True):
        from gateway.config import Platform
        # GatewayRunner keys self.adapters by Platform; accept the enum or its value string.
        self.adapters = {
            (k if isinstance(k, Platform) else Platform(k)): v
            for k, v in adapters.items()
        }
        # Fail-closed needs an admin list, like production; opt out per-test.
        admin_extra = {"allow_admin_from": ["6285157813352@s.whatsapp.net"],
                       "group_allow_admin_from": ["6285157813352@s.whatsapp.net"]}
        self.config = _GatewayConfig(
            {p: dict(admin_extra) if admins else {} for p in self.adapters}
        )


@pytest.fixture
def homes(tmp_path, monkeypatch):
    home = tmp_path / "profiles" / "osis"
    home.mkdir(parents=True)
    (home / "config.yaml").write_text(yaml.safe_dump({
        "platforms": {"waha": {
            "enabled": True, "dm_policy": "allowlist",
            "allow_from": ["6285157813352@s.whatsapp.net"],
            "group_allow_from": ["120363427512131387@g.us"],
        }},
    }))
    monkeypatch.setattr(gateway_run, "_hermes_home", home)
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home


def _config_yaml(home: Path) -> dict:
    return yaml.safe_load((home / "config.yaml").read_text())


# ---------------------------------------------------------------------------
# resolution — generic fallbacks
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generic_resolve_strips_discord_mention_wrapper(homes):
    runner = _Runner({"discord": _FakeAdapter({})})
    res = await runner._access_resolve(
        runner.adapters[Platform("discord")], _Event("", _Source("discord")), "user", "<@123456789>")
    assert res.canonical == "123456789"


@pytest.mark.asyncio
async def test_generic_resolve_uses_event_mention_metadata(homes):
    runner = _Runner({"waha": _FakeAdapter({})})
    event = _Event("", metadata={"mentions": [{"id": "6289682642242@s.whatsapp.net", "label": "Niken"}]})
    res = await runner._access_resolve(runner.adapters[Platform("waha")], event, "user", "@Niken")
    assert res.canonical == "6289682642242@s.whatsapp.net"


@pytest.mark.asyncio
async def test_generic_resolve_reply_keyword_uses_reply_author(homes):
    runner = _Runner({"waha": _FakeAdapter({})})
    event = _Event("", reply_to_author_id="6289603167061@s.whatsapp.net")
    res = await runner._access_resolve(runner.adapters[Platform("waha")], event, "user", "reply")
    assert res.canonical == "6289603167061@s.whatsapp.net"


@pytest.mark.asyncio
async def test_generic_resolve_passes_through_raw_ids(homes):
    runner = _Runner({"telegram": _FakeAdapter({})})
    res = await runner._access_resolve(runner.adapters[Platform("telegram")], _Event("", _Source("telegram")), "user", "123456789")
    assert res.canonical == "123456789"


# ---------------------------------------------------------------------------
# apply — dual write: live adapter state + routed-profile config.yaml
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_allow_user_adds_to_live_set_and_config(homes):
    adapter = _FakeWhatsAppAdapter({"allow_from": ["6285157813352@s.whatsapp.net"]})
    runner = _Runner({"waha": adapter})
    reply = await runner._handle_access_command(_Event("allow user 089682642242"))
    assert "✅" in reply
    assert "6289682642242@s.whatsapp.net" in adapter._allow_from
    block = _config_yaml(homes)["platforms"]["waha"]
    assert "6289682642242@s.whatsapp.net" in block["allow_from"]
    # The bridged extra (what the authz union reads) is updated too.
    assert "6289682642242@s.whatsapp.net" in adapter.config.extra["allow_from"]


@pytest.mark.asyncio
async def test_deny_user_removes_from_both_layers(homes):
    adapter = _FakeWhatsAppAdapter({"allow_from": ["6285157813352@s.whatsapp.net", "6289603167061@s.whatsapp.net"]})
    runner = _Runner({"waha": adapter})
    reply = await runner._handle_access_command(_Event("deny user 6289603167061@s.whatsapp.net"))
    assert "no longer allowed" in reply
    assert "6289603167061@s.whatsapp.net" not in adapter._allow_from
    assert "6289603167061@s.whatsapp.net" not in _config_yaml(homes)["platforms"]["waha"]["allow_from"]
    assert "6285157813352@s.whatsapp.net" in adapter._allow_from  # untouched


@pytest.mark.asyncio
async def test_allow_group_and_deny_group_are_symmetric(homes):
    adapter = _FakeWhatsAppAdapter({"group_allow_from": ["120363427512131387@g.us"]})
    runner = _Runner({"waha": adapter})
    await runner._handle_access_command(_Event("allow group 120363426491664891@g.us"))
    assert "120363426491664891@g.us" in adapter._group_allow_from
    await runner._handle_access_command(_Event("deny group 120363427512131387@g.us"))
    assert "120363427512131387@g.us" not in adapter._group_allow_from
    block = _config_yaml(homes)["platforms"]["waha"]
    assert block["group_allow_from"] == ["120363426491664891@g.us"]


@pytest.mark.asyncio
async def test_allow_is_idempotent_and_deny_missing_is_a_noop(homes):
    adapter = _FakeWhatsAppAdapter({"allow_from": ["6285157813352@s.whatsapp.net"]})
    runner = _Runner({"waha": adapter})
    reply = await runner._handle_access_command(_Event("allow user 6285157813352@s.whatsapp.net"))
    assert "already" in reply
    reply = await runner._handle_access_command(_Event("deny user 6289999999999@s.whatsapp.net"))
    assert "was not" in reply
    assert len(adapter._allow_from) == 1


@pytest.mark.asyncio
async def test_loose_membership_matches_jid_variants_of_same_number(homes):
    # The same human stored as @s.whatsapp.net must not be re-added as @c.us.
    adapter = _FakeWhatsAppAdapter({"allow_from": ["6285157813352@s.whatsapp.net"]})
    runner = _Runner({"waha": adapter})
    reply = await runner._handle_access_command(_Event("allow user 6285157813352@s.whatsapp.net"))
    assert "already" in reply
    assert adapter._allow_from == {"6285157813352@s.whatsapp.net"}


@pytest.mark.asyncio
async def test_persist_writes_routed_profile_not_default(tmp_path, monkeypatch):
    """Regression shape from #87939: the write must land in the routed profile's
    config.yaml (HERMES_HOME), never the launch home."""
    default_home = tmp_path / "default"
    routed_home = tmp_path / "profiles" / "osis"
    default_home.mkdir()
    routed_home.mkdir(parents=True)
    (default_home / "config.yaml").write_text("platforms: {}\n")
    (routed_home / "config.yaml").write_text(yaml.safe_dump(
        {"platforms": {"waha": {"allow_from": ["6285157813352@s.whatsapp.net"]}}}))
    monkeypatch.setattr(gateway_run, "_hermes_home", default_home)
    monkeypatch.setenv("HERMES_HOME", str(default_home))

    from gateway.run import _profile_runtime_scope

    adapter = _FakeWhatsAppAdapter({"allow_from": ["6285157813352@s.whatsapp.net"]})
    runner = _Runner({"waha": adapter})
    with _profile_runtime_scope(routed_home):
        await runner._handle_access_command(_Event("allow user 6289603167061@s.whatsapp.net"))

    assert "6289603167061@s.whatsapp.net" in _config_yaml(routed_home)["platforms"]["waha"]["allow_from"]
    assert _config_yaml(default_home)["platforms"] == {}


# ---------------------------------------------------------------------------
# usage + list rendering
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_bad_usage_returns_help(homes):
    runner = _Runner({"waha": _FakeAdapter({})})
    for args in ("", "list extra stuff ok", "allow user", "allow vehicle 123", "block user 123"):
        reply = await runner._handle_access_command(_Event(args))
        assert "Usage:" in reply, args


@pytest.mark.asyncio
async def test_list_renders_policies_and_counts(homes):
    adapter = _FakeWhatsAppAdapter({
        "allow_from": ["6285157813352@s.whatsapp.net"],
        "group_allow_from": ["120363427512131387@g.us"],
        "dm_policy": "allowlist", "group_policy": "allowlist",
    })
    runner = _Runner({"waha": adapter})
    reply = await runner._handle_access_command(_Event("list"))
    assert "DM policy: *allowlist*" in reply
    assert "6285157813352@s.whatsapp.net" in reply
    assert "120363427512131387@g.us" in reply


# ---------------------------------------------------------------------------
# WAHA resolver — phone normalization + group-name lookup
# ---------------------------------------------------------------------------


class _WahaLikeAdapter(_FakeWhatsAppAdapter):
    """Enough of WahaAdapter's /access surface to test the core wiring without a live
    WAHA instance: the real resolver methods are exercised in
    tests/plugins/test_waha_access_resolver.py against a stubbed _request."""

    def __init__(self, extra, groups):
        super().__init__(extra)
        self._groups = groups
        self._bot_ids = {"6287784454555@s.whatsapp.net"}

    async def resolve_access_ref(self, ref, *, scope="user", event=None):
        from gateway.whatsapp_identity import normalize_phone_e164
        text = str(ref or "").strip()
        if scope == "group":
            matches = [(gid, g["subject"]) for gid, g in self._groups.items()
                       if g["subject"].lower() == text.lower()]
            if len(matches) == 1:
                return AccessResolution(canonical=matches[0][0], label=matches[0][1])
            return AccessResolution(canonical=text) if "@" in text else None
        digits = "".join(c for c in text if c.isdigit())
        if digits and text[0].isnumeric():
            e164 = normalize_phone_e164(text, "62")
            return AccessResolution(canonical=f"{e164}@s.whatsapp.net") if e164 else AccessResolution()
        return None


@pytest.mark.asyncio
async def test_waha_style_resolver_normalizes_local_phone(homes):
    groups = {"120363426491664891@g.us": {"subject": "kami akan berubah mas mba"}}
    adapter = _WahaLikeAdapter({"allow_from": []}, groups)
    runner = _Runner({"waha": adapter})
    reply = await runner._handle_access_command(_Event("allow user 089682642242"))
    assert "6289682642242@s.whatsapp.net" in adapter._allow_from
    assert "✅" in reply


@pytest.mark.asyncio
async def test_waha_style_resolver_resolves_group_by_name(homes):
    groups = {"120363426491664891@g.us": {"subject": "kami akan berubah mas mba"}}
    adapter = _WahaLikeAdapter({"group_allow_from": []}, groups)
    runner = _Runner({"waha": adapter})
    reply = await runner._handle_access_command(_Event('allow group "kami akan berubah mas mba"'))
    assert "120363426491664891@g.us" in adapter._group_allow_from
    assert "✅" in reply


@pytest.mark.asyncio
async def test_raw_name_entry_never_matches_gate():
    # Invariant (documents the dead-entry gap): a raw-text allowlist entry such
    # as a display name must never match the WhatsApp DM gate, which only
    # matches phone/JID/LID identity forms.
    assert WhatsAppBehaviorMixin._matches_whatsapp_allowlist(
        "62812xxxx@s.whatsapp.net", ["Farel"]) is False


@pytest.mark.asyncio
async def test_telegram_substring_candidate_order():
    # Core renders candidates as (id, label); every resolver must return
    # that order so the copy-pasted id is usable.
    from gateway.slash_commands_resolve_platforms import PlatformAccessResolversMixin

    class _Tg(PlatformAccessResolversMixin):
        name = "telegram"
        _bot = None
        _seen_chats = {"kelas xi-c": "-100123", "kelas xi-d": "-100124"}

    res = await _Tg().resolve_access_ref("kelas", scope="group")
    assert res.candidates, res
    for cid, _label in res.candidates:
        assert cid.lstrip("-").isdigit(), res.candidates
    single = await _Tg().resolve_access_ref("xi-c", scope="group")
    assert single.canonical == "-100123", single
    assert single.label == "kelas xi-c", single


@pytest.mark.asyncio
async def test_access_refuses_dead_user_ref(homes):
    # Bare display names must not land in the allowlist as dead entries.
    adapter = _FakeWhatsAppAdapter({"allow_from": []})
    runner = _Runner({"waha": adapter})
    reply = await runner._handle_access_command(_Event("allow user Farel"))
    assert "Could not resolve" in reply
    assert adapter._allow_from == set()


@pytest.mark.asyncio
async def test_access_refuses_dead_group_ref(homes):
    # Unresolvable group names must not land in the allowlist either.
    adapter = _FakeWhatsAppAdapter({"group_allow_from": []})
    runner = _Runner({"waha": adapter})
    reply = await runner._handle_access_command(_Event("allow group Kelas XI-C"))
    assert "Could not resolve" in reply
    assert adapter._group_allow_from == set()


@pytest.mark.asyncio
async def test_access_warns_when_policy_not_allowlist(homes):
    # Editing a list the policy never consults must say so, not claim success.
    adapter = _FakeWhatsAppAdapter({"dm_policy": "open", "allow_from": []})
    runner = _Runner({"waha": adapter})
    reply = await runner._handle_access_command(_Event("allow user 089682642242"))
    assert "Stored nothing" in reply
    assert adapter._allow_from == set()


@pytest.mark.asyncio
async def test_access_allows_good_phone(homes):
    # Guard against over-blocking: a real local number still works.
    adapter = _FakeWhatsAppAdapter({"allow_from": []})
    runner = _Runner({"waha": adapter})
    reply = await runner._handle_access_command(_Event("allow user 089682642242"))
    assert "✅" in reply
    assert "6289682642242@s.whatsapp.net" in adapter._allow_from


def test_concurrent_access_edits_keep_both_entries(homes, monkeypatch):
    # Two edits racing through the whole read-compute-write must merge,
    # not clobber. The slowed file read widens the race window so an
    # unlocked implementation deterministically loses one entry.
    import asyncio
    import threading

    import hermes_cli.config as cli_config

    real_read = cli_config.read_user_config_raw

    def slow_read(path):
        import time
        time.sleep(0.3)
        return real_read(path)

    monkeypatch.setattr(cli_config, "read_user_config_raw", slow_read)
    adapter = _FakeWhatsAppAdapter({"allow_from": []})
    runner = _Runner({"waha": adapter})

    def allow(number):
        asyncio.run(runner._handle_access_command(_Event(f"allow user {number}")))

    first, second = threading.Thread(target=allow, args=("089682642242",)), \
        threading.Thread(target=allow, args=("089682642243",))
    first.start()
    second.start()
    first.join()
    second.join()
    block = _config_yaml(homes)["platforms"]["waha"]
    assert "6289682642242@s.whatsapp.net" in block["allow_from"]
    assert "6289682642243@s.whatsapp.net" in block["allow_from"]


@pytest.mark.asyncio
async def test_access_merges_twin_config_blocks(homes):
    # The loader reads top-level <name>: over platforms.<name>; /access must
    # write the winner and fold the loser in, never diverge the two.
    import yaml as _yaml

    cfg = _config_yaml(homes)
    cfg["waha"] = {"allow_from": ["6281111111111@s.whatsapp.net"]}
    (homes / "config.yaml").write_text(_yaml.safe_dump(cfg))
    adapter = _FakeWhatsAppAdapter({"allow_from": []})
    runner = _Runner({"waha": adapter})
    reply = await runner._handle_access_command(_Event("allow user 089682642242"))
    assert "✅" in reply
    fresh = _config_yaml(homes)
    assert "6289682642242@s.whatsapp.net" in fresh["waha"]["allow_from"]
    assert "6281111111111@s.whatsapp.net" in fresh["waha"]["allow_from"]
    assert "allow_from" not in fresh["platforms"]["waha"]


@pytest.mark.asyncio
async def test_access_list_shows_source_and_truncates(homes):
    adapter = _FakeWhatsAppAdapter({
        "allow_from": [f"62810000000{i:02d}@s.whatsapp.net" for i in range(35)],
    })
    adapter._dm_allowlist_source = "config"
    runner = _Runner({"waha": adapter})
    reply = await runner._handle_access_command(_Event("list"))
    assert "(source: config)" in reply
    assert "…and 5 more" in reply
    assert "6281000000034@s.whatsapp.net" not in reply


@pytest.mark.asyncio
async def test_access_writes_audit_trail(homes):
    import json as _json

    adapter = _FakeWhatsAppAdapter({"allow_from": []})
    runner = _Runner({"waha": adapter})
    reply = await runner._handle_access_command(_Event("allow user 089682642242"))
    assert "✅" in reply
    lines = (homes / "access_audit.jsonl").read_text(encoding="utf-8").strip().split("\n")
    entry = _json.loads(lines[-1])
    assert entry["op"] == "allow"
    assert entry["canonical"] == "6289682642242@s.whatsapp.net"
    assert entry["scope"] == "user"
    assert entry["actor"] == "6285157813352@s.whatsapp.net"


def test_pushname_learned_lookup_roundtrip():
    # Transports without roster names (Baileys bridge, NOWEB) resolve @Name
    # through pushnames learned from inbound traffic — pin that fallback.
    holder = _FakeWhatsAppAdapter({})
    holder.remember_pushname("6289000000001@s.whatsapp.net", "Budi")
    assert holder._access_pushname_lookup("budi") == "6289000000001@s.whatsapp.net"
    assert holder._access_pushname_lookup("unknown") is None


@pytest.mark.asyncio
async def test_access_refuses_without_admin_list(homes):
    # Fail closed: no admin list means anyone could reach the handler, so the
    # handler itself must refuse — for edits AND for list (membership leaks).
    adapter = _FakeWhatsAppAdapter({"allow_from": []})
    runner = _Runner({"waha": adapter}, admins=False)
    reply = await runner._handle_access_command(_Event("allow user 089682642242"))
    assert "no admin list" in reply
    assert adapter._allow_from == set()
    assert "6289682642242@s.whatsapp.net" not in _config_yaml(homes)["platforms"]["waha"]["allow_from"]
    assert "no admin list" in await runner._handle_access_command(_Event("list"))


@pytest.mark.asyncio
async def test_access_works_with_admin_list(homes):
    # The default test runner carries admin lists, like a configured gateway.
    adapter = _FakeWhatsAppAdapter({"allow_from": []})
    runner = _Runner({"waha": adapter})
    reply = await runner._handle_access_command(_Event("allow user 089682642242"))
    assert "✅" in reply
