"""Property: a saved platform block reads the same outside the gateway as the gateway loads it.

Readers of a saved platform block outside the gateway (the left-core migrator's ``_merged_platform_block``,
cron's ``_unloaded_home_platforms``) read ``gateway.config_loader.platform_blocks``, the one normalized view.
The corpus draws config.yaml platform sections with the platform's name in mixed case and under its bundled
manifest aliases, duplicate lower/upper keys across ``platforms``, ``gateway.platforms``, ``gateway.<name>``
and a top-level ``<name>:`` block, ``enabled`` on or off, malformed ``extra`` values (the gateway then drops
the whole config.yaml layer) and a global ``unauthorized_dm_behavior`` a platform's own value falls back to,
and asserts that the view's home channel and shared-key ``extra`` equal ``load_gateway_config()``'s.
A counterexample a reviewer finds joins ``known`` / ``_block``.
"""

from __future__ import annotations

import random

import pytest

_CASES = 80
_PLATFORMS = ("telegram", "irc")
_DM = ("pair", "ignore", "bogus")


def _aliases(name: str) -> list[str]:
    from gateway.config import Platform
    return [alias for alias, target in Platform._scan_bundled_plugin_platforms()[1].items() if target == name]


def _variants(name: str) -> list[str]:
    return [name, name.title(), name.upper(), *_aliases(name), *(a.upper() for a in _aliases(name))]


def _block(rng: random.Random, name: str, i: int) -> dict:
    block: dict = {}
    if rng.random() < 0.6:
        block["home_channel"] = {"platform": name, "chat_id": f"chat-{i}-{rng.randint(0, 99)}", "name": "Home",
                                 **({"thread_id": str(rng.randint(1, 9))} if rng.random() < 0.4 else {})}
    if rng.random() < 0.5:
        block["enabled"] = rng.random() < 0.5
    if rng.random() < 0.3:
        block["extra"] = rng.choice([{}, {"reply_prefix": "x"}, None, "oops"])
    if rng.random() < 0.3:
        block["unauthorized_dm_behavior"] = rng.choice(_DM)
    return block


def _config(rng: random.Random, name: str, i: int) -> dict:
    config: dict = {}
    if rng.random() < 0.4:
        config["unauthorized_dm_behavior"] = rng.choice(_DM)
    if rng.random() < 0.2:
        config.setdefault("gateway", {})["unauthorized_dm_behavior"] = rng.choice(_DM)
    for _ in range(rng.randint(1, 4)):
        where = rng.choice(["platforms", "gateway.platforms", "gateway.<name>", "root"])
        key = rng.choice(_variants(name))
        if where == "platforms":
            config.setdefault("platforms", {})[key] = _block(rng, name, i)
        elif where == "gateway.platforms":
            config.setdefault("gateway", {}).setdefault("platforms", {})[key] = _block(rng, name, i)
        elif where == "gateway.<name>":
            config.setdefault("gateway", {})[key] = _block(rng, name, i)
        else:
            config[rng.choice([name, name, name.title()])] = _block(rng, name, i)
    return config


def _seen(home, extra: dict) -> tuple:
    return ((str(home.chat_id), home.thread_id or None) if home is not None and home.chat_id else None,
            extra.get("unauthorized_dm_behavior"))


def _gateway_view(name: str):
    from gateway.config import Platform, load_gateway_config
    platform = load_gateway_config().platforms.get(Platform(name))
    return None if platform is None else _seen(platform.home_channel, platform.extra)


def _saved_view(home_dir, name: str):
    from gateway.config import PlatformConfig
    from hermes_cli.left_core_migration import _merged_platform_block
    block = _merged_platform_block(home_dir, name)
    if block is None:
        return None
    config = PlatformConfig.from_dict(block)
    return _seen(config.home_channel, config.extra)


@pytest.mark.parametrize("name", _PLATFORMS)
def test_the_saved_block_view_reads_what_the_gateway_loads(tmp_path, monkeypatch, name):
    import os
    import hermes_yaml as yaml
    for env in [k for k in os.environ if k.startswith(name.upper())]:
        monkeypatch.delenv(env)
    rng = random.Random(f"platform-blocks-{name}")
    failures = []
    home_block = {"home_channel": {"platform": name, "chat_id": "old", "name": "Home"}}
    known = [  # review counterexamples
        # a mixed-case key's old home beside a lower-case disable
        {"platforms": {name.title(): dict(home_block)}, name: {"enabled": False}},
        {"platforms": {name.upper(): dict(home_block), name: {"enabled": True}}},
        # a malformed block later in config.yaml: the gateway keeps none of the file, not the half merged before it
        {"platforms": {name: dict(home_block), "discord": {"extra": None}}},
        # an invalid per-platform DM choice falls back to the global one the gateway bridged first
        {"unauthorized_dm_behavior": "ignore", "platforms": {name: {**home_block, "unauthorized_dm_behavior": "bogus"}}},
        {"gateway": {"unauthorized_dm_behavior": "ignore"}, name: {"unauthorized_dm_behavior": "bogus"}},
    ]
    for i in range(_CASES):
        home = tmp_path / f"case{i}"
        home.mkdir()
        config = known[i] if i < len(known) else _config(rng, name, i)
        (home / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
        monkeypatch.setenv("HERMES_HOME", str(home))
        gateway, saved = _gateway_view(name), _saved_view(home, name)
        # No block and an empty block read alike; anything saved must agree.
        if gateway != saved and {gateway, saved} - {None, (None, None)}:
            failures.append((i, config, gateway, saved))
    assert not failures, failures[:2]
