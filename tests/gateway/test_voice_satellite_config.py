"""voice_satellite config defaults + gateway config-chain tests."""

import textwrap

from tests.gateway._plugin_adapter_loader import load_plugin_adapter

_mod = load_plugin_adapter("voice_satellite")


def test_default_config_has_voice_satellite_section():
    from hermes_cli.config import DEFAULT_CONFIG

    section = DEFAULT_CONFIG["voice_satellite"]
    assert section["satellites"] == []  # off until a satellite is configured
    # every endpointing default maps onto an EndpointDetector kwarg (invariant)
    audio = _mod._import_sibling("audio")
    det = audio.EndpointDetector(**{
        k: v for k, v in section["endpointing"].items()
    })
    assert det.silence_threshold == section["endpointing"]["silence_threshold"]


def test_gateway_config_chain_enables_platform(tmp_path, monkeypatch):
    """A real config.yaml reaches PlatformConfig.extra + enabled via the
    actual gateway loader (no mocking of load_gateway_config internals).

    Resolved against the real loader (gateway/config.py):
    - ``load_gateway_config()`` takes no arguments; it always reads
      ``$HERMES_HOME/config.yaml`` (gateway/config.py:964-974).
    - The ``apply_yaml_config_fn`` hook (gateway/config.py ~1253-1290) is
      called as ``entry.apply_yaml_config_fn(yaml_cfg, platform_cfg)`` where
      ``platform_cfg`` is the *raw* ``yaml_cfg["voice_satellite"]`` dict, and
      only the function's *return value* is merged into
      ``PlatformConfig.extra`` (``extra.update(seeded)``). Any in-place
      mutation of ``platform_cfg`` (e.g. setting ``platform_cfg["enabled"]``)
      is discarded — it never reaches ``plat_data["enabled"]`` or
      ``PlatformConfig.enabled``. Verified empirically: a config.yaml with
      only a top-level ``voice_satellite:`` block (no ``platforms:`` map)
      loads with ``enabled=False`` even though the old hook code mutated
      ``platform_cfg["enabled"] = True``.
    - The loader's actual enablement path is the top-level ``platforms:``
      map (``_merge_platform_map``, gateway/config.py ~1085-1105), which
      copies ``enabled`` straight into ``platforms_data[name]`` by dict key
      (no ``Platform`` enum membership required at that point). So
      enablement requires a ``platforms: voice_satellite: enabled: true``
      entry alongside the ``voice_satellite:`` section.
    """
    from gateway import config as gw_config

    _mod.register(_RegistryCtx())  # ensure entry registered for this process

    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(
        textwrap.dedent(
            """
            voice_satellite:
              satellites:
                - name: kitchen
                  host: 192.168.1.40
                  port: 10700

            platforms:
              voice_satellite:
                enabled: true
            """
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    config = gw_config.load_gateway_config()
    pc = config.platforms.get(gw_config.Platform("voice_satellite"))
    assert pc is not None
    assert pc.enabled is True
    assert pc.extra["satellites"][0]["name"] == "kitchen"


class _RegistryCtx:
    def register_platform(self, **kwargs):
        from gateway.platform_registry import PlatformEntry, platform_registry

        allowed = {f.name for f in PlatformEntry.__dataclass_fields__.values()}
        entry = PlatformEntry(
            **{k: v for k, v in kwargs.items() if k in allowed}
        )
        platform_registry.register(entry)
