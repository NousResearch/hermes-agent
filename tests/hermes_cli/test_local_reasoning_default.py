"""Managed reasoning defaults belong to the model preset, below explicit client choices."""

import json
import struct

from hermes_cli.local_runtime import bootstrap, catalog, presets
from hermes_cli.local_runtime.estimator import HardwareBudget


def _stage_header(path):
    """A metadata-only GGUF exercises the real reader without model weights."""
    metadata = {
        "general.architecture": "llama",
        "llama.context_length": 65536,
        "general.sampling.temperature": 0.42,
    }

    def string(value):
        raw = value.encode()
        return struct.pack("<Q", len(raw)) + raw

    data = b"GGUF" + struct.pack("<IQQ", 3, 0, len(metadata))
    for key, value in metadata.items():
        kind, raw = ((8, string(value)) if isinstance(value, str) else
                     (4, struct.pack("<I", value)) if isinstance(value, int) else
                     (6, struct.pack("<f", value)))
        data += string(key) + struct.pack("<I", kind) + raw
    path.write_bytes(data)


def test_qwen_medium_default_reaches_only_its_model_preset(tmp_path, monkeypatch):
    # Both profile config and machine-scoped model/runtime assets are disposable.
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr("hermes_constants.get_default_hermes_root", lambda: tmp_path)
    config_path = tmp_path / "config.yaml"
    config = b"agent:\n  reasoning_effort: off\n"
    config_path.write_bytes(config)
    entry = catalog.catalog_by_id()["qwen3.8-27b"]
    model_id = entry.variants[0].model_id
    control_id = catalog.catalog_by_id()["qwen3.8-flash-next"].variants[0].model_id
    model_dir = bootstrap.models_dir()
    model_dir.mkdir()
    for name in (model_id, control_id, "off-catalog-model"):
        _stage_header(model_dir / f"{name}.gguf")
    projector = bootstrap.assets_dir() / entry.mmproj.local_name
    projector.parent.mkdir()
    projector.touch()

    ini = tmp_path / "presets.ini"
    generated = presets.generate_presets(
        model_dir, HardwareBudget(64 << 30, 64 << 30, 64 << 30), ini)
    reread = presets.read_preset_decisions(ini)
    assert set(reread) == {p.model_id for p in generated}
    target = reread[model_id].keys
    assert json.loads(target.get("chat-template-kwargs", "{}")) == {"reasoning_effort": "medium"}
    for name in (control_id, "off-catalog-model"):
        assert "reasoning_effort" not in json.loads(reread[name].keys.get("chat-template-kwargs", "{}"))
    # The default must not remove vision/MTP or displace the GGUF's sampling.
    assert target["mmproj"] == str(projector)
    assert target["spec-draft-n-max"] == str(entry.mtp_draft_depth)
    assert target["temp"] == "0.42"
    assert config_path.read_bytes() == config


def test_explicit_reasoning_choices_survive_shared_resolution_and_wire(tmp_path, monkeypatch):
    from agent.transports.chat_completions import ChatCompletionsTransport
    from hermes_cli.config_effective import load_user_config_effective
    from hermes_constants import resolve_reasoning_config
    from providers import get_provider_profile

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr("hermes_constants.get_default_hermes_root", lambda: tmp_path)
    model_id = catalog.catalog_by_id()["qwen3.8-27b"].variants[0].model_id
    config_path = tmp_path / "config.yaml"
    transport = ChatCompletionsTransport()
    profile = get_provider_profile("llamacpp")
    assert profile is not None
    # The Off control uses "none"; YAML's unquoted off/false becomes False.
    for effort in (None, "none", False, "low", "medium", "high", "xhigh"):
        for source in ("global", "per-model"):
            agent_cfg = {}
            if effort is not None:
                agent_cfg = ({"reasoning_effort": effort} if source == "global" else {
                    "reasoning_effort": "high" if effort != "high" else "low",
                    "reasoning_overrides": {model_id: effort},
                })
            config = json.dumps({"agent": agent_cfg})
            config_path.write_text(config, encoding="utf-8")
            resolved = resolve_reasoning_config(load_user_config_effective(), model_id)
            wire = transport.build_kwargs(
                model=model_id, messages=[{"role": "user", "content": "Hello"}],
                provider_profile=profile, base_url="http://127.0.0.1:1/v1",
                reasoning_config=resolved)
            if effort is None:
                assert "reasoning_effort" not in wire  # Leave the default to this model's server preset.
            else:
                assert wire["reasoning_effort"] == ("none" if effort is False else effort)
            assert "chat_template_kwargs" not in wire.get("extra_body", {})
            assert config_path.read_text(encoding="utf-8") == config
