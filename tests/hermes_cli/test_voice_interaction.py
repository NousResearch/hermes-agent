from hermes_cli.voice_interaction import (
    choose_voice_acknowledgement,
    elevenlabs_pronunciation_locators,
    prepare_spoken_text,
    resolve_voice_tts_profile,
    strip_unsupported_expressive_tags,
)


def test_tts_normalization_plan_examples():
    cases = [
        (
            "BGP is down on 192.168.130.144/24 and TCP/179 is blocked.",
            "B G P is down on one ninety two dot one sixty eight dot one thirty dot one forty four slash twenty four and T C P port one seventy nine is blocked.",
        ),
        (
            "Restart kea-dhcp4 on dc01.space.bartos.casa.",
            "Restart kea D H C P four on D C zero one dot space dot bartos dot casa.",
        ),
        (
            "svc_ansible_baseline needs rights in OU=Linux,OU=Computers,OU=Lab,DC=space,DC=bartos,DC=casa.",
            "Service account ansible baseline needs rights in O U Linux, O U Computers, O U Lab, D C space, D C bartos, D C casa.",
        ),
        (
            "VLANs 10,20,140 are allowed on LACP trunk ae1.",
            "V lans ten, twenty, and one forty are allowed on L A C P trunk A E one.",
        ),
    ]
    for raw, expected in cases:
        assert prepare_spoken_text(raw) == expected


def test_spoken_final_strips_markdown_json_and_code():
    raw = """## Result\n\n```json\n{\"ok\": true, \"port\": 443}\n```\n\n| Interface | VLAN |\n| --- | --- |\n| eth0 | 140 |\n\nUse `systemctl restart kea-dhcp4` if needed."""
    spoken = prepare_spoken_text(raw)
    assert "```" not in spoken
    assert "|" not in spoken
    assert "system C T L restart kea D H C P four" in spoken


def test_expressive_tags_are_gated_and_limited():
    cfg = {"voice": {"expressive_tags_enabled": False, "expressive_tag_model_allowlist": ["eleven_v3"]}}
    assert strip_unsupported_expressive_tags("[sighs] nope [laughs]", config=cfg, model_id="eleven_v3") == " nope "

    cfg["voice"]["expressive_tags_enabled"] = True
    assert strip_unsupported_expressive_tags("[sighs] ok [laughs] done", config=cfg, model_id="eleven_v3") == "[sighs] ok  done"
    assert strip_unsupported_expressive_tags("[sighs] ok", config=cfg, model_id="eleven_multilingual_v2") == " ok"


def test_acknowledgement_heuristic_slow_and_approval():
    assert choose_voice_acknowledgement("what is two plus two") == ""
    assert choose_voice_acknowledgement("check the current weather and summarize it") == "Checking that now."
    assert choose_voice_acknowledgement("sudo restart the service") == "I need approval before I can run that."


def test_elevenlabs_pronunciation_locator_config_shapes():
    cfg = {
        "elevenlabs": {
            "pronunciation_dictionary_locators": [
                "dict_123",
                {"id": "dict_456", "version_id": "ver_1"},
            ]
        }
    }
    assert elevenlabs_pronunciation_locators(cfg) == [
        {"pronunciation_dictionary_id": "dict_123"},
        {"pronunciation_dictionary_id": "dict_456", "version_id": "ver_1"},
    ]


def test_tts_profile_controls_normalization_and_tag_policy():
    cfg = {
        "voice": {
            "tts_profile": "raw",
            "expressive_tags_enabled": True,
            "expressive_tag_model_allowlist": ["eleven_v3"],
            "tts_profiles": {
                "raw": {"normalize": False, "expressive_tags": False},
            },
        }
    }

    resolved = resolve_voice_tts_profile(cfg["voice"])
    assert resolved["active_tts_profile"] == "raw"
    assert resolved["normalize"] is False
    assert resolved["expressive_tags_enabled"] is False
    # The raw profile disables technical normalization and strips tags even
    # though the global voice config would otherwise allow them.
    assert prepare_spoken_text("[sighs] BGP 192.168.1.1", config=cfg, model_id="eleven_v3") == "BGP 192.168.1.1"


def test_streaming_elevenlabs_request_includes_pronunciation_locators(monkeypatch):
    import queue
    import threading

    from tools import tts_tool

    captured = {}

    class FakeTextToSpeech:
        def convert(self, **kwargs):
            captured.update(kwargs)
            return [b"\x00\x00"]

    class FakeClient:
        def __init__(self, api_key):
            self.api_key = api_key
            self.text_to_speech = FakeTextToSpeech()

    text_q = queue.Queue()
    text_q.put("Hello world.")
    text_q.put(None)
    stop_evt = threading.Event()
    done_evt = threading.Event()

    monkeypatch.setattr(tts_tool, "get_env_value", lambda name, default=None: "test-key" if name == "ELEVENLABS_API_KEY" else default)
    monkeypatch.setattr(tts_tool, "_import_elevenlabs", lambda: FakeClient)
    monkeypatch.setattr(tts_tool, "_import_sounddevice", lambda: (_ for _ in ()).throw(ImportError("no sounddevice")))
    monkeypatch.setattr(tts_tool, "_load_tts_config", lambda: {
        "elevenlabs": {
            "voice_id": "voice_1",
            "streaming_model_id": "eleven_flash_v2_5",
            "pronunciation_dictionary_locators": [
                {"id": "dict_123", "version_id": "ver_1"},
            ],
        }
    })
    monkeypatch.setattr(tts_tool, "_load_voice_config", lambda: {})
    monkeypatch.setattr("tools.voice_mode.play_audio_file", lambda path: None)

    tts_tool.stream_tts_to_speaker(text_q, stop_evt, done_evt)

    assert done_evt.is_set()
    assert captured["pronunciation_dictionary_locators"] == [
        {"pronunciation_dictionary_id": "dict_123", "version_id": "ver_1"}
    ]
