from hermes_cli.voice_interaction import (
    choose_voice_acknowledgement,
    elevenlabs_pronunciation_locators,
    prepare_spoken_text,
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
