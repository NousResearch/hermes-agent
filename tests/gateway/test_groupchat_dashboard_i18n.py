"""Execute the dashboard bundle against the SDK contract, without a browser."""
import shutil
import subprocess
from pathlib import Path

import pytest


def test_log_directory_tracks_selected_profile(monkeypatch, tmp_path):
    pytest.importorskip("fastapi")
    from contextlib import contextmanager
    from plugins.groupchat.dashboard.plugin_api import get_settings
    import hermes_cli.web_server as server
    import hermes_cli.config as config
    import hermes_constants

    selected = []
    @contextmanager
    def profile_scope(profile):
        selected.append(profile)
        try:
            yield
        finally:
            selected.pop()

    monkeypatch.setattr(server, "_config_profile_scope", profile_scope)
    monkeypatch.setattr(config, "read_raw_config", lambda: {})
    monkeypatch.setattr(hermes_constants, "get_hermes_home", lambda: tmp_path / selected[-1])
    for profile in ("first", "second"):
        assert get_settings(profile)["decision_log_directory"] == str(tmp_path / profile / "logs")
        assert get_settings(profile)["persistent_context_directory"] == str(tmp_path / profile / "groupchat")
    assert not (tmp_path / "first").exists()


def test_matrix_participation_uses_effective_saved_settings(monkeypatch, tmp_path):
    pytest.importorskip("fastapi")
    from types import SimpleNamespace
    from plugins.groupchat.dashboard.plugin_api import groupchat_participation
    import hermes_cli.profiles as profiles

    active = tmp_path / "active"
    mention_only = tmp_path / "mention_only"
    disabled = tmp_path / "disabled"
    for home in (active, mention_only, disabled):
        home.mkdir()
    base = """platforms:\n  matrix:\n    enabled: true\ngroupchat:\n  enabled: true\n  platforms: [matrix]\n  relevance:\n    enabled: true\nplugins:\n  enabled: [groupchat]\n"""
    (active / "config.yaml").write_text(base)
    (active / ".env").write_text("MATRIX_REQUIRE_MENTION=false\nSECRET=not-returned\n")
    (mention_only / "config.yaml").write_text(base.replace("enabled: true\ngroupchat:", "enabled: true\n    require_mention: true\ngroupchat:"))
    (disabled / "config.yaml").write_text("platforms:\n  matrix:\n    enabled: true\n")
    infos = [SimpleNamespace(name=name, path=home, is_default=False, gateway_running=True)
             for name, home in (("active", active), ("mention_only", mention_only), ("disabled", disabled))]
    infos.extend([SimpleNamespace(name="stale", path=tmp_path, is_default=False, gateway_running=False),
                  SimpleNamespace(name="default", path=tmp_path, is_default=True, gateway_running=False)])
    monkeypatch.setattr(profiles, "list_profiles", lambda: infos)
    result = groupchat_participation()
    assert result == [
        {"profile": "active", "require_mention": False, "groupchat_enabled": True, "participates": True},
        {"profile": "mention_only", "require_mention": True, "groupchat_enabled": True, "participates": False},
        {"profile": "disabled", "require_mention": True, "groupchat_enabled": False, "participates": False},
    ]
    assert "SECRET" not in str(result) and "not-returned" not in str(result)


def test_dashboard_uses_host_locale_and_english_fallback():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is needed to execute the dashboard bundle")
    bundle = Path(__file__).resolve().parents[2] / "plugins/groupchat/dashboard/index.js"
    harness = r'''
const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");
const bundle = fs.readFileSync(process.argv[1], "utf8");
let locale = "en", component, cursor = 0;
const draft = {enabled:false, platforms:["matrix"],
 filter_model:{provider:"mistral",model:"test",fallbacks:[{provider:"openrouter",model:"fallback"}]},
 relevance:{enabled:true,score_delays:{1:120,2:30,3:10,4:3,5:0}, info_delay:300,max_context_messages:5},
 pingpong_guard:{enabled:true,min_chars:60}};
const states = ["current", [], draft, draft, false, "saved", "", null,
 "/test/profile/logs", ["matrix", "mattermost", "discord"],
 [{profile:"bastian_kern",require_mention:false,groupchat_enabled:true,participates:true},{profile:"lena_brandner",require_mention:true,groupchat_enabled:false,participates:false}],
 "/test/profile/groupchat"];
const sdk = {React:{createElement:(tag,props,...children)=>({tag,props,children}),Fragment:"fragment"},
 api:{getProfiles:()=>Promise.resolve({profiles:[]})},fetchJSON:()=>Promise.resolve({settings:draft,decision_log_directory:"/test/profile/logs",persistent_context_directory:"/test/profile/groupchat",groupchat_participation:states[10]}),
 hooks:{useState:()=>[states[cursor++],()=>{}],useEffect:fn=>{fn();}},
 useI18n:()=>({locale})};
const context = {window:{__HERMES_PLUGIN_SDK__:sdk,__HERMES_PLUGINS__:{register:(name,fn)=>{assert.equal(name,"groupchat");component=fn;}}}};
vm.runInNewContext(bundle,context);
function render() {cursor=0;return JSON.stringify(component());}
let text = render();
assert.match(text,/Enable Groupchat/);
assert.match(text,/Mattermost/);
assert.match(text,/every selected channel/);
assert.match(text,/require_mention=false/);
assert.match(text,/bastian_kern.*require_mention=.*false.*active/);
assert.match(text,/lena_brandner.*require_mention=.*true.*Groupchat off/);
assert.match(text,/Decision logs/);
assert.match(text,/Log directory/);
assert.equal((text.match(/\/test\/profile\/logs/g) || []).length, 1);
assert.match(text,/Persistent score-1 context directory/);
assert.equal((text.match(/\/test\/profile\/groupchat/g) || []).length, 1);
assert.match(text,/developed by RechnerLotsen/);
assert.doesNotMatch(text,/matrix-groupchat-outbound.jsonl/);
states[8]="/test/another-profile/logs";
assert.match(render(),/\/test\/another-profile\/logs/);
states[8]="/test/profile/logs";
assert.match(text,/German and English/);
assert.match(text,/Primary Model/);
assert.match(text,/Remove fallback 1/);
assert.match(text,/Saved\. Restart this profile/);
assert.doesNotMatch(text,/Groupchat aktivieren|Gespeichert/);
locale="de";text=render();
assert.match(text,/Groupchat aktivieren/);
assert.match(text,/Entscheidungslogs/);
assert.match(text,/Log-Verzeichnis/);
assert.match(text,/dauerhaften Stufe-1-Kontext/);
assert.match(text,/von RechnerLotsen entwickelt/);
assert.match(text,/Matrix-Teilnahme/);
assert.match(text,/bastian_kern.*require_mention=.*false.*aktiv/);
assert.match(text,/Primär Modell/);
assert.match(text,/Fallback 1 entfernen/);
assert.match(text,/Gespeichert/);
locale="de-AT";assert.match(render(),/Groupchat aktivieren/);
locale="fr";assert.match(render(),/Enable Groupchat/);
locale=undefined;assert.match(render(),/Enable Groupchat/);
delete sdk.useI18n;
vm.runInNewContext(bundle,context);
assert.match(render(),/Enable Groupchat/);
'''
    completed = subprocess.run([node, "-e", harness, str(bundle)], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
