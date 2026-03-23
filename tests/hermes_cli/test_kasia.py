import os
import json
from types import SimpleNamespace

import hermes_cli.kasia as kasia_mod


def _test_io(env_values, saved_env, *, prompt_values, yes_no_answers):
    infos = []
    successes = []
    warnings = []
    errors = []

    def get_env_value(name):
        return env_values.get(name)

    def save_env_value(name, value):
        saved_env[name] = value
        env_values[name] = value

    def prompt(question, default=None, password=False):
        return prompt_values.get(question, default or "")

    def prompt_yes_no(question, default=False):
        return yes_no_answers.get(question, default)

    io = kasia_mod.KasiaCLIIO(
        get_env_value=get_env_value,
        save_env_value=save_env_value,
        prompt=prompt,
        prompt_yes_no=prompt_yes_no,
        print_info=infos.append,
        print_success=successes.append,
        print_warning=warnings.append,
        print_error=errors.append,
    )
    return io, infos, successes, warnings, errors


def test_is_kasia_configured_recognizes_plural_endpoint_env_vars(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("KASIA_ENABLED", raising=False)
    monkeypatch.delenv("KASIA_SEED_PHRASE", raising=False)
    monkeypatch.delenv("KASIA_INDEXER_URL", raising=False)
    monkeypatch.delenv("KASIA_NODE_WBORSH_URL", raising=False)
    monkeypatch.setenv(
        "KASIA_INDEXER_URLS",
        "https://indexer-a.example.com,https://indexer-b.example.com",
    )
    monkeypatch.setenv(
        "KASIA_NODE_WBORSH_URLS",
        "ws://node-a.example.com,ws://node-b.example.com",
    )

    assert kasia_mod.is_kasia_configured(os.getenv) is True


def test_prompt_kasia_seed_phrase_shows_hidden_note_and_retries_invalid(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    entered = iter(
        [
            "bad words only",
            "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu",
        ]
    )
    infos = []
    errors = []

    result = kasia_mod.prompt_kasia_seed_phrase(
        get_env_value=lambda _name: None,
        prompt=lambda question, default=None, password=False: next(entered),
        print_info=infos.append,
        print_error=errors.append,
        validate_seed_phrase=lambda value: (
            (False, "invalid mnemonic")
            if value == "bad words only"
            else (True, None)
        ),
    )

    assert result == "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu"
    assert any("hidden as you type" in line for line in infos)
    assert errors == ["invalid mnemonic"]


def test_complete_kasia_contact_approval_prefers_handshake_response(monkeypatch):
    monkeypatch.setenv("KASIA_BRIDGE_PORT", "3099")
    monkeypatch.setattr(
        kasia_mod,
        "fetch_kasia_bridge_health",
        lambda port: {"status": "connected", "bridge_port": port},
    )

    requests = []

    def fake_request(path, **kwargs):
        requests.append((path, kwargs.get("payload")))
        return {"status": "sent", "chatId": "kaspa:qpeeraddress"}

    monkeypatch.setattr(kasia_mod, "_request_kasia_bridge_json", fake_request)

    result = kasia_mod.complete_kasia_contact_approval(
        "kaspa:qpeeraddress",
        display_name="peer.kas",
    )

    assert result["status"] == "responded"
    assert requests == [
        ("/handshakes/respond", {"chatId": "kaspa:qpeeraddress"})
    ]


def test_complete_kasia_contact_approval_falls_back_to_initiate(monkeypatch):
    monkeypatch.setenv("KASIA_BRIDGE_PORT", "3099")
    monkeypatch.setattr(
        kasia_mod,
        "fetch_kasia_bridge_health",
        lambda port: {"status": "connected", "bridge_port": port},
    )

    requests = []

    def fake_request(path, **kwargs):
        requests.append((path, kwargs.get("payload")))
        if path == "/handshakes/respond":
            raise RuntimeError("No Kasia conversation found")
        return {"status": "sent", "chatId": "kaspa:qpeeraddress"}

    monkeypatch.setattr(kasia_mod, "_request_kasia_bridge_json", fake_request)

    result = kasia_mod.complete_kasia_contact_approval(
        "kaspa:qpeeraddress",
        display_name="peer.kas",
    )

    assert result["status"] == "initiated"
    assert requests == [
        ("/handshakes/respond", {"chatId": "kaspa:qpeeraddress"}),
        (
            "/handshakes/initiate",
            {
                "chatId": "kaspa:qpeeraddress",
                "displayName": "peer.kas",
                "retry": False,
            },
        ),
    ]


def test_validate_kasia_seed_phrase_rejects_non_12_or_24_word_lengths(monkeypatch):
    calls = []

    monkeypatch.setattr(
        kasia_mod.subprocess,
        "run",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    is_valid, error = kasia_mod.validate_kasia_seed_phrase(
        "one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen"
    )

    assert is_valid is False
    assert error == "Kasia seed phrase should contain 12 or 24 words."
    assert calls == []


def test_run_kasia_setup_installs_bridge_dependencies_before_prompting(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(kasia_mod, "PROJECT_ROOT", tmp_path)

    bridge_dir = tmp_path / "scripts" / "kasia-bridge"
    bridge_dir.mkdir(parents=True)
    (bridge_dir / "bridge.js").write_text("// test bridge\n")

    calls = []

    def fake_run(*args, **kwargs):
        calls.append((args, kwargs))
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(kasia_mod.subprocess, "run", fake_run)

    env_values = {}
    saved_env = {}
    prompt_values = {
        "Kasia indexer URL": "https://indexer.kasia.fyi",
        "Kaspa node URL": "wss://wrpc.kasia.fyi",
        "Kaspa network": "mainnet",
        "Allowed Kasia addresses (comma-separated, leave empty to set later)": "kaspa:qpeeraddress",
        "Kasia home channel address (leave empty to set later)": "kaspa:qhomeaddress",
    }
    yes_no_answers = {
        "Allow all Kasia users to message Hermes?": False,
    }

    io, _infos, successes, _warnings, errors = _test_io(
        env_values,
        saved_env,
        prompt_values=prompt_values,
        yes_no_answers=yes_no_answers,
    )

    configured = kasia_mod.run_kasia_setup(
        io,
        prompt_seed_phrase=lambda: "seed words go here",
    )

    assert configured is True
    assert calls == [
        (
            (["npm", "install", "--silent"],),
            {
                "cwd": str(bridge_dir),
                "capture_output": True,
                "text": True,
                "timeout": 120,
            },
        )
    ]
    assert any("Kasia bridge dependencies installed" in line for line in successes)
    assert errors == []


def test_run_kasia_setup_saves_expected_values(monkeypatch):
    monkeypatch.setattr(kasia_mod, "_ensure_kasia_bridge_dependencies", lambda _io: True)
    env_values = {}
    saved_env = {}
    prompt_values = {
        "Kasia indexer URL": "https://indexer.kasia.fyi",
        "Kaspa node URL": "wss://wrpc.kasia.fyi",
        "Kaspa network": "mainnet",
        "Allowed Kasia addresses (comma-separated, leave empty to set later)": "kaspa:qpeeraddress",
        "Kasia home channel address (leave empty to set later)": "kaspa:qhomeaddress",
    }
    yes_no_answers = {
        "Allow all Kasia users to message Hermes?": False,
    }

    io, infos, successes, warnings, errors = _test_io(
        env_values,
        saved_env,
        prompt_values=prompt_values,
        yes_no_answers=yes_no_answers,
    )

    configured = kasia_mod.run_kasia_setup(
        io,
        prompt_seed_phrase=lambda: "seed words go here",
    )

    assert configured is True
    assert saved_env["KASIA_ENABLED"] == "true"
    assert saved_env["KASIA_SEED_PHRASE"] == "seed words go here"
    assert saved_env["KASIA_INDEXER_URL"] == "https://indexer.kasia.fyi"
    assert saved_env["KASIA_NODE_WBORSH_URL"] == "wss://wrpc.kasia.fyi"
    assert saved_env["KASIA_NETWORK"] == "mainnet"
    assert saved_env["KASIA_FEE_POLICY"] == "auto"
    assert saved_env["KASIA_ALLOW_ALL_USERS"] == "false"
    assert saved_env["KASIA_ALLOWED_USERS"] == "kaspa:qpeeraddress"
    assert saved_env["KASIA_HOME_CHANNEL"] == "kaspa:qhomeaddress"
    assert "KASIA_KNS_URL" not in saved_env
    assert any("Kasia configured!" in line for line in successes)
    assert errors == []
    assert warnings == []
    assert any("dedicated Kaspa wallet" in line for line in infos)
    assert any("recommended default shown in brackets" in line for line in infos)
    assert any("Kaspa network saved: mainnet" in line for line in successes)
    assert any("Kasia fee policy saved: auto" in line for line in successes)


def test_run_kasia_setup_defaults_allow_all_to_false(monkeypatch):
    monkeypatch.setattr(kasia_mod, "_ensure_kasia_bridge_dependencies", lambda _io: True)
    env_values = {}
    saved_env = {}
    prompt_values = {
        "Kasia indexer URL": "https://indexer.kasia.fyi",
        "Kaspa node URL": "wss://wrpc.kasia.fyi",
        "Kaspa network": "mainnet",
        "Allowed Kasia addresses (comma-separated, leave empty to set later)": "kaspa:qpeeraddress",
        "Kasia home channel address (leave empty to set later)": "kaspa:qhomeaddress",
    }

    io, _infos, _successes, _warnings, _errors = _test_io(
        env_values,
        saved_env,
        prompt_values=prompt_values,
        yes_no_answers={},
    )

    configured = kasia_mod.run_kasia_setup(
        io,
        prompt_seed_phrase=lambda: "seed words go here",
    )

    assert configured is True
    assert saved_env["KASIA_ALLOW_ALL_USERS"] == "false"


def test_run_kasia_setup_resets_existing_priority_fee_policy_to_auto(monkeypatch):
    monkeypatch.setattr(kasia_mod, "_ensure_kasia_bridge_dependencies", lambda _io: True)
    env_values = {
        "KASIA_FEE_POLICY": "priority",
    }
    saved_env = {}
    prompt_values = {
        "Kasia indexer URL": "https://indexer.kasia.fyi",
        "Kaspa node URL": "wss://wrpc.kasia.fyi",
        "Kaspa network": "mainnet",
        "Allowed Kasia addresses (comma-separated, leave empty to set later)": "kaspa:qpeeraddress",
        "Kasia home channel address (leave empty to set later)": "kaspa:qhomeaddress",
    }
    yes_no_answers = {
        "Allow all Kasia users to message Hermes?": False,
    }

    io, _infos, successes, _warnings, _errors = _test_io(
        env_values,
        saved_env,
        prompt_values=prompt_values,
        yes_no_answers=yes_no_answers,
    )

    configured = kasia_mod.run_kasia_setup(
        io,
        prompt_seed_phrase=lambda: "seed words go here",
    )

    assert configured is True
    assert saved_env["KASIA_FEE_POLICY"] == "auto"
    assert any("Kasia fee policy saved: auto" in line for line in successes)


def test_run_kasia_doctor_includes_live_health(monkeypatch, capsys, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("KASIA_ENABLED", "true")
    monkeypatch.setenv("KASIA_SEED_PHRASE", "seed words go here")
    monkeypatch.setenv("KASIA_INDEXER_URL", "https://indexer.example.com")
    monkeypatch.setenv(
        "KASIA_INDEXER_URLS",
        "https://indexer.example.com,https://indexer-backup.example.com",
    )
    monkeypatch.setenv("KASIA_NODE_WBORSH_URL", "wss://node.example.com")
    monkeypatch.setenv(
        "KASIA_NODE_WBORSH_URLS",
        "wss://node.example.com,wss://node-backup.example.com",
    )
    monkeypatch.setenv("KASIA_KNS_URL", "https://kns.example.com/api/v1")
    monkeypatch.setenv("KASIA_ALLOWED_BROADCAST_CHANNELS", "alerts,ops")
    monkeypatch.setenv("KASIA_HOME_CHANNEL", "kaspa:qhome")
    monkeypatch.setattr(kasia_mod, "PROJECT_ROOT", tmp_path)

    bridge_dir = tmp_path / "scripts" / "kasia-bridge"
    bridge_dir.mkdir(parents=True)
    (bridge_dir / "bridge.js").write_text("// test bridge\n")
    (bridge_dir / "node_modules").mkdir()

    monkeypatch.setattr(
        kasia_mod,
        "fetch_kasia_bridge_health",
        lambda _port: {
            "indexerPool": {"activeUrl": "https://indexer-backup.example.com", "degraded": True},
            "nodePool": {"activeUrl": "wss://node-backup.example.com"},
            "walletFundingState": "ready",
            "walletBalanceSompi": "500000000",
            "availableMatureBalanceSompi": "500000000",
            "recommendedMinBalanceSompi": "40000000",
        },
    )
    monkeypatch.setattr(
        kasia_mod.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="v22.1.0", stderr=""),
    )

    assert kasia_mod.run_kasia_doctor() is True

    output = capsys.readouterr().out
    assert "Kasia Doctor" in output
    assert "KNS:        https://kns.example.com/api/v1 (override)" in output
    assert "Indexers:   2 configured" in output
    assert "Nodes:      2 configured" in output
    assert "Broadcasts: publish allowlist for #alerts, #ops" in output
    assert "✓ Wallet funding" in output
    assert "ready (5.00000000 KAS on-chain, 5.00000000 KAS spendable, recommended >= 0.40000000 KAS)" in output
    assert "Active indexer: https://indexer-backup.example.com" in output
    assert "Active node:    wss://node-backup.example.com" in output
    assert "Indexer pool is degraded / failover active" in output


def test_run_kasia_doctor_shows_network_default_kns_when_unset(
    monkeypatch, capsys, tmp_path
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("KASIA_ENABLED", "true")
    monkeypatch.setenv("KASIA_SEED_PHRASE", "seed words go here")
    monkeypatch.setenv("KASIA_INDEXER_URL", "https://indexer.example.com")
    monkeypatch.setenv("KASIA_NODE_WBORSH_URL", "wss://node.example.com")
    monkeypatch.setenv("KASIA_NETWORK", "testnet-10")
    monkeypatch.setattr(kasia_mod, "PROJECT_ROOT", tmp_path)

    bridge_dir = tmp_path / "scripts" / "kasia-bridge"
    bridge_dir.mkdir(parents=True)
    (bridge_dir / "bridge.js").write_text("// test bridge\n")
    (bridge_dir / "node_modules").mkdir()

    monkeypatch.setattr(
        kasia_mod,
        "fetch_kasia_bridge_health",
        lambda _port: {
            "indexerPool": {"activeUrl": "https://indexer.example.com"},
            "nodePool": {"activeUrl": "wss://node.example.com"},
        },
    )
    monkeypatch.setattr(
        kasia_mod.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="v22.1.0", stderr=""),
    )

    assert kasia_mod.run_kasia_doctor() is True

    output = capsys.readouterr().out
    assert "KNS:        https://api.knsdomains.org/tn10/api/v1 (network default)" in output


def test_run_kasia_doctor_fails_when_bridge_is_unreachable(monkeypatch, capsys, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("KASIA_ENABLED", "true")
    monkeypatch.setenv("KASIA_SEED_PHRASE", "seed words go here")
    monkeypatch.setenv("KASIA_INDEXER_URL", "https://indexer.example.com")
    monkeypatch.setenv("KASIA_NODE_WBORSH_URL", "wss://node.example.com")
    monkeypatch.setattr(kasia_mod, "PROJECT_ROOT", tmp_path)

    bridge_dir = tmp_path / "scripts" / "kasia-bridge"
    bridge_dir.mkdir(parents=True)
    (bridge_dir / "bridge.js").write_text("// test bridge\n")
    (bridge_dir / "node_modules").mkdir()

    monkeypatch.setattr(kasia_mod, "fetch_kasia_bridge_health", lambda _port: None)
    monkeypatch.setattr(
        kasia_mod.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="v22.1.0", stderr=""),
    )

    assert kasia_mod.run_kasia_doctor() is False

    output = capsys.readouterr().out
    assert "Bridge health" in output
    assert "Kasia doctor found configuration or dependency issues." in output
    assert "Kasia configuration looks good." not in output


def test_run_kasia_doctor_fails_when_wallet_funding_is_low(monkeypatch, capsys, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("KASIA_ENABLED", "true")
    monkeypatch.setenv("KASIA_SEED_PHRASE", "seed words go here")
    monkeypatch.setenv("KASIA_INDEXER_URL", "https://indexer.example.com")
    monkeypatch.setenv("KASIA_NODE_WBORSH_URL", "wss://node.example.com")
    monkeypatch.setattr(kasia_mod, "PROJECT_ROOT", tmp_path)

    bridge_dir = tmp_path / "scripts" / "kasia-bridge"
    bridge_dir.mkdir(parents=True)
    (bridge_dir / "bridge.js").write_text("// test bridge\n")
    (bridge_dir / "node_modules").mkdir()

    monkeypatch.setattr(
        kasia_mod,
        "fetch_kasia_bridge_health",
        lambda _port: {
            "indexerPool": {"activeUrl": "https://indexer.example.com"},
            "nodePool": {"activeUrl": "wss://node.example.com"},
            "walletFundingState": "low",
            "walletBalanceSompi": "27881431",
            "availableMatureBalanceSompi": "27881431",
            "recommendedMinBalanceSompi": "40000000",
        },
    )
    monkeypatch.setattr(
        kasia_mod.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="v22.1.0", stderr=""),
    )

    assert kasia_mod.run_kasia_doctor() is False

    output = capsys.readouterr().out
    assert "✗ Wallet funding" in output
    assert "low (0.27881431 KAS on-chain, 0.27881431 KAS spendable, recommended >= 0.40000000 KAS)" in output
    assert "Kasia doctor found configuration or dependency issues." in output


def test_run_kasia_doctor_counts_paired_users_as_access(monkeypatch, capsys, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("KASIA_ENABLED", "true")
    monkeypatch.setenv("KASIA_SEED_PHRASE", "seed words go here")
    monkeypatch.setenv("KASIA_INDEXER_URL", "https://indexer.example.com")
    monkeypatch.setenv("KASIA_NODE_WBORSH_URL", "wss://node.example.com")
    monkeypatch.delenv("KASIA_ALLOWED_USERS", raising=False)
    monkeypatch.delenv("KASIA_ALLOW_ALL_USERS", raising=False)
    monkeypatch.setattr(kasia_mod, "PROJECT_ROOT", tmp_path)

    pairing_dir = tmp_path / "pairing"
    pairing_dir.mkdir(parents=True)
    (pairing_dir / "kasia-approved.json").write_text(
        json.dumps(
            {
                "kaspa:qpeeraddress": {
                    "user_name": "luke.kas",
                    "approved_at": 1774153759.7806861,
                }
            }
        ),
        encoding="utf-8",
    )

    bridge_dir = tmp_path / "scripts" / "kasia-bridge"
    bridge_dir.mkdir(parents=True)
    (bridge_dir / "bridge.js").write_text("// test bridge\n")
    (bridge_dir / "node_modules").mkdir()

    monkeypatch.setattr(
        kasia_mod,
        "fetch_kasia_bridge_health",
        lambda _port: {
            "indexerPool": {"activeUrl": "https://indexer.example.com"},
            "nodePool": {"activeUrl": "wss://node.example.com"},
        },
    )
    monkeypatch.setattr(
        kasia_mod.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="v22.1.0", stderr=""),
    )

    assert kasia_mod.run_kasia_doctor() is True

    output = capsys.readouterr().out
    assert "✓ Access" in output
    assert "1 paired user(s) approved" in output
    assert "no allowlist configured" not in output


def test_run_kasia_doctor_reports_no_access_when_no_allowlist_or_pairing(
    monkeypatch, capsys, tmp_path
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("KASIA_ENABLED", "true")
    monkeypatch.setenv("KASIA_SEED_PHRASE", "seed words go here")
    monkeypatch.setenv("KASIA_INDEXER_URL", "https://indexer.example.com")
    monkeypatch.setenv("KASIA_NODE_WBORSH_URL", "wss://node.example.com")
    monkeypatch.delenv("KASIA_ALLOWED_USERS", raising=False)
    monkeypatch.delenv("KASIA_ALLOW_ALL_USERS", raising=False)
    monkeypatch.setattr(kasia_mod, "PROJECT_ROOT", tmp_path)

    bridge_dir = tmp_path / "scripts" / "kasia-bridge"
    bridge_dir.mkdir(parents=True)
    (bridge_dir / "bridge.js").write_text("// test bridge\n")
    (bridge_dir / "node_modules").mkdir()

    monkeypatch.setattr(
        kasia_mod,
        "fetch_kasia_bridge_health",
        lambda _port: {
            "indexerPool": {"activeUrl": "https://indexer.example.com"},
            "nodePool": {"activeUrl": "wss://node.example.com"},
        },
    )
    monkeypatch.setattr(
        kasia_mod.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="v22.1.0", stderr=""),
    )

    assert kasia_mod.run_kasia_doctor() is True

    output = capsys.readouterr().out
    assert "✗ Access" in output
    assert "no allowlist or paired users configured" in output
