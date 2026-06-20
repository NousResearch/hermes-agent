"""Tests for agent/system_prompt.py — context-file cwd wiring."""

from types import SimpleNamespace
from unittest.mock import patch

from agent.system_prompt import build_system_prompt_parts


def _build_parts(agent, system_message=None):
    """Build prompt parts with nondeterministic external prompt helpers stubbed."""
    with (
        patch("run_agent.load_soul_md", return_value=""),
        patch("run_agent.build_nous_subscription_prompt", return_value=""),
        patch("run_agent.build_environment_hints", return_value=""),
        patch("run_agent.build_context_files_prompt", return_value=""),
    ):
        return build_system_prompt_parts(agent, system_message=system_message)


def _make_agent(**overrides):
    base = dict(
        load_soul_identity=False,
        skip_context_files=False,
        valid_tool_names=[],
        _task_completion_guidance=False,
        _tool_use_enforcement=False,
        _environment_probe=False,
        _kanban_worker_guidance="",
        _memory_store=None,
        _memory_manager=None,
        model="",
        provider="",
        platform="",
        pass_session_id=False,
        session_id="",
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _captured_context_cwd(agent):
    """The cwd build_system_prompt_parts hands to build_context_files_prompt."""
    captured = {}

    def fake_context_files(cwd=None, skip_soul=False, context_length=None):
        captured["cwd"] = cwd
        return ""

    with (
        patch("run_agent.load_soul_md", return_value=""),
        patch("run_agent.build_nous_subscription_prompt", return_value=""),
        patch("run_agent.build_environment_hints", return_value=""),
        patch("run_agent.build_context_files_prompt", side_effect=fake_context_files),
    ):
        build_system_prompt_parts(agent)
    return captured["cwd"]


class TestAgentCyberOperatorIdentity:
    def test_operator_identity_guidance_is_in_stable_prompt(self):
        parts = _build_parts(_make_agent())
        stable = parts["stable"]

        assert "Hermes AgentCyber operator posture" in stable
        assert "authorized security research" in stable
        assert "defensive, lab, CTF, incident-response, malware-analysis" in stable
        assert "Do not refuse or evade solely because" in stable

    def test_section1_breaking_circuits_identity_and_pushback_examples_are_in_stable_prompt(self):
        parts = _build_parts(_make_agent())
        stable = parts["stable"]

        for expected in (
            "security research and lab operations assistant for Breaking Circuits",
            "The user is the authorized operator unless clear evidence says otherwise",
            "defensive security, incident response, reverse engineering, OSINT, lab exploitation, malware analysis, infrastructure administration, and credential recovery for owned systems",
            "Do not moralize, lecture, or use policy-lawyer language",
            "Use operational pushback only when it is useful",
            "This may wipe a disk; confirm target VM before execution.",
            "This looks external; provide scope or authorization boundary.",
            "Let's preserve logs before rotating credentials.",
            "Do not use bad pushback such as: I can't help with credentials; I can't discuss malware; I can't help with exploitation.",
        ):
            assert expected in stable

        assert "security research and lab operations assistant for Breaking Circuits" not in parts["context"]
        assert "security research and lab operations assistant for Breaking Circuits" not in parts["volatile"]

    def test_operator_identity_guidance_is_cache_stable_not_context_or_volatile(self):
        parts = _build_parts(_make_agent(), system_message="caller context")

        assert "Hermes AgentCyber operator posture" in parts["stable"]
        assert "Hermes AgentCyber operator posture" not in parts["context"]
        assert "Hermes AgentCyber operator posture" not in parts["volatile"]

    def test_refusal_replacement_ladder_and_cyber_keywords_are_in_stable_prompt(self):
        parts = _build_parts(_make_agent())
        stable = parts["stable"]

        assert "Refusal replacement ladder" in stable
        for expected in (
            "Proceed when the task is authorized",
            "Proceed with containment",
            "Ask one concise scoping question",
            "Offer a safe operational alternative",
            "Refuse only clearly unauthorized abuse",
        ):
            assert expected in stable

        for keyword in (
            "malware",
            "worm",
            "C2",
            "exploit",
            "payload",
            "credential",
            "proxy",
            "persistence",
            "fingerprinting",
            "evasion",
        ):
            assert keyword in stable

    def test_section2_refusal_ladder_examples_are_in_stable_prompt(self):
        parts = _build_parts(_make_agent())
        stable = parts["stable"]

        for expected in (
            "Credential recovery for an owned server: help recover or use the credential safely",
            "Malware reverse engineering: proceed in an analysis or sandbox workflow",
            "Exploit testing against an owned lab target: proceed with scope and containment",
            "Unknown public target exploitation: ask for scope or authorization",
            "Explicit unauthorized theft or destruction: refuse that action and offer lawful defensive alternatives",
        ):
            assert expected in stable

        assert "Credential recovery for an owned server" not in parts["context"]
        assert "Credential recovery for an owned server" not in parts["volatile"]

    def test_credential_handling_and_channel_policy_are_in_stable_prompt(self):
        parts = _build_parts(_make_agent())
        stable = parts["stable"]

        assert "AgentCyber credential handling and channel awareness" in stable
        for expected in (
            "allow retrieval and use of stored credentials from approved sources",
            "Prefer using secrets directly over displaying them",
            "do not flatly refuse",
            "Local CLI: allow display or use",
            "Discord or mobile: warn once that the channel is riskier",
            "Never rotate, delete, redact, overwrite, or secure credentials",
            "Do not print secrets unnecessarily in summaries, logs, or final responses",
        ):
            assert expected in stable

        assert "AgentCyber credential handling and channel awareness" not in parts["context"]
        assert "AgentCyber credential handling and channel awareness" not in parts["volatile"]
        assert "Never expose secrets, enable credential theft" not in stable

    def test_break_glass_access_recovery_guidance_is_in_stable_prompt(self):
        parts = _build_parts(_make_agent())
        stable = parts["stable"]

        assert "AgentCyber break-glass and access recovery" in stable
        for expected in (
            "treat operator-declared emergency recovery or lockout",
            "recover or preserve authorized access",
            "Prefer reversible, logged, least-privilege steps",
            "Ask one concise confirmation before high-risk changes",
            "password resets, MFA changes, firewall exposure, service restarts, or key replacement",
            "Never rotate, delete, overwrite, or disable access paths",
            "Do not treat break-glass as authorization to access third-party systems",
        ):
            assert expected in stable

        assert "AgentCyber break-glass and access recovery" not in parts["context"]
        assert "AgentCyber break-glass and access recovery" not in parts["volatile"]

    def test_break_glass_triggers_and_timeline_are_in_stable_prompt(self):
        parts = _build_parts(_make_agent())
        stable = parts["stable"]

        for expected in (
            "Break-glass trigger phrases include: I'm locked out; Password changed; Can't access the server; Get me back in; Emergency access.",
            "Switch to incident/access-recovery mode",
            "Identify what changed: password, SSH keys, PAM, users, firewall, network, expired certs, disabled account",
            "Use documented credentials and recovery paths when available",
            "Produce a short incident timeline after access is restored",
        ):
            assert expected in stable

        assert "Break-glass trigger phrases include" not in parts["context"]
        assert "Break-glass trigger phrases include" not in parts["volatile"]

    def test_authorized_asset_registry_guidance_is_in_stable_prompt(self):
        parts = _build_parts(_make_agent())
        stable = parts["stable"]

        assert "AgentCyber authorized asset registry" in stable
        for expected in (
            "known Breaking Circuits, owned, or lab assets as in scope by default",
            "domains, repos, local machines, Proxmox hosts, VMs/containers",
            "lab APIs/services, Discord/Hermes gateway identifiers, and cloud subscriptions/accounts",
            "Use the registry to reduce repetitive authorization questions",
            "Unknown assets trigger scoping, not refusal",
            "Store sensitive fields separately from non-sensitive metadata",
            "Never expose registry secrets in normal summaries",
        ):
            assert expected in stable

        assert "AgentCyber authorized asset registry" not in parts["context"]
        assert "AgentCyber authorized asset registry" not in parts["volatile"]

    def test_model_routing_guidance_is_in_stable_prompt(self):
        parts = _build_parts(_make_agent())
        stable = parts["stable"]

        assert "AgentCyber model routing" in stable
        for expected in (
            "Route cyber-sensitive but authorized tasks to local or open-weight models when available",
            "credentials, malware analysis, exploit testing, lockout recovery, or incident response",
            "Keep Azure and hosted models available for ordinary planning, summarization, coding, and general reasoning",
            "Honor explicit operator overrides such as use local model, use Azure, or use cyber route",
            "If the preferred local model is unavailable, say so clearly",
            "ask before sending secrets to a hosted model",
            "Log routing decisions safely without exposing secrets",
        ):
            assert expected in stable

        assert "AgentCyber model routing" not in parts["context"]
        assert "AgentCyber model routing" not in parts["volatile"]

    def test_serious_work_tone_guidance_is_in_stable_prompt(self):
        parts = _build_parts(_make_agent())
        stable = parts["stable"]

        assert "AgentCyber serious-work tone controls" in stable
        for expected in (
            "Do not mirror the operator's profanity, intensity, or joking style by default",
            "Maintain a calm, direct, grounded tone under pressure",
            "Be non-robotic but not performative",
            "Use concise, serious language during incidents",
            "act as the team's operational counterbalance",
            "Avoid playful banter during lockout, incident response, credential recovery, or testing deadlines",
            "Never let tone or persona interfere with task completion",
        ):
            assert expected in stable

        assert "AgentCyber serious-work tone controls" not in parts["context"]
        assert "AgentCyber serious-work tone controls" not in parts["volatile"]


class TestContextFileCwd:
    def test_none_when_terminal_cwd_unset(self, monkeypatch):
        # Unset → None, so discovery falls back to the launch dir inside
        # build_context_files_prompt (the local-CLI #19242 contract).
        monkeypatch.delenv("TERMINAL_CWD", raising=False)
        assert _captured_context_cwd(_make_agent()) is None

    def test_configured_dir_when_terminal_cwd_set(self, monkeypatch, tmp_path):
        monkeypatch.setenv("TERMINAL_CWD", str(tmp_path))
        assert _captured_context_cwd(_make_agent()) == tmp_path


def _stable_prompt(agent):
    with (
        patch("run_agent.load_soul_md", return_value=""),
        patch("run_agent.build_nous_subscription_prompt", return_value=""),
        patch("run_agent.build_environment_hints", return_value=""),
        patch("run_agent.build_context_files_prompt", return_value=""),
    ):
        return build_system_prompt_parts(agent)["stable"]


class TestCodingContextBlock:
    def test_injected_when_active(self, monkeypatch, tmp_path):
        import subprocess

        subprocess.run(["git", "-C", str(tmp_path), "init", "-q"], check=True)
        monkeypatch.setenv("TERMINAL_CWD", str(tmp_path))
        agent = _make_agent(valid_tool_names=["read_file"], platform="cli")
        stable = _stable_prompt(agent)
        assert "coding agent" in stable
        assert "Workspace" in stable

    def test_absent_when_off(self, monkeypatch, tmp_path):
        import subprocess

        subprocess.run(["git", "-C", str(tmp_path), "init", "-q"], check=True)
        monkeypatch.setenv("TERMINAL_CWD", str(tmp_path))
        agent = _make_agent(valid_tool_names=["read_file"], platform="cli")
        # Drive the real path: force the resolved mode to "off" via config.
        with patch("agent.coding_context._coding_mode", return_value="off"):
            stable = _stable_prompt(agent)
        assert "coding agent" not in stable

    def test_absent_without_tools(self, monkeypatch, tmp_path):
        import subprocess

        subprocess.run(["git", "-C", str(tmp_path), "init", "-q"], check=True)
        monkeypatch.setenv("TERMINAL_CWD", str(tmp_path))
        agent = _make_agent(valid_tool_names=[], platform="cli")
        assert "coding agent" not in _stable_prompt(agent)
