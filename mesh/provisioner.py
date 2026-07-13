"""Provisioner — orchestrates `hermes mesh add` end-to-end.

Pure stdlib for v0.1 except `paho-mqtt` (registry/validation) and `pyyaml` (config).
SSH/SCP via subprocess. Templating via string.Template.
"""
from __future__ import annotations

import os
import shlex
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from string import Template
from typing import Optional

from . import registry, validation


# ============================================================================
# Dataclasses
# ============================================================================

@dataclass
class NodeSpec:
    """Operator's intent — what they're asking for. Source: CLI flags or yaml.

    Discovered facts about the remote (actual runtime user, OS, arch)
    live in ProbeResult, not here. Keep intent and reality separate.
    """
    host: str
    role: str
    namespace: str
    broker: str
    capabilities: list[str] = field(default_factory=list)
    user: Optional[str] = None  # SSH user; runtime user comes from probe.


@dataclass
class ProbeResult:
    """What we discovered on the remote when we SSH'd in (Step 1)."""
    uname: str
    hostname: str
    user: str


class _Redacted(str):
    """str subclass that hides its value from repr/print/log.

    Wrap secrets so accidental `print(vars_)` or stray logging doesn't leak them.

    Caveat: hides from repr() and dict-print only. f-string interpolation
    (e.g. ``f"{password}"``) goes through __str__ and DOES leak. That's
    intentional — _Redacted has to be useful in the env-rendering path.
    """
    __slots__ = ()

    def __repr__(self) -> str:
        return "'***'"


@dataclass
class ControllerConfig:
    namespace: str
    broker: str
    broker_user: str
    broker_password: _Redacted
    ca_cert_path: Optional[Path]
    template_dirs: list[Path]
    # template_dirs order: user override dir FIRST, repo defaults SECOND.
    # Constructed in mesh.config.load() / init_interactive(); see config.py.


# ============================================================================
# SSH / SCP primitives
# ============================================================================

def ssh_run(
    host: str,
    command: str,
    user: Optional[str] = None,
    timeout: int = 30,
    check: bool = True,
    capture: bool = True,
) -> subprocess.CompletedProcess:
    """Run a remote command. Raises by default on non-zero exit."""
    target = f"{user}@{host}" if user else host
    cmd = ["ssh", "-o", "BatchMode=yes", "-o", f"ConnectTimeout={timeout}", target, command]
    return subprocess.run(
        cmd,
        capture_output=capture,
        text=True,
        check=check,
        timeout=timeout + 5,
    )


def scp_put(host: str, local: Path, remote: str, user: Optional[str] = None, timeout: int = 60, preserve: bool = False) -> None:
    target = f"{user}@{host}:{remote}" if user else f"{host}:{remote}"
    # preserve=True passes scp -p so the source file's mode (e.g. 0600 for a
    # secrets file) is carried to the remote, avoiding a world-readable window
    # on the node before any follow-up chmod.
    cmd = ["scp", "-q", "-o", "BatchMode=yes", "-o", f"ConnectTimeout={timeout}"]
    if preserve:
        cmd.append("-p")
    cmd += [str(local), target]
    subprocess.run(cmd, check=True, timeout=timeout + 5)


# ============================================================================
# Pipeline steps
# ============================================================================

def probe(spec: NodeSpec) -> ProbeResult:
    """Step 1: verify SSH reachability + capture host facts."""
    out = ssh_run(spec.host, "uname -srm && hostname && id -un", user=spec.user, timeout=10)
    lines = out.stdout.strip().splitlines()
    return ProbeResult(uname=lines[0], hostname=lines[1], user=lines[2])


def render_role(spec: NodeSpec, cfg: ControllerConfig, facts: ProbeResult) -> dict[str, str]:
    """Steps 2–5: find role template, render python script + systemd unit + .env."""
    role_dir = _find_role_template(spec.role, cfg.template_dirs)
    if role_dir is None:
        raise FileNotFoundError(
            f"Role template not found: {spec.role}. Searched: "
            f"{[str(p) for p in cfg.template_dirs]}"
        )

    # DO NOT LOG vars_ — contains broker_password.
    vars_ = {
        "host": spec.host,
        "namespace": spec.namespace,
        "broker": spec.broker,
        "broker_user": cfg.broker_user,
        "broker_password": str(cfg.broker_password),  # actual value into rendered file
        "capabilities": ",".join(spec.capabilities),
        "service_name": f"{spec.namespace}-{spec.host}",
        "node_user": facts.user,
    }
    return {
        "script": Template((role_dir / "script.py.tmpl").read_text()).substitute(vars_),
        "service": Template((role_dir / "service.tmpl").read_text()).substitute(vars_),
        "env": _render_env(vars_, cfg),
        "deps": (role_dir / "deps.txt").read_text() if (role_dir / "deps.txt").exists() else "",
    }


def deploy(
    spec: NodeSpec,
    rendered: dict[str, str],
    facts: ProbeResult,
    cfg: ControllerConfig,
    skip_tls_setup: bool = False,
) -> None:
    """Steps 6–8: scp files, install deps, enable service.

    Idempotent: re-running on an already-provisioned node updates files in place.
    Drops a `.provisioned-by-hermes-mesh-<service>` marker as the rollback gate.
    """
    service_name = f"{spec.namespace}-{spec.host}"
    home = f"/home/{facts.user}"
    script_path = f"{home}/{service_name}.py"
    env_path = f"{home}/.{service_name}.env"
    service_path_remote = f"/etc/systemd/system/{service_name}.service"
    venv_path = f"{home}/{spec.namespace}-venv"
    marker_path = f"{home}/.provisioned-by-hermes-mesh-{service_name}"

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "script.py").write_text(rendered["script"])
        (tmp / "service").write_text(rendered["service"])
        # The .env holds broker_password — create it 0600 from the start; scp -p
        # below carries that mode to the remote so it is never briefly
        # world-readable on the node ahead of the chmod.
        _env_fd = os.open(tmp / "env", os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(_env_fd, "w") as _env_f:
            _env_f.write(rendered["env"])

        # Upload script + env (user-writable destinations).
        scp_put(spec.host, tmp / "script.py", script_path, user=spec.user)
        scp_put(spec.host, tmp / "env", env_path, user=spec.user, preserve=True)
        ssh_run(spec.host, f"chmod 600 {shlex.quote(env_path)}", user=spec.user)

        # Service unit goes under /etc — stage in /tmp then sudo-move.
        scp_put(spec.host, tmp / "service", f"/tmp/{service_name}.service", user=spec.user)
        ssh_run(
            spec.host,
            f"sudo -n mv /tmp/{shlex.quote(service_name)}.service {shlex.quote(service_path_remote)} "
            f"&& sudo -n chown root:root {shlex.quote(service_path_remote)} "
            f"&& sudo -n chmod 644 {shlex.quote(service_path_remote)}",
            user=spec.user,
        )

    # Optional TLS CA cert distribution.
    if cfg.ca_cert_path and not skip_tls_setup:
        ca_remote_dir = f"/etc/{spec.namespace}"
        ca_remote = f"{ca_remote_dir}/ca.crt"
        scp_put(spec.host, cfg.ca_cert_path, f"/tmp/ca.crt", user=spec.user)
        ssh_run(
            spec.host,
            f"sudo -n mkdir -p {shlex.quote(ca_remote_dir)} "
            f"&& sudo -n mv /tmp/ca.crt {shlex.quote(ca_remote)} "
            f"&& sudo -n chmod 644 {shlex.quote(ca_remote)}",
            user=spec.user,
        )

    # Venv + deps.
    ssh_run(
        spec.host,
        f"test -d {shlex.quote(venv_path)} || python3 -m venv {shlex.quote(venv_path)}",
        user=spec.user,
        timeout=60,
    )
    deps = (rendered.get("deps") or "").strip()
    if deps:
        # deps.txt is line-per-package; write to tmp and pip install -r.
        deps_remote = f"/tmp/{service_name}.deps.txt"
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".txt") as f:
            f.write(deps + "\n")
            deps_local = Path(f.name)
        try:
            scp_put(spec.host, deps_local, deps_remote, user=spec.user)
            ssh_run(
                spec.host,
                f"{shlex.quote(venv_path)}/bin/pip install --upgrade --disable-pip-version-check "
                f"-r {shlex.quote(deps_remote)}",
                user=spec.user,
                timeout=180,
            )
        finally:
            deps_local.unlink(missing_ok=True)

    # Drop the rollback marker BEFORE enabling the service. Rollback only
    # touches state where this marker exists.
    ssh_run(
        spec.host,
        f"touch {shlex.quote(marker_path)}",
        user=spec.user,
    )

    # Enable + start (idempotent).
    ssh_run(
        spec.host,
        f"sudo -n systemctl daemon-reload "
        f"&& sudo -n systemctl enable --now {shlex.quote(service_name + '.service')}",
        user=spec.user,
        timeout=30,
    )


def validate_and_register(spec: NodeSpec, facts: ProbeResult, cfg: ControllerConfig) -> None:
    """Steps 9–10: wait for heartbeat, publish manifest, append nodes.yaml.

    `facts.user` (probe-discovered runtime user) is threaded into nodes.yaml
    so the registry holds ground truth, not operator intent.
    """
    if not validation.wait_for_heartbeat(spec.namespace, spec.host, cfg, timeout=30):
        raise TimeoutError(f"Node {spec.host} never published heartbeat — rolling back")
    registry.publish_manifest(spec, cfg)
    registry.append_to_nodes_yaml(spec, cfg, node_user=facts.user)


# ============================================================================
# High-level entry points
# ============================================================================

def init_controller():
    """`hermes mesh init` — delegates to mesh.config.init_interactive."""
    from . import config
    return config.init_interactive()


def ssh_setup(host: str, user: Optional[str] = None, sudoers_for_systemctl: bool = True) -> None:
    """`hermes mesh ssh-setup <host>` — bootstrap passwordless SSH + (optional) sudoers.

    Steps:
      1. Generate ~/.ssh/hermes_mesh_ed25519 if missing.
      2. Copy public key to <user>@<host> via ssh-copy-id.
      3. Verify passwordless SSH works.
      4. (sudoers_for_systemctl=True, default) Drop /etc/sudoers.d/hermes-mesh-<user>
         allowing NOPASSWD for systemctl + journalctl. Required for `mesh add`
         to enable services without prompting.
    """
    key_path = Path.home() / ".ssh" / "hermes_mesh_ed25519"
    pub_path = key_path.with_suffix(".pub")
    if not key_path.exists():
        subprocess.run(
            ["ssh-keygen", "-t", "ed25519", "-f", str(key_path), "-N", "", "-C", "hermes-mesh"],
            check=True,
        )
        print(f"✓ Generated {key_path}")
    else:
        print(f"  Reusing existing {key_path}")

    target = f"{user}@{host}" if user else host
    pubkey_text = pub_path.read_text().strip()

    # Try to install the pubkey via existing SSH trust first — no password prompt.
    # If that fails, fall back to ssh-copy-id (which will prompt once).
    install_cmd = (
        f"mkdir -p ~/.ssh && chmod 700 ~/.ssh && "
        f"touch ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys && "
        f"grep -qxF {shlex.quote(pubkey_text)} ~/.ssh/authorized_keys || "
        f"echo {shlex.quote(pubkey_text)} >> ~/.ssh/authorized_keys"
    )
    try:
        ssh_run(host, install_cmd, user=user, timeout=10)
        print(f"✓ Public key authorized on {target} (existing SSH trust)")
    except subprocess.CalledProcessError:
        # No existing trust — ssh-copy-id will prompt for password ONCE.
        # The one acceptable interactive step in the whole pipeline.
        subprocess.run(["ssh-copy-id", "-i", str(pub_path), target], check=True)
        print(f"✓ Public key authorized on {target} (ssh-copy-id)")

    # Verify passwordless SSH works
    verify = ssh_run(host, "echo ok", user=user, timeout=10)
    if verify.stdout.strip() != "ok":
        raise RuntimeError(f"Passwordless SSH verification failed: {verify.stdout!r}")
    print("✓ Passwordless SSH verified")

    if not sudoers_for_systemctl:
        return

    # Drop a scoped sudoers file. Find paths to systemctl/journalctl on the remote.
    paths_out = ssh_run(
        host,
        "command -v systemctl; command -v journalctl; command -v mkdir; command -v mv; command -v chown; command -v chmod",
        user=user, timeout=10,
    ).stdout.strip().splitlines()
    if len(paths_out) < 6:
        print("  (sudoers skipped — couldn't locate all required binaries on remote)")
        return
    systemctl, journalctl, mkdir_bin, mv_bin, chown_bin, chmod_bin = paths_out[:6]

    # Discover the deploy user on remote (matches probe).
    remote_user = ssh_run(host, "id -un", user=user, timeout=10).stdout.strip()
    sudoers_content = (
        f"# Managed by `hermes mesh ssh-setup` — provisioner needs sudo-NOPASSWD\n"
        f"# for systemd + file moves under /etc. Scoped to specific binaries.\n"
        f"{remote_user} ALL=(root) NOPASSWD: {systemctl}, {journalctl}, "
        f"{mkdir_bin}, {mv_bin}, {chown_bin}, {chmod_bin}\n"
    )
    sudoers_path = f"/etc/sudoers.d/hermes-mesh-{remote_user}"
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".sudoers") as f:
        f.write(sudoers_content)
        local_sudoers = Path(f.name)
    try:
        scp_put(host, local_sudoers, "/tmp/hermes-mesh.sudoers", user=user)
        # Validate with visudo before installing — refuse to ship a broken sudoers file.
        # Use `install -o root -g root -m 440` for an atomic move-with-ownership;
        # avoids the cosmetic window where the file is in /etc/sudoers.d with
        # wrong perms/owner and sudo warns mid-execution.
        ssh_run(
            host,
            f"sudo visudo -cf /tmp/hermes-mesh.sudoers "
            f"&& sudo install -o root -g root -m 440 /tmp/hermes-mesh.sudoers "
            f"{shlex.quote(sudoers_path)} "
            f"&& rm -f /tmp/hermes-mesh.sudoers",
            user=user, timeout=20,
        )
        print(f"✓ Wrote {sudoers_path} (visudo-validated)")
    finally:
        local_sudoers.unlink(missing_ok=True)


def add_node(spec: NodeSpec, cfg: ControllerConfig, rollback_on_failure: bool = True, skip_tls_setup: bool = False) -> None:
    """`hermes mesh add` — full pipeline with rollback on validation failure."""
    print(f"→ {spec.host}: probing…")
    facts = probe(spec)
    print(f"  uname:    {facts.uname}")
    print(f"  hostname: {facts.hostname}")
    print(f"  user:     {facts.user}")

    print(f"→ {spec.host}: rendering role '{spec.role}'…")
    rendered = render_role(spec, cfg, facts)

    try:
        print(f"→ {spec.host}: deploying…")
        deploy(spec, rendered, facts, cfg, skip_tls_setup=skip_tls_setup)
        print(f"→ {spec.host}: waiting for heartbeat (≤30s)…")
        validate_and_register(spec, facts, cfg)
    except Exception:
        if rollback_on_failure:
            print(f"✗ {spec.host}: failed — rolling back…")
            try:
                _rollback(spec, facts, cfg)
            except Exception as rb_exc:  # noqa: BLE001
                print(f"  [rollback warning] cleanup failed: {rb_exc}", flush=True)
        raise

    print(f"✓ {spec.host}: provisioned and registered")


def remove_node(host: str, cfg: ControllerConfig, purge: bool = False) -> None:
    """`hermes mesh remove` — decommission a node.

    Default: disable service, unpublish retained topics, remove nodes.yaml entry.
             Files stay on disk.
    --purge: also nuke script, service unit, .env, venv, marker.
    """
    # Pull SSH user from nodes.yaml if registered. A host can appear under
    # multiple namespaces; prefer the controller's own namespace, then fall
    # back to the first match.
    all_nodes = registry.list_nodes()
    matches = [n for n in all_nodes if n["host"] == host]
    if not matches:
        entry = None
    else:
        entry = next((n for n in matches if n.get("namespace") == cfg.namespace), matches[0])
    ssh_user = entry.get("user") if entry else None
    node_namespace = entry.get("namespace", cfg.namespace) if entry else cfg.namespace
    service_name = f"{node_namespace}-{host}"

    # Determine the runtime user for path resolution. Probe is ground truth;
    # nodes.yaml entry is the fallback. We refuse to guess — silently targeting
    # the wrong home dir would be worse than failing fast.
    try:
        runtime_user = ssh_run(host, "id -un", user=ssh_user, timeout=10).stdout.strip()
    except subprocess.SubprocessError:
        if ssh_user:
            runtime_user = ssh_user
            print(f"  [warn] SSH unreachable; using nodes.yaml user '{runtime_user}' for path resolution")
        else:
            raise RuntimeError(
                f"Cannot determine runtime user for {host} — SSH unreachable AND no "
                f"nodes.yaml entry. Either restore SSH and retry, or clean up manually "
                f"(see docs/mesh/SECURITY.md for manual-cleanup steps)."
            )

    home = f"/home/{runtime_user}"
    marker = f"{home}/.provisioned-by-hermes-mesh-{service_name}"

    # Stop + disable (idempotent).
    try:
        ssh_run(
            host,
            f"sudo -n systemctl disable --now {shlex.quote(service_name + '.service')} 2>/dev/null || true",
            user=ssh_user, timeout=15, check=False,
        )
    except Exception as e:
        print(f"  [warn] could not stop service: {e}")

    # Clear retained topics.
    try:
        registry.unpublish_manifest(host, cfg)
    except Exception as e:
        print(f"  [warn] could not clear registry topic: {e}")
    try:
        # Also clear the alive retained topic.
        _clear_retained(cfg, f"{cfg.namespace}/{host}/alive")
    except Exception as e:
        print(f"  [warn] could not clear alive topic: {e}")

    # nodes.yaml entry.
    registry.remove_from_nodes_yaml(host, namespace=node_namespace)

    if purge:
        try:
            ssh_run(
                host,
                f"sudo -n rm -f /etc/systemd/system/{shlex.quote(service_name + '.service')} "
                f"&& sudo -n systemctl daemon-reload "
                f"&& rm -f {shlex.quote(home + '/' + service_name + '.py')} "
                f"        {shlex.quote(home + '/.' + service_name + '.env')} "
                f"        {shlex.quote(marker)}",
                user=ssh_user, timeout=20, check=False,
            )
        except Exception as e:
            print(f"  [warn] purge cleanup failed: {e}")

    print(f"✓ {host}: removed{' (purged)' if purge else ''}")


# ============================================================================
# Helpers
# ============================================================================

def _find_role_template(role: str, dirs: list[Path]) -> Optional[Path]:
    for d in dirs:
        candidate = d / role
        if candidate.is_dir():
            return candidate
    return None


def _render_env(vars_: dict, cfg: ControllerConfig) -> str:
    # NOTE: vars_ contains broker_password — DO NOT LOG.
    # v0.2: capabilities should move to JSON array (comma-join breaks on richer types).
    lines = [
        f"MQTT_BROKER={vars_['broker']}",
        f"MQTT_USER={vars_['broker_user']}",
        f"MQTT_PASSWORD={vars_['broker_password']}",
        f"NAMESPACE={vars_['namespace']}",
        f"NODE_HOST={vars_['host']}",
    ]
    if cfg.ca_cert_path:
        lines.append(f"MQTT_CA_CERT=/etc/{vars_['namespace']}/ca.crt")
    return "\n".join(lines) + "\n"


def _clear_retained(cfg: ControllerConfig, topic: str) -> None:
    """Publish empty payload with retain=True to clear a retained topic."""
    import paho.mqtt.publish as mqtt_publish
    auth = {"username": cfg.broker_user, "password": str(cfg.broker_password)}
    tls = None
    port = 1883
    if cfg.ca_cert_path:
        tls = {"ca_certs": str(cfg.ca_cert_path)}
        port = 8883
    mqtt_publish.single(
        topic, payload="", qos=1, retain=True,
        hostname=cfg.broker, port=port, auth=auth, tls=tls,
    )


def _rollback(spec: NodeSpec, facts: ProbeResult, cfg: ControllerConfig) -> None:
    """Best-effort cleanup when `add` fails mid-pipeline.

    Marker-guarded: only acts if `.provisioned-by-hermes-mesh-<service>` exists,
    so we never touch pre-existing nodes. Step-wise: each cleanup in its own try;
    one failure doesn't abort the rest. The original add-failure exception
    is always what surfaces to the caller — rollback errors are logged only.
    """
    service_name = f"{cfg.namespace}-{spec.host}"
    home = f"/home/{facts.user}"
    marker = f"{home}/.provisioned-by-hermes-mesh-{service_name}"

    # Guard: only roll back if WE provisioned this node.
    try:
        check = ssh_run(spec.host, f"test -f {shlex.quote(marker)} && echo yes || echo no",
                        user=spec.user, timeout=10, check=False)
        if check.stdout.strip() != "yes":
            print(f"  [rollback] no marker on {spec.host} — skipping (preserves existing state)")
            return
    except Exception as e:
        print(f"  [rollback] marker check failed: {e} — refusing to touch state")
        return

    steps = [
        ("stop service",
         f"sudo -n systemctl disable --now {shlex.quote(service_name + '.service')} 2>/dev/null || true"),
        ("remove service unit",
         f"sudo -n rm -f /etc/systemd/system/{shlex.quote(service_name + '.service')} "
         f"&& sudo -n systemctl daemon-reload"),
        ("remove script + env + marker",
         f"rm -f {shlex.quote(home + '/' + service_name + '.py')} "
         f"      {shlex.quote(home + '/.' + service_name + '.env')} "
         f"      {shlex.quote(marker)}"),
    ]
    for label, cmd in steps:
        try:
            ssh_run(spec.host, cmd, user=spec.user, timeout=20, check=False)
        except Exception as e:
            print(f"  [rollback] {label}: {e}")

    # Clear retained topics best-effort.
    for topic in (f"{cfg.namespace}/registry/{spec.host}", f"{cfg.namespace}/{spec.host}/alive"):
        try:
            _clear_retained(cfg, topic)
        except Exception as e:
            print(f"  [rollback] clear {topic}: {e}")
