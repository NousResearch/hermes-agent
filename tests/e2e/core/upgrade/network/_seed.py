"""One real install per module, then every cell runs inside a private network namespace.

``seed_install`` runs HEAD's ``scripts/install.sh`` into an empty fake HOME the way
``test_install_fresh`` does, with the clone rewritten to a local bare origin and on the host
network, because the installer provisions ~100 MB of pinned upstream artifacts (Python, Node,
ffmpeg, ...) that no fixture may fake. The install is HEAD, so the updater under test is HEAD's
own; each cell then publishes a NEW upstream commit (or a new release record), so the update is
a real fetch, checkout and rebuild, never the "already current" branch.

After the seed the sandbox loses the ``insteadOf`` rewrite: the checkout's origin is the official
``https://github.com/NousResearch/hermes-agent.git`` again, and inside the namespace the only way
to reach it is the test's proxy, which routes ``github.com`` to a git smart-HTTP server over the
same bare origin. The partial clone (``--filter=blob:none``) makes every lazy blob fetch of the
checkout cross that proxy too.

``Installed.run(..., edge=...)`` runs one command under ``bwrap --unshare-net``: no route, no
DNS, only the edge's proxy (and optional plain-HTTP mirrors) bridged onto the namespace's
loopback. The module fixture proves that isolation before any cell runs.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import re
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator

from tests.e2e.core.upgrade import _helpers as H
from tests.e2e.core.upgrade import _install_helpers as I
from tests.e2e.core.upgrade.network import _netedge as N

ASSETS = "hermes-assets.nousresearch.com"
REPOSITORY = "NousResearch/hermes-agent"
# Hosts a correctly isolated update must never reach directly; with the proxy they appear in
# the proxy log as "refused" (the proxy has no route for them).
PUBLIC_INDEXES = ("pypi.org", "files.pythonhosted.org", "registry.npmjs.org")


@dataclass
class Result:
    cp: subprocess.CompletedProcess
    secs: float
    edge: Edge | None

    @property
    def rc(self) -> int:
        return self.cp.returncode

    @property
    def out(self) -> str:
        return (self.cp.stdout or "") + (self.cp.stderr or "")

    def report(self, inst: Installed, *extra: str) -> str:
        parts = [f"exit={self.rc} after {self.secs:.1f}s", I.describe(self.cp)]
        if self.edge is not None:
            parts.append("--- network edge log ---\n" + self.edge.transcript())
        parts.append("--- install state ---\n" + json.dumps(inst.state(), indent=1))
        parts.append(inst.logs_tail())
        parts.extend(extra)
        return "\n".join(parts)


@dataclass
class Edge:
    """The namespace's only egress: a TLS-inspecting proxy plus optional plain-HTTP mirrors."""

    inst: Installed
    proxy: N.EdgeProxy
    sites: list[N.HttpSite] = field(default_factory=list)
    use_auth: bool = False

    @property
    def ports(self) -> list[int]:
        return [self.proxy.port, *(s.port for s in self.sites)]

    def env(self) -> dict[str, str]:
        return N.proxy_env(self.proxy.url_with_auth() if self.use_auth else self.proxy.url)

    def transcript(self) -> str:
        rows = [self.proxy.transcript(80)]
        for site in self.sites:
            rows.append(f"[mirror {site.url}] " + (", ".join(str(h) for h in site.hits[-40:]) or "(no requests)"))
        return "\n".join(rows)

    def close(self) -> None:
        self.proxy.close()
        for site in self.sites:
            site.close()


@dataclass
class Installed:
    root: Path
    sb: I.Sandbox
    origin: Path
    gitroot: Path
    ca: N.TestCA
    trust: list[tuple[Path, Path]]
    _n: int = 0

    # -- upstream --------------------------------------------------------------
    def head(self) -> str:
        return I.git("rev-parse", "HEAD", cwd=self.sb.checkout)

    def publish(self, tag: str, files: dict[str, str] | None = None) -> str:
        """A new upstream commit on origin/main, directly on top of what is installed."""
        self._n += 1
        I.git("update-ref", "refs/heads/main", self.head(), cwd=self.origin)
        files = files or {f"E2E_NETWORK_{tag}.txt": f"{tag} {time.time_ns()}\n"}
        return I.publish_commit(self.origin, self.root, f"e2e network: {tag} #{self._n}", files)

    # -- edges -----------------------------------------------------------------
    def edge(self, *, assets: N.App | None = None, git: N.App | None = None,
             extra: dict[str, N.App] | None = None, sites: Iterable[N.HttpSite] = (),
             eof_hosts: Iterable[str] = (), auth: tuple[str, str] | None = None) -> Edge:
        """Proxy routes: the release-channel host (default: no records published, so ``main``
        follows the branch), github.com (default: the git server over the bare origin)."""
        routes = {ASSETS: assets or N.static_app({}), "github.com": git or N.git_app(self.gitroot)}
        routes.update(extra or {})
        for host in eof_hosts:
            routes.pop(host, None)
        proxy = N.EdgeProxy(self.ca, routes, eof_hosts=eof_hosts, auth=auth)
        return Edge(self, proxy, list(sites), use_auth=auth is not None)

    # -- running -----------------------------------------------------------------
    def run(self, argv: list[str], *, edge: Edge | None, env: dict[str, str] | None = None,
            corporate_root: bool = True, timeout: float = 900) -> Result:
        """``argv`` inside the sandbox with its own network namespace. ``edge=None`` is fully
        offline. ``corporate_root`` overlays the distro trust store with one that also trusts
        the proxy's root (what an admin's ``update-ca-certificates`` leaves behind)."""
        e = dict(self.sb.env)
        if edge is not None:
            e.update(edge.env())
        e.update(env or {})
        bridge = N.Bridge(self.root / "bridges" / f"b{time.time_ns()}", edge.ports if edge else [])
        t0 = time.monotonic()
        try:
            cp = H.run(bridge.wrap(argv), env=e, cwd=self.sb.root, writable=[self.sb.root],
                       timeout=timeout, unshare_net=True, ro_binds=self.trust if corporate_root else ())
        finally:
            bridge.close()
        return Result(cp, time.monotonic() - t0, edge)

    def hermes(self, *args: str, edge: Edge | None, **kw) -> Result:
        return self.run([self.sb.hermes, *args], edge=edge, **kw)

    # -- observation -------------------------------------------------------------
    def state(self) -> dict:
        """What an update is allowed to change: the checkout HEAD, the selected PM
        environment, the PM runtime generation, and the installed tool store."""
        from pm.environments import install_key

        key = install_key(self.sb.checkout)
        inst = self.sb.hermes_home / "installs" / key

        def read(p: Path) -> str:
            try:
                return hashlib.sha256(p.read_bytes()).hexdigest()[:16]
            except OSError:
                return "missing"

        tools = self.sb.hermes_home / "tools"
        return {
            "head": self.head(),
            "facts": read(inst / "facts.json"),
            "pm_runtime": read(inst / "pm-runtime" / "selected.json"),
            "tools": sorted(p.name for p in tools.iterdir() if p.is_dir()) if tools.is_dir() else [],
        }

    def logs_tail(self, lines: int = 40) -> str:
        out = []
        for log in sorted((self.sb.hermes_home / "logs").glob("*.log")):
            text = log.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:]
            out.append(f"--- {log.name} (tail) ---\n" + "\n".join(text))
        return "\n".join(out) or "(no logs)"

    def version_works(self) -> Result:
        """The next ``hermes`` invocation, offline: the install must still start."""
        return self.hermes("--version", edge=None, timeout=120)


def seed_install(root: Path) -> Installed:
    root.mkdir(parents=True, exist_ok=True)
    origin = I.make_origin(root, I.head_sha())
    I.git("config", "uploadpack.allowFilter", "true", cwd=origin)
    sb = I.new_sandbox(root / "sb", origin)
    # The host's SSL_CERT_FILE (allowlisted by _helpers) would replace the sandbox's trust store
    # and hide the corporate root the cells install; the cells control trust explicitly.
    sb.env.pop("SSL_CERT_FILE", None)
    script = sb.root / "install.sh"
    shutil.copy(H.WORKTREE / "scripts" / "install.sh", script)
    cp = sb.run(["bash", str(script), "--non-interactive", "--skip-browser"], timeout=1800, input="")
    assert cp.returncode == 0, "seed install failed:\n" + I.describe(cp)
    # From here on the checkout talks to the official URL, which only the proxy can serve.
    (sb.home / ".gitconfig").write_text("", encoding="utf-8")
    gitroot = root / "gitroot"
    (gitroot / "NousResearch").mkdir(parents=True, exist_ok=True)
    (gitroot / "NousResearch" / "hermes-agent.git").symlink_to(origin)
    ca = N.TestCA(root / "corporate-ca")
    inst = Installed(root, sb, origin, gitroot, ca, ca.os_trust_store(root / "corporate-ca" / "etc-ssl-certs"))
    assert_isolated(inst)
    return inst


def assert_isolated(inst: Installed) -> None:
    """The namespace has no route out and no DNS; only the bridged proxy port answers."""
    probe = (
        "import socket,sys\n"
        "bad=[]\n"
        "for host in ('github.com','pypi.org','hermes-assets.nousresearch.com'):\n"
        "    try: socket.getaddrinfo(host,443); bad.append('dns:'+host)\n"
        "    except OSError: pass\n"
        "for ip in ('140.82.112.3','1.1.1.1'):\n"
        "    s=socket.socket(); s.settimeout(3)\n"
        "    try: s.connect((ip,443)); bad.append('tcp:'+ip)\n"
        "    except OSError: pass\n"
        "print('LEAK' if bad else 'ISOLATED', bad)\n"
    )
    edge = inst.edge()
    try:
        r = inst.run([inst.sb.python, "-c", probe], edge=edge, timeout=60)
    finally:
        edge.close()
    assert r.rc == 0 and "ISOLATED" in r.cp.stdout, "network namespace is not isolated:\n" + I.describe(r.cp)


# ---------------------------------------------------------------------------
# Release-channel records (the R2 objects under https://hermes-assets.nousresearch.com/).
# ---------------------------------------------------------------------------

def canonical(value: object) -> bytes:
    """The channel wire encoding (sorted keys, compact, trailing newline)."""
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode()


def _identity(name: str, token: str) -> dict:
    pascal = f"HermesChannel{token}"
    return {"token": token, "displayName": f"Hermes {name}", "appId": f"ai.hermes.channel.h{token}",
            "appNamePascal": pascal, "artifactNamePascal": pascal, "cliName": f"hermes-{name}",
            "windowsExecutableName": pascal, "msixAppIdWithOrg": f"NousResearch.{pascal}"}


def stable_objects(commit: str, *, version: str = "2099.1.1", build_id: str = "c" * 32,
                   repository: str = REPOSITORY) -> dict[str, bytes]:
    """A published stable release pinned at ``commit``: the channel record and its build manifest."""
    identity = _identity("stable", "5" * 16)
    prefix = f"releases/channel-builds/{build_id}/"
    request = {"schema": 1, "buildId": build_id, "channel": "stable", "sequence": 1,
               "repository": repository, "commit": commit, "sourceVersion": version, "version": version,
               "windowsVersion": version + ".0", "releaseTag": "v" + version, "identity": identity,
               "bundleEnv": {}, "publicBase": f"https://{ASSETS}"}
    manifest = {"schema": 1, "receiverProtocol": 1, "request": request, "packages": [
        {"platform": "darwin", "arch": "arm64", "variant": "bundled", "identity": identity["appId"],
         "version": version, "teamId": "ABCDEFGHIJ",
         "artifact": {"key": prefix + "Hermes.dmg", "sha256": "d" * 64, "size": 100},
         "feed": {"key": prefix + "stable-mac.yml", "channel": "stable"}}]}
    body = canonical(manifest)
    record = {"schema": 1, "name": "stable", "repository": repository, "policy": "stable-release",
              "state": "active", "revision": 1, "nextSequence": 2, "identity": identity,
              "head": {"buildId": build_id, "sequence": 1, "manifestKey": prefix + "build.json",
                       "sha256": hashlib.sha256(body).hexdigest()}}
    return {"/releases/channels/stable.json": canonical(record), f"/{prefix}build.json": body}


# ---------------------------------------------------------------------------
# Mirrors: a PEP 503 simple index and an npm registry holding a small pre-fetched set.
# ---------------------------------------------------------------------------

def prefetch(url: str, sha256: str, dest: Path) -> Path:
    """Fixture-time download (host network, like the seed install) verified against its pin."""
    import urllib.request

    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and hashlib.sha256(dest.read_bytes()).hexdigest() == sha256:
        return dest
    with urllib.request.urlopen(url, timeout=120) as resp:
        data = resp.read()
    got = hashlib.sha256(data).hexdigest()
    assert got == sha256, f"prefetched {url} has sha256 {got}, pinned {sha256}"
    dest.write_bytes(data)
    return dest


def lock_wheels(lock: Path) -> list[tuple[str, str, str]]:
    """``(project, wheel_url, sha256)`` for every pure-Python wheel pinned in a uv.lock."""
    import tomllib

    out = []
    for pkg in tomllib.loads(lock.read_text(encoding="utf-8")).get("package", []):
        for wheel in pkg.get("wheels", []):
            if wheel["url"].endswith("-none-any.whl"):
                out.append((pkg["name"], wheel["url"], wheel["hash"].split(":", 1)[1]))
    return out


def simple_index_app(files: dict[str, list[tuple[Path, str]]]) -> N.App:
    """PEP 503 index: ``/simple/<name>/`` lists links (with ``#sha256=``) to ``/files/<wheel>``."""
    routes: dict[str, bytes] = {"/simple/": b"<html><body>" + b"".join(
        f'<a href="/simple/{n}/">{n}</a>'.encode() for n in files) + b"</body></html>"}
    for name, wheels in files.items():
        links = "".join(f'<a href="/files/{p.name}#sha256={sha}">{p.name}</a><br/>' for p, sha in wheels)
        page = f"<!DOCTYPE html><html><body>{links}</body></html>".encode()
        for spelling in {name, re.sub(r"[-_.]+", "-", name).lower()}:
            routes[f"/simple/{spelling}/"] = page
        for p, _ in wheels:
            routes[f"/files/{p.name}"] = p.read_bytes()
    base = N.static_app(routes)

    def app(req: N.Request) -> N.Response:
        resp = base(req)
        if resp.status == 200 and req.path.startswith("/simple/"):
            resp.headers["Content-Type"] = "text/html"
        return resp

    return app


def env_without(env: dict[str, str], *names: str) -> dict[str, str]:
    return {k: v for k, v in env.items() if k not in names}


def installed_tool(inst: Installed, name: str) -> dict:
    facts = json.loads((inst.sb.hermes_home / "tools" / "facts.json").read_text(encoding="utf-8"))
    return facts["packages"][name]


@contextlib.contextmanager
def tool_missing(inst: Installed, name: str) -> Iterator[dict]:
    """A provisioned tool's store entry is gone (pruned or deleted), so PM must fetch it again.
    The entry is parked, not deleted, and put back afterwards unless the cell re-provisioned it."""
    tool = installed_tool(inst, name)
    entry = inst.sb.hermes_home / "tools" / tool["entry"]
    parked = inst.root / "parked" / tool["entry"]
    parked.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(entry), str(parked))
    try:
        yield tool
    finally:
        if not entry.exists():
            shutil.move(str(parked), str(entry))
        else:
            shutil.rmtree(parked, ignore_errors=True)


SUCCESS_BANNERS = ("Update complete", "Code updated", "Already up to date")


def assert_nothing_changed(inst: Installed, before: dict, r: Result, what: str) -> None:
    """The failure contract: non-zero, says why, nothing reported as success, nothing moved,
    and the next ``hermes`` still starts."""
    assert r.rc != 0, f"{what}: the command reported success\n" + r.report(inst)
    claimed = [b for b in SUCCESS_BANNERS if b in r.out]
    assert not claimed, f"{what}: failed run printed a success banner {claimed}\n" + r.report(inst)
    after = inst.state()
    assert after == before, f"{what}: install state moved {before} -> {after}\n" + r.report(inst)
    v = inst.version_works()
    assert v.rc == 0 and I.TRACEBACK not in v.out, f"{what}: `hermes --version` broken afterwards\n" + I.describe(v.cp)
